"""DMD2 training: distribution matching distillation of the flow teacher into a 1-step (or few-step) generator.

Yin et al. (2024), "Improved Distribution Matching Distillation for Fast Image Synthesis": a frozen real score
(our guided teacher), a fake score trained online on generator samples with a two time-scale update rule, and a
diffusion-GAN classifier on the fake score's bottleneck trained against real train images. The generator starts
from the 2-rectified flow student, so it is a one-step map from the first step.

    python train.py dmd --run_dir runs/dmd1_main [options]
With --steps N > 1 the generator is trained on the N-step grid with backward simulation (DMD2, Sec. 4.5): each
micro-batch runs the generator without gradients up to a random step k, using the RF++ update
x_{t'} = (1 - t') x_hat + t' z (the ModelFewNFE sampler), and trains the step-k prediction.

Snapshots use the train_fm format (generator weights only), so eval_fm.py and export_ckpt.py read them as-is.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch._dynamo
from torchvision.utils import save_image

from fastgen.data import DEFAULT_CACHE, GPUData, augment, balanced_classes, grid_classes, load_cache, to_eval_uint8
from fastgen.dmd import DM_RANGE, DiscHead, add_noise, dm_loss, gan_d_loss, gan_g_loss, guided_velocity, karras_t
from fastgen.ema import EMA
from fastgen.fid import FIDMonitor, memorization_ratio
from fastgen.flow import finalize, fm_loss, sample_euler, time_grid
from fastgen.train_fm import FID_BATCH, VRAM_LIMIT_BYTES, CSVLog, atomic_save
from fastgen.unet import NUM_CLASSES, UNET_CONFIGS, FlowUNet
from src.utils import count_parameters

PARAM_LIMIT = 100_000_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DMD2 distillation of the flow teacher")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--arch", default="unet79", choices=sorted(UNET_CONFIGS))
    p.add_argument("--sigma_data", type=float, default=0.65)
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--gen_init", default="runs/rf2_main/snapshots/kimg_060000.pt")
    p.add_argument("--gen_init_ema", default="2000",
                   help="EMA key, 'raw', or 'best' (the weights a DMD best.pt was saved for)")
    p.add_argument("--teacher", default="runs/fm_main/snapshots/kimg_030000.pt", help="real score; also fake init")
    p.add_argument("--teacher_ema", default="2000")
    p.add_argument("--fake_from", default=None, help="DMD latest.pt to take the fake score + GAN head from")
    p.add_argument("--cfg", type=float, default=1.5, help="real-score guidance scale")
    p.add_argument("--interval", default="0.2,0.85", help="guidance interval 'lo,hi' or 'none'")
    p.add_argument("--steps", type=int, default=1, help="generator steps (> 1: backward simulation)")
    p.add_argument("--shift", type=float, default=1.0, help="time-grid shift for --steps > 1")
    p.add_argument("--update", choices=["rfpp", "euler"], default="rfpp",
                   help="multi-step update between generator steps (backward simulation and evaluation)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--micro_batch", type=int, default=64)
    p.add_argument("--lr_gen", type=float, default=2e-6)
    p.add_argument("--lr_fake", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_iters", type=int, default=500, help="linear lr warm-up (iterations = fake updates)")
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--ttur", type=int, default=5, help="fake-score updates per generator update")
    p.add_argument("--fake_warmup_iters", type=int, default=500, help="fake-only iterations before the first G update")
    p.add_argument("--gan_d_weight", type=float, default=1e-2)
    p.add_argument("--gan_g_weight", type=float, default=3e-3)
    p.add_argument("--total_kimg", type=float, default=100000, help="generator images (kimg)")
    p.add_argument("--time_budget_h", type=float, default=None, help="stop after this much pure training time")
    p.add_argument("--ema_halflives_kimg", default="50,200,500")
    p.add_argument("--fid_every_kimg", type=float, default=250)
    p.add_argument("--snapshot_every_kimg", type=float, default=1000)
    p.add_argument("--fid_per_class", type=int, default=20)
    p.add_argument("--log_every", type=int, default=50, help="iterations")
    p.add_argument("--latest_every_min", type=float, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def build_net(arch, sigma_data):
    # Dropout off: the fake score trains on unlimited generator samples, and the generator is evaluated in eval mode.
    return FlowUNet(sigma_data=sigma_data, **dict(UNET_CONFIGS[arch], dropout=0.0))


def load_weights(net, snapshot, key):
    snap = torch.load(snapshot, map_location="cpu", weights_only=False)
    if key == "best":  # a DMD best.pt records which of raw / EMAs scored the best FID
        key = snap["dmd"]["fid_key"]
        print(f"{snapshot}: best = {key!r} (FID {snap['dmd']['fid']:.2f})")
    weights = snap["raw"] if key == "raw" else snap["ema"][key]
    net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in weights.items()})


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 32  # several FlowUNet instances and batch shapes share one forward
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    latest_path = run_dir / "latest.pt"
    if latest_path.exists() and not args.resume:
        raise SystemExit(f"{latest_path} exists; pass --resume or use a new --run_dir")
    resuming = args.resume and latest_path.exists()
    assert args.batch % args.micro_batch == 0
    accum = args.batch // args.micro_batch
    interval = None if args.interval == "none" else tuple(float(v) for v in args.interval.split(","))
    halflives = [float(h) for h in args.ema_halflives_kimg.split(",")]
    ts = time_grid(args.steps, args.shift).tolist()  # generator times; [1, 0] for one step

    cache = load_cache(args.cache)
    data = GPUData(cache, device)

    gen_net = build_net(args.arch, args.sigma_data)
    fake = build_net(args.arch, args.sigma_data)
    real = build_net(args.arch, args.sigma_data)
    head = DiscHead(UNET_CONFIGS[args.arch]["model_channels"] * UNET_CONFIGS[args.arch]["channel_mult"][-1])
    n_gen = count_parameters(gen_net, trainable_only=False)
    n_fake = count_parameters(fake, trainable_only=False) + count_parameters(head, trainable_only=False)
    print(f"generator {n_gen:,} params; fake score + GAN head {n_fake:,} params")
    assert n_gen <= PARAM_LIMIT and n_fake <= PARAM_LIMIT
    load_weights(real, args.teacher, args.teacher_ema)
    if not resuming:
        load_weights(gen_net, args.gen_init, args.gen_init_ema)
        if args.fake_from:
            ckpt = torch.load(args.fake_from, map_location="cpu", weights_only=False)
            fake.load_state_dict(ckpt["fake"])
            head.load_state_dict(ckpt["head"])
            del ckpt
        else:
            load_weights(fake, args.teacher, args.teacher_ema)
    for net in (gen_net, fake, real, head):
        net.to(device).to(memory_format=torch.channels_last)
    real.eval().requires_grad_(False)

    fake_params = list(fake.parameters()) + (list(head.parameters()) if args.gan_d_weight > 0 else [])
    gen_params = list(gen_net.parameters())
    opt_gen = torch.optim.AdamW(gen_params, lr=args.lr_gen, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    opt_fake = torch.optim.AdamW(fake_params, lr=args.lr_fake, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    emas = [EMA(gen_net, hl, rampup_ratio=None) for hl in halflives]  # start at the (already good) init
    gen = torch.Generator(device=device).manual_seed(args.seed)

    state = {"it": 0, "gstep": 0, "nimg": 0, "fake_nimg": 0, "train_sec": 0.0, "wall_sec": 0.0}
    if resuming:
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        gen_net.load_state_dict(ckpt["gen"])
        fake.load_state_dict(ckpt["fake"])
        head.load_state_dict(ckpt["head"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        opt_fake.load_state_dict(ckpt["opt_fake"])
        for ema, sd in zip(emas, ckpt["emas"]):
            ema.load_state_dict(sd)
        gen.set_state(ckpt["rng"].cpu())
        state = ckpt["state"]
        del ckpt
        print(f"Resumed from {latest_path} at iteration {state['it']}, {state['nimg'] / 1e3:.0f} generator kimg")
    else:
        config = vars(args) | {"num_params_gen": n_gen, "num_params_fake_head": n_fake,
                               "unet_config": UNET_CONFIGS[args.arch], "time_grid": ts}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    compile_ = (lambda f: f) if args.no_compile else (lambda f: torch.compile(f, dynamic=False))
    c_gen, c_fake, c_real = compile_(gen_net), compile_(fake), compile_(real)

    def disc(x, t, y):
        return head(fake.bottleneck(x, t, class_labels=y))

    c_disc = compile_(disc)
    eval_net = build_net(args.arch, args.sigma_data).to(device).to(memory_format=torch.channels_last).eval()
    eval_net.requires_grad_(False)
    c_eval = compile_(eval_net)

    monitor = FIDMonitor(device, num_workers=4)
    monitor.load_or_build_reference(cache)
    grid_y = grid_classes().to(device)
    grid_noise = torch.randn(len(grid_y), 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(123))
    fid_y = balanced_classes(per_class=args.fid_per_class).to(device)
    fid_noise = torch.randn(len(fid_y), 3, 64, 64, device=device, generator=torch.Generator(device).manual_seed(1234))
    sampler = dict(steps=args.steps, shift=args.shift, update=args.update if args.steps > 1 else "euler")
    sampler_name = f"euler{args.steps}" + (f"_{args.update}_s{args.shift:g}" if args.steps > 1 else "")

    train_log = CSVLog(run_dir / "log.csv", [
        "it", "gstep", "kimg", "fake_kimg", "fake_loss", "d_loss", "logit_real", "logit_fake", "d_acc", "dm_loss",
        "dm_grad_rms", "g_gan_loss", "gn_fake", "gn_gen", "lr_gen", "lr_fake", "gen_img_per_sec", "it_per_sec",
        "peak_mem_gib", "train_h", "wall_h"])
    fid_log = CSVLog(run_dir / "fid.csv", ["it", "gstep", "kimg", "train_h", "ema_kimg", "sampler", "nfe", "fid",
                                           "memorized"])

    def sample(model, noise, labels, batch):
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(labels), batch):
                out.append(finalize(sample_euler(model, noise[i:i + batch], labels[i:i + batch], **sampler).float()))
        return torch.cat(out)

    def run_fid():
        t0 = time.time()
        kimg = state["nimg"] / 1000
        best = None
        models = [("raw", gen_net)] + [(f"{hl:g}", ema) for hl, ema in zip(halflives, emas)]
        if state["gstep"] == 0:
            models = models[:1]  # every EMA equals the init
        for key, model in models:
            if key == "raw":
                eval_net.load_state_dict(gen_net.state_dict())
            else:
                model.copy_to(eval_net)
            x = sample(c_eval, fid_noise, fid_y, FID_BATCH)
            fid = monitor.fid(to_eval_uint8(x))
            mem = memorization_ratio(x, data.images)
            fid_log.write({"it": state["it"], "gstep": state["gstep"], "kimg": f"{kimg:.0f}",
                           "train_h": f"{state['train_sec'] / 3600:.3f}", "ema_kimg": key, "sampler": sampler_name,
                           "nfe": args.steps, "fid": f"{fid:.3f}", "memorized": f"{mem:.4f}"})
            contrast = x.float().std(dim=(1, 2, 3)).mean().item()  # train images: 0.476
            print(f"[fid] kimg {kimg:.0f} {key}: FID {fid:.2f}, memorized {mem:.3%}, contrast {contrast:.3f}",
                  flush=True)
            if best is None or fid < best[0]:
                best = (fid, key)
        # Keep the best evaluation so far (all of raw / EMAs) in best.pt, so a late drift can't lose it.
        if args.snapshot_every_kimg > 0 and state["gstep"] > 0 and best[0] < state.get("best_fid", math.inf):
            state["best_fid"] = best[0]
            save_snapshot(run_dir / "best.pt", {"fid": best[0], "fid_key": best[1]})
        # Grid from the best model at this evaluation.
        key = best[1]
        if key == "raw":
            eval_net.load_state_dict(gen_net.state_dict())
        else:
            emas[[f"{hl:g}" for hl in halflives].index(key)].copy_to(eval_net)
        x = sample(c_eval, grid_noise, grid_y, len(grid_y))
        save_image(to_eval_uint8(x).float() / 255, run_dir / "samples" / f"kimg{round(kimg):06d}_{key}.png", nrow=8)
        print(f"[fid] eval took {time.time() - t0:.0f}s", flush=True)

    def save_snapshot(path=None, extra=None):
        kimg = round(state["nimg"] / 1000)
        path = path or run_dir / "snapshots" / f"kimg_{kimg:06d}.pt"
        snap = {"arch": args.arch, "sigma_data": args.sigma_data, "unet_config": UNET_CONFIGS[args.arch],
                "nimg": state["nimg"], "step": state["gstep"], "raw": gen_net.state_dict(),
                "ema": {f"{hl:g}": ema.state_dict() for hl, ema in zip(halflives, emas)},
                "dmd": {"steps": args.steps, "shift": args.shift, "update": args.update, "time_grid": ts, **(extra or {})}}
        atomic_save(snap, path)
        print(f"[snapshot] kimg {kimg} -> {path.name}", flush=True)

    def save_latest():
        atomic_save({"gen": gen_net.state_dict(), "fake": fake.state_dict(), "head": head.state_dict(),
                     "opt_gen": opt_gen.state_dict(), "opt_fake": opt_fake.state_dict(),
                     "emas": [e.state_dict() for e in emas], "rng": gen.get_state(), "state": state,
                     "args": vars(args)}, latest_path)

    def crossed(prev, cur, every_kimg):
        every = every_kimg * 1000
        return every > 0 and math.floor(cur / every) > math.floor(prev / every)

    def generator_out(z, y, grad):
        """Generator sample x_hat_0 (float32). Multi-step: backward simulation from a random step k."""
        k = 0 if args.steps == 1 else int(torch.randint(args.steps, (1,), device=device, generator=gen))
        x = z
        with torch.no_grad():
            for i in range(k):
                v = c_gen(x, z.new_full((len(z),), ts[i]), class_labels=y).float()
                if args.update == "rfpp":
                    x = (1 - ts[i + 1]) * (x - ts[i] * v) + ts[i + 1] * z
                else:
                    x = x + (ts[i + 1] - ts[i]) * v
        with torch.set_grad_enabled(grad):
            return x - ts[k] * c_gen(x, z.new_full((len(z),), ts[k]), class_labels=y).float()

    def randn(shape):
        return torch.randn(shape, device=device, generator=gen).contiguous(memory_format=torch.channels_last)

    if state["it"] == 0:
        run_fid()  # must reproduce the init's 1-NFE FID (2-RF student: 47.49)
    total_nimg = args.total_kimg * 1000
    last_latest = time.time()
    wall_start = time.time() - state["wall_sec"]
    acc = {}
    acc_t0, acc_it, acc_gimg, acc_sec = time.time(), 0, 0, 0.0
    bad = 0
    last_eval = {"fid_nimg": state["nimg"] if state["it"] == 0 else None, "snapshot_nimg": None}
    gen_net.train()
    fake.train()
    shape = (args.micro_batch, 3, 64, 64)

    def add(key, value):  # running sums stay on the GPU until logging (no per-step syncs)
        acc[key] = acc.get(key, 0.0) + value
        acc["n_" + key] = acc.get("n_" + key, 0) + 1

    while state["nimg"] < total_nimg:
        if args.time_budget_h is not None and state["train_sec"] >= args.time_budget_h * 3600:
            print("Time budget reached")
            break
        step_t0 = time.time()
        g_iter = state["it"] - args.fake_warmup_iters
        gen_turn = g_iter >= 0 and g_iter % args.ttur == 0
        lr_fake = args.lr_fake * min(1.0, (state["it"] + 1) / max(args.warmup_iters, 1))
        lr_gen = args.lr_gen * min(1.0, max(g_iter + 1, 0) / max(args.warmup_iters, 1))
        for group in opt_fake.param_groups:
            group["lr"] = lr_fake
        for group in opt_gen.param_groups:
            group["lr"] = lr_gen

        labels = [torch.randint(NUM_CLASSES, (args.micro_batch,), device=device, generator=gen) for _ in range(accum)]
        samples = [None] * accum
        ok = True

        # Generator turn: distribution matching + GAN generator loss; gradients only into the generator.
        if gen_turn:
            for m in range(accum):
                y = labels[m]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x_g = generator_out(randn(shape), y, grad=True)
                    t = karras_t(args.micro_batch, device, gen, *DM_RANGE)
                    with torch.no_grad():
                        x_t = add_noise(x_g.detach(), t, randn(shape))
                        tb = t[:, None, None, None]
                        x_real = x_t - tb * guided_velocity(c_real, x_t, t, y, args.cfg, interval)
                        x_fake = x_t - tb * c_fake(x_t, t, class_labels=y).float()
                    loss_dm, g_rms = dm_loss(x_g, x_real, x_fake)
                    loss = loss_dm
                    if args.gan_g_weight > 0:
                        t_gan = karras_t(args.micro_batch, device, gen)
                        loss_g_gan = gan_g_loss(c_disc(add_noise(x_g, t_gan, randn(shape)), t_gan, y).float())
                        loss = loss + args.gan_g_weight * loss_g_gan
                        add("g_gan_loss", loss_g_gan.detach())
                (loss / accum).backward(inputs=gen_params)
                samples[m] = x_g.detach()
                add("dm_loss", loss_dm.detach())
                add("dm_grad_rms", g_rms.mean())
            gn_gen = torch.nn.utils.clip_grad_norm_(gen_params, args.grad_clip)
            if torch.isfinite(gn_gen):
                opt_gen.step()
                add("gn_gen", gn_gen)
            else:
                ok = False
            opt_gen.zero_grad(set_to_none=True)

        # Fake-score turn: flow-matching loss on generator samples + diffusion-GAN classifier loss.
        for m in range(accum):
            y = labels[m]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if samples[m] is None:
                    samples[m] = generator_out(randn(shape), y, grad=False)
                x_g = samples[m].contiguous(memory_format=torch.channels_last)
                t = karras_t(args.micro_batch, device, gen)
                per_sample, _ = fm_loss(c_fake, x_g, t, randn(shape), y, None, args.sigma_data)
                loss_fake = per_sample.mean()
                loss = loss_fake
                if args.gan_d_weight > 0:
                    x_r, _ = augment(data.class_batch(y, gen), 0.0, gen)  # h-flip only
                    x_both = torch.cat([x_g, x_r]).contiguous(memory_format=torch.channels_last)
                    t_gan = karras_t(2 * args.micro_batch, device, gen)
                    logits = c_disc(add_noise(x_both, t_gan, randn((2 * args.micro_batch, 3, 64, 64))), t_gan,
                                    torch.cat([y, y])).float()
                    logit_fake, logit_real = logits.chunk(2)
                    loss_d = gan_d_loss(logit_real, logit_fake)
                    loss = loss + args.gan_d_weight * loss_d
                    add("d_loss", loss_d.detach())
                    add("logit_real", logit_real.detach().mean())
                    add("logit_fake", logit_fake.detach().mean())
                    add("d_acc", ((logit_real > 0).float().mean() + (logit_fake < 0).float().mean()).detach() / 2)
            (loss / accum).backward(inputs=fake_params)
            add("fake_loss", loss_fake.detach())
        gn_fake = torch.nn.utils.clip_grad_norm_(fake_params, args.grad_clip)
        if torch.isfinite(gn_fake):
            opt_fake.step()
            add("gn_fake", gn_fake)
        else:
            ok = False
        opt_fake.zero_grad(set_to_none=True)

        if not ok:
            bad += 1
            print(f"Non-finite gradient at iteration {state['it']} ({bad} in a row)", flush=True)
            if bad >= 10:
                save_latest()
                raise SystemExit("Too many non-finite steps; aborting")
        else:
            bad = 0
        prev_nimg = state["nimg"]
        state["it"] += 1
        state["fake_nimg"] += args.batch
        if gen_turn and ok:
            state["gstep"] += 1
            state["nimg"] += args.batch
            for ema in emas:
                ema.update(gen_net, state["nimg"], args.batch)
            acc_gimg += args.batch
        step_sec = time.time() - step_t0
        state["train_sec"] += step_sec
        acc_sec += step_sec
        acc_it += 1

        if state["it"] % args.log_every == 0:
            state["wall_sec"] = time.time() - wall_start
            mean = {k: float(acc[k]) / acc["n_" + k] for k in acc if not k.startswith("n_")}
            peak = torch.cuda.max_memory_allocated()
            row = {"it": state["it"], "gstep": state["gstep"], "kimg": f"{state['nimg'] / 1000:.1f}",
                   "fake_kimg": f"{state['fake_nimg'] / 1000:.1f}",
                   **{k: f"{v:.5f}" for k, v in mean.items()},
                   "lr_gen": f"{lr_gen:.2e}", "lr_fake": f"{lr_fake:.2e}",
                   "gen_img_per_sec": f"{acc_gimg / max(acc_sec, 1e-9):.1f}",
                   "it_per_sec": f"{acc_it / max(acc_sec, 1e-9):.3f}", "peak_mem_gib": f"{peak / 2 ** 30:.2f}",
                   "train_h": f"{state['train_sec'] / 3600:.3f}", "wall_h": f"{state['wall_sec'] / 3600:.3f}"}
            train_log.write(row)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            if peak > VRAM_LIMIT_BYTES:
                print(f"WARNING: peak memory {peak / 1e9:.2f} GB exceeds the 20 GB limit", flush=True)
            acc = {}
            acc_t0, acc_it, acc_gimg, acc_sec = time.time(), 0, 0, 0.0

        if crossed(prev_nimg, state["nimg"], args.fid_every_kimg):
            run_fid()
            last_eval["fid_nimg"] = state["nimg"]
        if crossed(prev_nimg, state["nimg"], args.snapshot_every_kimg):
            save_snapshot()
            last_eval["snapshot_nimg"] = state["nimg"]
        if time.time() - last_latest > args.latest_every_min * 60:
            state["wall_sec"] = time.time() - wall_start
            save_latest()
            last_latest = time.time()

    state["wall_sec"] = time.time() - wall_start
    if last_eval["fid_nimg"] != state["nimg"]:
        run_fid()
    if args.snapshot_every_kimg > 0 and last_eval["snapshot_nimg"] != state["nimg"]:  # pilots: latest.pt only
        save_snapshot()
    save_latest()
    summary = {"it": state["it"], "gstep": state["gstep"], "kimg": state["nimg"] / 1000,
               "train_h": state["train_sec"] / 3600, "wall_h": state["wall_sec"] / 3600,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
