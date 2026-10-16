"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import torch
from torch import nn

from fastgen.flow import finalize, sample_euler
from fastgen.unet import UNET_CONFIGS, FlowUNet
from src.utils import count_parameters, parameter_summary


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()





# Frozen per-model configs: evaluate.py constructs each class with no arguments, so the
# architecture and sampler settings of the shipped checkpoints live here.
# Both ship DMD2 generators (Yin et al., 2024) from the clean Stage 3 path in EXPERIMENTS.md:
# - one_nfe: runs/dmdA best.pt EMA 500, a one-step generator x = z - v(z, t=1) (Euler-1).
# - few_nfe: runs/dmdB best.pt EMA 500, trained by backward simulation on exactly this sampler: 4 Euler steps on the
#   shift-0.5 grid t = (1, 0.6, 0.333, 0.143, 0), i.e. one large step then refinements at low noise.
ONE_NFE_BACKBONE = dict(sigma_data=0.65, **UNET_CONFIGS["unet79"])
ONE_NFE_SAMPLER = dict(steps=1, shift=1.0, cfg_scale=1.0, cfg_interval=None)
FEW_NFE_BACKBONE = dict(sigma_data=0.65, **UNET_CONFIGS["unet79"])
FEW_NFE_SAMPLER = dict(steps=4, shift=0.5, cfg_scale=1.0, cfg_interval=None, update="euler")
SAMPLE_SEED = 0


class _FlowSamplerMixin:
    """Euler sampling of a velocity backbone with a model-owned noise generator."""

    SAMPLER: dict = {}

    def _noise(self, shape, device):
        device = torch.device(device)
        # Seeded once per device, not per call: evaluate.py calls sample() once per batch.
        if getattr(self, "_generator", None) is None or self._generator_device != device.type:
            self._generator = torch.Generator(device=device).manual_seed(SAMPLE_SEED)
            self._generator_device = device.type
        return torch.randn(shape, device=device, generator=self._generator)

    @torch.no_grad()
    def _sample(self, shape, device, category):
        z = self._noise(shape, device)
        if category is not None:
            category = category.to(device=device, dtype=torch.long)
        x = sample_euler(self.backbone, z, category, **self.SAMPLER)
        return finalize(x)


class ModelOneNFE(_FlowSamplerMixin, Model):
    """Student model evaluated with exactly one model function evaluation."""

    SAMPLER = ONE_NFE_SAMPLER

    def __init__(self, device="cpu", **kwargs):
        super().__init__(FlowUNet(**ONE_NFE_BACKBONE))

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._sample(shape, device, category)


class ModelFewNFE(_FlowSamplerMixin, Model):
    """Student model evaluated with no more than four model evaluations."""

    SAMPLER = FEW_NFE_SAMPLER

    def __init__(self, device="cpu", **kwargs):
        super().__init__(FlowUNet(**FEW_NFE_BACKBONE))

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._sample(shape, device, category)
