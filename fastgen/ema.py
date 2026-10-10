"""Exponential moving averages with half-life measured in training images (EDM-style)."""

import torch


class EMA:
    def __init__(self, model, halflife_kimg, rampup_ratio=0.05):
        self.halflife_kimg = halflife_kimg
        self.rampup_ratio = rampup_ratio
        self.names = [name for name, _ in model.named_parameters()]
        self.params = [p.detach().clone().float() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model, cur_nimg, batch_size):
        halflife = self.halflife_kimg * 1000
        if self.rampup_ratio is not None:
            halflife = min(halflife, cur_nimg * self.rampup_ratio)
        beta = 0.5 ** (batch_size / max(halflife, 1e-8))
        torch._foreach_lerp_(self.params, [p.detach().float() for p in model.parameters()], 1 - beta)

    def state_dict(self):
        return {name: p for name, p in zip(self.names, self.params)}

    def load_state_dict(self, state):
        for name, p in zip(self.names, self.params):
            p.copy_(state[name])

    @torch.no_grad()
    def copy_to(self, model):
        for p_model, p_ema in zip(model.parameters(), self.params):
            p_model.copy_(p_ema)
