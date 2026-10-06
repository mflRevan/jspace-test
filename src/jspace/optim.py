"""bf16-native optimiser.

:class:`KahanSGD` keeps parameters, momentum and a Kahan compensation buffer in
bf16. Plain bf16 updates silently drop increments smaller than ~0.4% of a
weight; the compensation buffer carries the rounding error into the next step,
so small updates accumulate correctly without fp32 master weights.
"""

from __future__ import annotations

import torch


class KahanSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.9) -> None:
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["comp"] = torch.zeros_like(p)
                m = st["m"].float().mul_(mu).add_(p.grad.float())
                st["m"].copy_(m)
                y = m.mul_(-lr).sub_(st["comp"].float())  # update minus carried rounding error
                old = p.float()
                p.copy_(old + y)  # rounds to bf16
                st["comp"].copy_((p.float() - old) - y)
