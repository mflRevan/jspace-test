import torch

from jspace.optim import KahanSGD


def test_kahan_sgd_tracks_fp32_reference():
    """Many tiny updates on bf16 weights must track fp32 SGD-momentum; plain bf16 must not."""
    torch.manual_seed(0)
    w0 = torch.randn(4096) * 0.05
    grads = [torch.randn(4096) for _ in range(200)]
    lr, mu = 1e-4, 0.9

    ref = w0.clone().requires_grad_(True)
    opt_ref = torch.optim.SGD([ref], lr=lr, momentum=mu)
    kah = w0.to(torch.bfloat16).requires_grad_(True)
    opt_kah = KahanSGD([kah], lr=lr, momentum=mu)
    plain = w0.to(torch.bfloat16).requires_grad_(True)
    opt_plain = torch.optim.SGD([plain], lr=lr, momentum=mu)
    for g in grads:
        for p, o in ((ref, opt_ref), (kah, opt_kah), (plain, opt_plain)):
            p.grad = g.to(p.dtype)
            o.step()
    delta = (ref - w0).detach()
    err_kah = ((kah.float() - w0) - delta).norm() / delta.norm()
    err_plain = ((plain.float() - w0) - delta).norm() / delta.norm()
    assert err_kah < 0.02, err_kah
    assert err_plain > 5 * err_kah, (err_plain, err_kah)
