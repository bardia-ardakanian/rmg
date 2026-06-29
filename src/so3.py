"""
SO(3) geometry for flow matching — pure PyTorch, batched over arbitrary leading dims.
(Same verified module as the single-joint sanity check; see its README for the errors log.)

Run `python so3.py` for the self-tests.
"""
import math

import torch


def skew(v):
    o = torch.zeros_like(v[..., 0])
    r0 = torch.stack([o, -v[..., 2], v[..., 1]], -1)
    r1 = torch.stack([v[..., 2], o, -v[..., 0]], -1)
    r2 = torch.stack([-v[..., 1], v[..., 0], o], -1)
    return torch.stack([r0, r1, r2], -2)


def exp(omega):
    """Axis-angle (...,3) -> rotation matrix (...,3,3). Gradient-safe at omega=0 (Taylor coeffs)."""
    theta = omega.norm(dim=-1, keepdim=True)
    theta2 = theta * theta
    small = theta < 1e-4
    A = torch.where(small, 1 - theta2 / 6.0, torch.sin(theta) / theta.clamp_min(1e-4))
    B = torch.where(small, 0.5 - theta2 / 24.0, (1 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    K = skew(omega)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    return I + A[..., None] * K + B[..., None] * (K @ K)


def log(R, eps=1e-12):
    """Rotation matrix (...,3,3) -> axis-angle (...,3). Robust for ALL angles in [0, pi].

    The angle is taken from the well-conditioned atan2 form (same as geodesic_dist), which stays
    accurate near theta=pi where arccos((tr-1)/2) loses precision. The axis comes from the
    antisymmetric part (R - R^T) everywhere except a narrow band around pi, where that part vanishes
    and the axis is instead recovered from the symmetric part M=(R+I)/2 ~ a a^T (sign aligned with the
    residual antisymmetric part; sign is genuinely arbitrary exactly at pi since Exp(pi*a)=Exp(-pi*a)).
    Computed in fp64 internally, then cast back. Both branches are always evaluated (vectorized,
    autograd-safe: torch.where routes the gradient to the selected branch only).

    Note: for an fp32 *input* matrix the near-pi axis is limited to ~1e-3 accuracy — information lost
    in the fp32 rounding of R cannot be recovered. This is the geometric resolution limit there, not a
    bug; away from pi the result is exact to machine precision."""
    in_dtype = R.dtype
    Rd = R.double()
    I = torch.eye(3, device=Rd.device, dtype=Rd.dtype).expand_as(Rd)
    tr = Rd[..., 0, 0] + Rd[..., 1, 1] + Rd[..., 2, 2]
    vee = torch.stack([Rd[..., 2, 1] - Rd[..., 1, 2],
                       Rd[..., 0, 2] - Rd[..., 2, 0],
                       Rd[..., 1, 0] - Rd[..., 0, 1]], -1)              # = 2 sin(theta) * axis
    nvee = vee.norm(dim=-1, keepdim=True)                              # = 2|sin(theta)|
    theta = torch.atan2(0.5 * nvee.squeeze(-1), (tr - 1) * 0.5)        # robust angle in [0, pi]

    # axis from the antisymmetric part; theta * (unit axis) IS the log. Correct at small angle too
    # (theta*vee/||vee|| -> vee/2 as theta->0; ->0 at identity).
    out_std = theta[..., None] * (vee / nvee.clamp_min(eps))

    # axis from the symmetric part: dominant column of M=(R+I)/2 is a_k*a -> normalize
    M = 0.5 * (Rd + I)
    diag = torch.diagonal(M, dim1=-2, dim2=-1)                          # (...,3) ~ a_i^2
    k = diag.argmax(dim=-1)
    col = torch.gather(M, -1, k[..., None, None].expand(*k.shape, 3, 1)).squeeze(-1)
    axis = col / col.norm(dim=-1, keepdim=True).clamp_min(eps)
    s = torch.sign((axis * vee).sum(-1, keepdim=True))                  # sign from residual asym part
    axis = axis * torch.where(s == 0, torch.ones_like(s), s)
    out_pi = theta[..., None] * axis

    near_pi = (theta > (math.pi - 2e-4))[..., None]                     # ~optimal antisym/symmetric crossover
    return torch.where(near_pi, out_pi, out_std).to(in_dtype)


def geodesic(R0, R1, t):
    omega = log(R0.transpose(-1, -2) @ R1)
    scaled = omega * (t[..., None] if torch.is_tensor(t) else t)
    return R0 @ exp(scaled)


def riemannian_gaussian(shape, sigma, device=None, dtype=torch.float32):
    v = torch.randn(*shape, 3, device=device, dtype=dtype) * sigma
    return exp(v)


def geodesic_dist(R1, R2):
    """atan2 form: accurate for small AND large angles."""
    Rrel = R1.transpose(-1, -2) @ R2
    vee = torch.stack([Rrel[..., 2, 1] - Rrel[..., 1, 2],
                       Rrel[..., 0, 2] - Rrel[..., 2, 0],
                       Rrel[..., 1, 0] - Rrel[..., 0, 1]], -1)
    sin_t = vee.norm(dim=-1) * 0.5
    cos_t = (Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2] - 1) * 0.5
    return torch.atan2(sin_t, cos_t)


def matrix_to_6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def sixd_to_matrix(d, eps=1e-5):
    a1, a2 = d[..., :3], d[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _selftest():
    torch.manual_seed(0)
    dt = torch.float64
    omega = torch.randn(4000, 3, dtype=dt)
    omega = omega / omega.norm(dim=-1, keepdim=True) * torch.empty(4000, 1, dtype=dt).uniform_(0, 2.5)
    R = exp(omega); R1 = exp(omega.flip(0)); I = torch.eye(3, dtype=dt).expand_as(R)
    checks = {
        "log(exp(w)) == w":        (log(R) - omega).abs().max().item(),
        "exp(log(R)) == R":        (exp(log(R)) - R).abs().max().item(),
        "||log|| == dist(I,R)":    (log(R).norm(dim=-1) - geodesic_dist(I, R)).abs().max().item(),
        "geodesic(0)==R0":         (geodesic(R, R1, 0.0) - R).abs().max().item(),
        "geodesic(1)==R1":         (geodesic(R, R1, 1.0) - R1).abs().max().item(),
        "6d round-trip":           (sixd_to_matrix(matrix_to_6d(R)) - R).abs().max().item(),
    }
    tol = {k: 1e-5 for k in checks}

    # antipodal stress: angles within 1e-4 rad of pi (the previously-broken regime), fp64 + fp32.
    # exp(log(R))==R and ||log||==dist are sign-invariant, so they must hold even where the axis sign
    # is ambiguous (exactly pi). log(exp(w))==w is checked just below pi where the sign is determined.
    for dtype2, t in [(torch.float64, 1e-3), (torch.float32, 5e-3)]:
        tag = "fp64" if dtype2 == torch.float64 else "fp32"
        ax = torch.randn(3000, 3, dtype=torch.float64); ax /= ax.norm(dim=-1, keepdim=True)
        ang = torch.empty(3000, 1, dtype=torch.float64).uniform_(math.pi - 1e-4, math.pi)
        Rp = exp(ax * ang).to(dtype2); Ip = torch.eye(3, dtype=dtype2).expand_as(Rp)
        checks[f"exp(log(R))==R near-pi [{tag}]"] = (exp(log(Rp)).double() - Rp.double()).abs().max().item()
        checks[f"||log||==dist near-pi [{tag}]"] = (log(Rp).norm(dim=-1).double() - geodesic_dist(Ip, Rp).double()).abs().max().item()
        tol[f"exp(log(R))==R near-pi [{tag}]"] = t
        tol[f"||log||==dist near-pi [{tag}]"] = t
    wlo = ax * ang.clamp_max(math.pi - 5e-3)                            # below the near-pi band -> sign defined
    checks["log(exp(w))==w just-below-pi"] = (log(exp(wlo)) - wlo).abs().max().item()
    tol["log(exp(w))==w just-below-pi"] = 1e-3

    ok = True
    print("SO(3) self-tests:")
    for k, e in checks.items():
        p = e < tol[k]; ok &= p
        print(f"  [{'PASS' if p else 'FAIL'}]  {k:34s} max_err={e:.2e}  (tol {tol[k]:.0e})")
    return ok


if __name__ == "__main__":
    if not _selftest():
        raise SystemExit(1)
