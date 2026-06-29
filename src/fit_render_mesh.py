"""Fit SMPL-X to generated joint positions (SMPLify), then render a human body-mesh GIF from a 3/4 angle
with a shadow and a visible floor. Run in an env with smplx + pyrender (EGL headless).

    python fit_render_mesh.py --joints report/mesh_joints.npz --model_path ~/smplx/models --out report
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
import smplx
import trimesh
import pyrender
import imageio


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float32), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = s; m[:3, 1] = u; m[:3, 2] = -f; m[:3, 3] = eye
    return m


def smooth_time(x, win=9):
    """Gaussian temporal smoothing along axis 0 (x: (L, ...)) to remove frame-to-frame wobble."""
    if not win or win <= 1:
        return x
    win = win + 1 if win % 2 == 0 else win          # kernel must be odd to keep the frame count
    sig = win / 3.0
    t = np.arange(win) - win // 2
    k = np.exp(-0.5 * (t / sig) ** 2); k /= k.sum()
    sh = x.shape
    xf = x.reshape(sh[0], -1)
    xp = np.pad(xf, ((win // 2, win // 2), (0, 0)), mode="edge")
    out = np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(xf.shape[1])], axis=1)
    return out.reshape(sh)


def fit(bm, target, dev, iters=500):
    """target: (L,22,3) HumanML3D joints -> optimize SMPL-X params so joints[:22] match."""
    L = target.shape[0]
    go = torch.zeros(L, 3, device=dev, requires_grad=True)
    bp = torch.zeros(L, 63, device=dev, requires_grad=True)
    transl = target[:, 0].clone().detach().requires_grad_(True)
    betas = torch.zeros(1, 10, device=dev, requires_grad=True)
    opt = torch.optim.Adam([go, bp, transl, betas], lr=0.05)
    for it in range(iters):
        if it == int(iters * 0.6):
            for g in opt.param_groups:
                g["lr"] = 0.01
        out = bm(global_orient=go, body_pose=bp, transl=transl, betas=betas.expand(L, -1))
        data = ((out.joints[:, :22] - target) ** 2).sum(-1).mean()
        smooth = ((bp[1:] - bp[:-1]) ** 2).mean() + 0.1 * ((transl[1:] - transl[:-1]) ** 2).mean()
        loss = data + 0.05 * smooth + 5e-4 * (bp ** 2).mean() + 1e-3 * (betas ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = bm(global_orient=go, body_pose=bp, transl=transl, betas=betas.expand(L, -1))
        mpjpe = ((out.joints[:, :22] - target) ** 2).sum(-1).sqrt().mean().item()
    return out.vertices.detach().cpu().numpy(), bm.faces, mpjpe


def mat(rgb, rough=0.7):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=rgb + [1.0], metallicFactor=0.0, roughnessFactor=rough)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default="report/mesh_joints.npz")
    ap.add_argument("--model_path", default=os.path.expanduser("~/smplx/models"))
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--roty", type=float, default=180.0)
    ap.add_argument("--azim", type=float, default=-35.0)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--dist", type=float, default=2.7)
    ap.add_argument("--smooth", type=int, default=9, help="temporal smoothing window (0 = off)")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.joints, allow_pickle=True)
    prompts = list(d["prompts"]); J = d["joints"]
    J = np.stack([smooth_time(J[i], a.smooth) for i in range(J.shape[0])])   # de-wobble
    P, L = J.shape[0], J.shape[1]
    bm = smplx.create(a.model_path, model_type="smplx", gender="neutral", ext="npz",
                      use_pca=False, flat_hand_mean=True, batch_size=L).to(dev)
    r = pyrender.OffscreenRenderer(a.res, a.res)
    Rfix = trimesh.transformations.rotation_matrix(np.radians(a.roty), [0, 1, 0])[:3, :3]
    az, el = np.radians(a.azim), np.radians(a.elev)
    eye = [a.dist * np.cos(el) * np.sin(az), 0.95 + a.dist * np.sin(el), a.dist * np.cos(el) * np.cos(az)]
    cam_target = [0, 0.95, 0]
    key_pose = look_at([2.5, 4.5, 1.5], [0, 0, 0])

    for p in range(P):
        V, F, mpjpe = fit(bm, torch.tensor(J[p], dtype=torch.float32, device=dev), dev, a.iters)
        V = V @ Rfix.T
        floor_y = V[..., 1].min()
        frames = []
        for fr in range(L):
            v = V[fr].copy()
            v[:, [0, 2]] -= v[:, [0, 2]].mean(0)
            v[:, 1] -= floor_y
            scene = pyrender.Scene(bg_color=[0.93, 0.93, 0.96, 1.0], ambient_light=[0.4, 0.4, 0.4])
            scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v, F), smooth=True,
                      material=mat([0.74, 0.75, 0.78])))                       # gray clay mannequin (paper look)
            ground = trimesh.creation.box(extents=[10, 0.02, 10]); ground.apply_translation([0, -0.01, 0])
            scene.add(pyrender.Mesh.from_trimesh(ground, material=mat([0.42, 0.45, 0.52], rough=1.0)))
            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=look_at(eye, cam_target))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.2), pose=look_at(eye, cam_target))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5), pose=key_pose)
            frames.append(r.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0])
        slug = "".join(c if c.isalnum() else "_" for c in prompts[p])[:28]
        imageio.mimsave(os.path.join(a.out, f"humanmesh_{p}_{slug}.gif"), frames, fps=a.fps)
        idx = np.linspace(0, L - 1, 6).astype(int)
        imageio.imwrite(os.path.join(a.out, f"humanmesh_contact_{p}_{slug}.png"),
                        np.concatenate([frames[i] for i in idx], axis=1))
        print(f"wrote humanmesh_{p}_{slug}.gif | {prompts[p]} | fit MPJPE {mpjpe*1000:.1f}mm")
    r.delete()


if __name__ == "__main__":
    main()
