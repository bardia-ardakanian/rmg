"""
Render generated motion from JOINT POSITIONS (HumanML3D FK) — no SMPL needed:
  --mode capsule  : a body-ish figure (bones as capsules + joints as spheres)
  --mode skeleton : joints (red spheres) + bones (thin cylinders)
  --mode both     : produce both
Run in an env with pyrender + trimesh (EGL headless). Input = report/mesh_joints.npz (P,L,22,3).

    python render_body.py --joints report/mesh_joints.npz --mode both --out report
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import pyrender
import imageio

# HumanML3D 22-joint kinematic tree (parents)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


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
    win = win + 1 if win % 2 == 0 else win
    sig = win / 3.0
    t = np.arange(win) - win // 2
    k = np.exp(-0.5 * (t / sig) ** 2); k /= k.sum()
    sh = x.shape
    xf = x.reshape(sh[0], -1)
    xp = np.pad(xf, ((win // 2, win // 2), (0, 0)), mode="edge")
    out = np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(xf.shape[1])], axis=1)
    return out.reshape(sh)


def _bones(P, radius):
    g = []
    for j, par in enumerate(PARENTS):
        if par >= 0 and np.linalg.norm(P[j] - P[par]) > 1e-4:
            g.append(trimesh.creation.cylinder(radius=radius, segment=[P[par], P[j]], sections=10))
    return trimesh.util.concatenate(g)


def _joints(P, radius):
    g = []
    for j in range(len(P)):
        s = trimesh.creation.icosphere(radius=radius, subdivisions=1)
        s.apply_translation(P[j]); g.append(s)
    return trimesh.util.concatenate(g)


def build(P, mode):
    """Return list of (trimesh, rgb) parts for one frame."""
    if mode == "capsule":
        body = trimesh.util.concatenate([_bones(P, 0.045), _joints(P, 0.058)])
        return [(body, [0.65, 0.74, 0.86])]
    return [(_bones(P, 0.012), [0.40, 0.45, 0.52]), (_joints(P, 0.03), [0.85, 0.25, 0.25])]


def mat(rgb):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=rgb + [1.0], metallicFactor=0.1, roughnessFactor=0.85)


def render_mode(J, prompts, mode, r, Rfix, eye, out, fps):
    P_n, L = J.shape[0], J.shape[1]
    target = [0, 0.95, 0]
    key_pose = look_at([2.5, 4.5, 1.5], [0, 0, 0])          # key light from upper-right -> ground shadow
    for p in range(P_n):
        V = J[p] @ Rfix.T
        floor = V[..., 1].min()
        frames = []
        for fr in range(L):
            pj = V[fr].copy()
            pj[:, [0, 2]] -= pj[:, [0, 2]].mean(0)
            pj[:, 1] -= floor
            scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35, 0.35, 0.35])
            for m, rgb in build(pj, mode):
                scene.add(pyrender.Mesh.from_trimesh(m, material=mat(rgb), smooth=True))
            ground = trimesh.creation.box(extents=[8, 0.02, 8]); ground.apply_translation([0, -0.01, 0])
            scene.add(pyrender.Mesh.from_trimesh(ground, material=mat([0.86, 0.86, 0.9])))
            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=look_at(eye, target))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.2), pose=look_at(eye, target))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5), pose=key_pose)
            frames.append(r.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0])
        slug = "".join(c if c.isalnum() else "_" for c in prompts[p])[:28]
        imageio.mimsave(os.path.join(out, f"{mode}_{p}_{slug}.gif"), frames, fps=fps)
        idx = np.linspace(0, L - 1, 6).astype(int)
        imageio.imwrite(os.path.join(out, f"{mode}_contact_{p}_{slug}.png"),
                        np.concatenate([frames[i] for i in idx], axis=1))
        print(f"wrote {mode}_{p}_{slug}.gif | {prompts[p]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default="report/mesh_joints.npz")
    ap.add_argument("--mode", choices=["capsule", "skeleton", "both"], default="both")
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--roty", type=float, default=0.0)
    ap.add_argument("--azim", type=float, default=-35.0, help="camera azimuth deg (negative = from the left)")
    ap.add_argument("--elev", type=float, default=22.0, help="camera elevation deg (positive = from above)")
    ap.add_argument("--dist", type=float, default=2.6, help="camera distance")
    ap.add_argument("--smooth", type=int, default=9, help="temporal smoothing window (0 = off)")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.joints, allow_pickle=True)
    J, prompts = d["joints"], list(d["prompts"])
    J = np.stack([smooth_time(J[i], a.smooth) for i in range(J.shape[0])])   # de-wobble
    r = pyrender.OffscreenRenderer(a.res, a.res)
    Rfix = trimesh.transformations.rotation_matrix(np.radians(a.roty), [0, 1, 0])[:3, :3]
    az, el = np.radians(a.azim), np.radians(a.elev)         # 3/4 view: top-left, looking down
    eye = [0 + a.dist * np.cos(el) * np.sin(az), 0.95 + a.dist * np.sin(el), a.dist * np.cos(el) * np.cos(az)]
    for mode in (["capsule", "skeleton"] if a.mode == "both" else [a.mode]):
        render_mode(J, prompts, mode, r, Rfix, eye, a.out, a.fps)
    r.delete()


if __name__ == "__main__":
    main()
