"""
Artist-mannequin / ragdoll renderer (paper Figure 1 look), driven purely by joint positions.

A smooth gray mannequin built from capsules (limbs + torso) and ball joints + an ovoid head, rendered
3/4 from the top-left with onion-skin "ghosts": past poses are left behind and fade out with age, so a
walk spreads across the floor instead of walking out of frame. Two modes:

  --mode montage : one still PNG, K evenly spaced poses fading oldest->newest, with the prompt caption.
  --mode gif     : animation where each frame trails a few fading ghosts of the recent poses.

Runs in an env with pyrender + trimesh (EGL headless) + PIL. Input = report/mesh_joints.npz (P,L,22,3).

    python render_mannequin.py --joints report/mesh_joints.npz --mode montage --out report
    python render_mannequin.py --joints report/mesh_joints.npz --mode gif     --out report
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image, ImageDraw, ImageFont

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SPINE = [(0, 3), (3, 6), (6, 9), (9, 12)]      # torso column
HEAD_J = 15
CLAY = [0.70, 0.74, 0.82]                      # cool gray mannequin, clearly above the floor tone
FLOOR = [0.88, 0.88, 0.91]


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float32), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = s; m[:3, 1] = u; m[:3, 2] = -f; m[:3, 3] = eye
    return m


def smooth_time(x, win=9):
    if not win or win <= 1:
        return x
    win = win + 1 if win % 2 == 0 else win
    sig = win / 3.0
    t = np.arange(win) - win // 2
    k = np.exp(-0.5 * (t / sig) ** 2); k /= k.sum()
    sh = x.shape; xf = x.reshape(sh[0], -1)
    xp = np.pad(xf, ((win // 2, win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(xf.shape[1])], 1).reshape(sh)


def _capsule(a, b, r):
    seg = np.linalg.norm(b - a)
    if seg < 1e-5:
        return trimesh.creation.icosphere(radius=r, subdivisions=2).apply_translation(a)
    c = trimesh.creation.cylinder(radius=r, segment=[a, b], sections=16)
    caps = [trimesh.creation.icosphere(radius=r, subdivisions=2).apply_translation(p) for p in (a, b)]
    return trimesh.util.concatenate([c] + caps)


def mannequin(P, scale=1.0):
    """Build one smooth gray mannequin mesh from 22 joint positions."""
    r_limb, r_ball, r_torso, r_head = 0.062 * scale, 0.072 * scale, 0.115 * scale, 0.135 * scale
    parts = []
    for j, p in enumerate(PARENTS):                          # limbs
        if p >= 0 and (j, p) not in SPINE and (p, j) not in SPINE:
            parts.append(_capsule(P[p], P[j], r_limb))
    for a, b in SPINE:                                       # thicker torso column
        parts.append(_capsule(P[a], P[b], r_torso))
    parts.append(_capsule(P[1], P[2], r_torso * 0.8))        # pelvis width
    parts.append(_capsule(P[16], P[17], r_torso * 0.7))      # shoulder width
    for j in [0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]:   # ball joints
        parts.append(trimesh.creation.icosphere(radius=r_ball, subdivisions=2).apply_translation(P[j]))
    head = trimesh.creation.icosphere(radius=r_head, subdivisions=3)
    head.apply_scale([0.9, 1.2, 0.95]); head.apply_translation(P[HEAD_J] + np.array([0, 0.04, 0]))
    parts.append(head)
    return trimesh.util.concatenate(parts)


def _font(size):
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont(fm.FontProperties(weight="bold")), size)
    except Exception:
        return ImageFont.load_default()


def caption(img, text, pad=0.14):
    """Add a caption strip with the prompt under the image (good-sized bold text)."""
    h, w = img.shape[:2]
    bar = int(h * pad)
    canvas = Image.new("RGB", (w, h + bar), (247, 247, 250))
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(int(bar * 0.42))
    t = '"' + text + '"'
    tw = d.textbbox((0, 0), t, font=f)[2]
    d.text(((w - tw) / 2, h + bar * 0.28), t, fill=(40, 42, 48), font=f)
    return np.array(canvas)


FOV = np.pi / 3.8


class Renderer:
    def __init__(self, res, azim, elev, dist, target_y=0.85, ss=2):
        self.r = pyrender.OffscreenRenderer(res * ss, res * ss)
        self.res, self.ss = res, ss
        az, el = np.radians(azim), np.radians(elev)
        self.target = np.array([0, target_y, 0])
        self.eye = self.target + np.array([dist * np.cos(el) * np.sin(az), dist * np.sin(el),
                                           dist * np.cos(el) * np.cos(az)])
        self.key = look_at([2.5, 4.5, 1.5], [0, 0, 0])
        self.fill = look_at([-3.0, 2.0, 1.5], [0, 0.8, 0])

    def _down(self, a):
        return np.array(Image.fromarray(a).resize((self.res, self.res), Image.LANCZOS))

    def ground(self, center):
        sc = pyrender.Scene(bg_color=[0.95, 0.95, 0.97, 1.0], ambient_light=[0.6, 0.6, 0.6])
        g = trimesh.creation.box(extents=[16, 0.02, 16]); g.apply_translation([center[0], -0.01, center[2]])
        sc.add(pyrender.Mesh.from_trimesh(g, material=_mat(FLOOR, rough=1.0)))
        self._cam(sc)
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0), pose=self.key)
        return self._down(self.r.render(sc)[0])

    def body(self, mesh):
        """Render the mannequin alone; return (rgb, mask) at output res. Low ambient -> visible 3D form."""
        sc = pyrender.Scene(bg_color=[0.95, 0.95, 0.97, 1.0], ambient_light=[0.40, 0.40, 0.44])
        sc.add(pyrender.Mesh.from_trimesh(mesh, material=_mat(CLAY), smooth=True))
        self._cam(sc)
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.4), pose=self.key)
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.4), pose=self.fill)
        col, dep = self.r.render(sc)
        return self._down(col), self._down((dep > 0).astype(np.float32) * 255)[..., None] / 255.0

    def _cam(self, sc):
        sc.add(pyrender.PerspectiveCamera(yfov=FOV), pose=look_at(self.eye, self.target))


def _mat(rgb, rough=0.75):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=rgb + [1.0], metallicFactor=0.0, roughnessFactor=rough)


def composite(bg, layers):
    """Painter's compositing: layers = [(rgb, mask, alpha)] oldest->newest over bg."""
    acc = bg.astype(np.float32).copy()
    for rgb, mask, alpha in layers:
        a = mask * alpha
        acc = acc * (1 - a) + rgb.astype(np.float32) * a
    return acc.clip(0, 255).astype(np.uint8)


def fit_view(Jc, margin=1.20):
    """Camera distance + target height so the whole motion (xz trajectory AND jump height) fits.
    Jc is already floor-dropped and xz-centered. +0.55 covers the head/arm caps above the top joint."""
    H = float(Jc[..., 1].max())
    W = float(max(Jc[..., 0].ptp(), Jc[..., 2].ptp())) + 0.9           # + body footprint
    vert = H + 0.55
    th = np.tan(FOV / 2)
    dist = max(2.5, (vert / 2) / th, (W / 2) / th) * margin
    return dist, H * 0.46 + 0.1


def render_one(J, prompt, mode, out, idx, fps, n_ghost, every, res, azim, elev, dist_override):
    L = J.shape[0]
    # recenter the whole clip so its trajectory is centered on the origin (keeps it framed)
    Jc = J.copy()
    floor = Jc[..., 1].min(); Jc[..., 1] -= floor
    ctr = np.array([Jc[..., 0].mean(), 0, Jc[..., 2].mean()]); Jc[..., 0] -= ctr[0]; Jc[..., 2] -= ctr[2]
    dist, ty = fit_view(Jc)
    R = Renderer(res, azim, elev, dist_override or dist, target_y=ty)
    bg = R.ground(np.array([0, 0, 0]))
    slug = "".join(c if c.isalnum() else "_" for c in prompt)[:28]

    if mode == "montage":
        sel = np.linspace(0, L - 1, n_ghost).astype(int)
        alphas = np.linspace(0.16, 1.0, len(sel)) ** 1.3
        alphas = 0.14 + (1 - 0.14) * (alphas - alphas.min()) / (alphas.max() - alphas.min())
        layers = [(*R.body(mannequin(Jc[f])), a) for f, a in zip(sel, alphas)]
        img = caption(composite(bg, layers), prompt)
        imageio.imwrite(os.path.join(out, f"mannequin_{idx}_{slug}.png"), img)
        print(f"wrote mannequin_{idx}_{slug}.png | {prompt}")
    else:
        frames = []
        for t in range(L):
            ages = [k for k in range(every * n_ghost, 0, -every) if t - k >= 0]   # oldest..newest ghost
            layers = []
            for k in ages:
                a = 0.10 + 0.30 * (1 - k / (every * n_ghost))
                layers.append((*R.body(mannequin(Jc[t - k])), a))
            layers.append((*R.body(mannequin(Jc[t])), 1.0))
            frames.append(caption(composite(bg, layers), prompt))
        imageio.mimsave(os.path.join(out, f"mannequin_{idx}_{slug}.gif"), frames, fps=fps)
        print(f"wrote mannequin_{idx}_{slug}.gif | {prompt}")
    R.r.delete()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default="report/mesh_joints.npz")
    ap.add_argument("--mode", choices=["montage", "gif"], default="montage")
    ap.add_argument("--res", type=int, default=540)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--azim", type=float, default=-32.0)
    ap.add_argument("--elev", type=float, default=13.0)
    ap.add_argument("--dist", type=float, default=0.0, help="0 = auto-fit to the trajectory")
    ap.add_argument("--ghosts", type=int, default=6, help="montage: # poses; gif: # trailing ghosts")
    ap.add_argument("--every", type=int, default=6, help="gif: frames between ghosts")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--only", type=int, default=-1, help="render only clip i (-1 = all)")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.joints, allow_pickle=True)
    J, prompts = d["joints"], list(d["prompts"])
    J = np.stack([smooth_time(J[i], a.smooth) for i in range(J.shape[0])])
    rot = trimesh.transformations.rotation_matrix(np.radians(180.0), [0, 1, 0])[:3, :3]
    for i in range(J.shape[0]):
        if a.only >= 0 and i != a.only:
            continue
        Ji = J[i] @ rot.T
        render_one(Ji, prompts[i], a.mode, a.out, i, a.fps, a.ghosts, a.every,
                   a.res, a.azim, a.elev, a.dist)


if __name__ == "__main__":
    main()
