"""
Render the rigged robot mannequin (Mixamo Xbot) driven by RMG motion, offline, so the README matches the
web demo. We retarget the generated SMPL joint directions onto the robot's bones (swing-only, validated to
dot=1 against the pose), skin the mesh with linear blend skinning, and reuse the ghost/caption/camera
machinery from render_mannequin. Runs in an env with pygltflib + pyrender + trimesh + PIL.

    python render_robot.py --joints report/fig1_joints.npz --glb assets_xbot.glb --mode montage --out report
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import imageio
import pygltflib

import render_mannequin
from render_mannequin import Renderer, composite, caption, fit_view, smooth_time

# mixamo bone (glTF uses the "mixamorig:" prefix) -> child bone -> [smpl parent joint, child joint]
ENTRIES = [
    ("Spine", "Spine1", 0, 3), ("Spine1", "Spine2", 3, 6), ("Spine2", "Neck", 6, 9), ("Neck", "Head", 12, 15),
    ("LeftShoulder", "LeftArm", 13, 16), ("LeftArm", "LeftForeArm", 16, 18), ("LeftForeArm", "LeftHand", 18, 20),
    ("RightShoulder", "RightArm", 14, 17), ("RightArm", "RightForeArm", 17, 19), ("RightForeArm", "RightHand", 19, 21),
    ("LeftUpLeg", "LeftLeg", 1, 4), ("LeftLeg", "LeftFoot", 4, 7), ("LeftFoot", "LeftToeBase", 7, 10),
    ("RightUpLeg", "RightLeg", 2, 5), ("RightLeg", "RightFoot", 5, 8), ("RightFoot", "RightToeBase", 8, 11),
]
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _acc(g, blob, idx):
    a = g.accessors[idx]; bv = g.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    arr = np.frombuffer(blob, dtype=CT[a.componentType], count=a.count * NT[a.type], offset=off)
    return arr.reshape(a.count, NT[a.type]).astype(np.float64 if a.componentType == 5126 else np.int64)


def _local(n):
    if n.matrix:
        return np.array(n.matrix, np.float64).reshape(4, 4).T            # glTF is column-major
    M = np.eye(4)
    if n.scale:
        M[:3, :3] = np.diag(n.scale)
    if n.rotation:
        x, y, z, w = n.rotation
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        M[:3, :3] = R @ M[:3, :3]
    if n.translation:
        M[:3, 3] = n.translation
    return M


def _rot_from_to(u, v):
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    ax = np.cross(u, v); s = np.linalg.norm(ax); c = float(np.dot(u, v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else _axis_angle(_perp(u), np.pi)
    return _axis_angle(ax / s, np.arctan2(s, c))


def _axis_angle(ax, ang):
    x, y, z = ax; c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _perp(u):
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    p = np.cross(u, a); return p / (np.linalg.norm(p) + 1e-12)


class Robot:
    def __init__(self, glb):
        g = pygltflib.GLTF2().load(glb); blob = g.binary_blob()
        self.g = g
        self.nodes = g.nodes
        self.name2i = {n.name.split(":")[-1]: i for i, n in enumerate(g.nodes)}
        self.parent = [-1] * len(g.nodes)
        for i, n in enumerate(g.nodes):
            for c in (n.children or []):
                self.parent[c] = i
        self.order = self._toposort()
        self.local0 = [_local(n) for n in g.nodes]                       # bind local matrices
        self.T0 = [m[:3, 3].copy() for m in self.local0]
        self.RS0 = [m[:3, :3].copy() for m in self.local0]
        self.world0 = self._fk(self.RS0_as_local())                     # bind world matrices
        sk = g.skins[0]
        self.joints = sk.joints                                         # node indices, len 67
        self.ibm = _acc(g, blob, sk.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
        # gather skinned meshes (Beta_Joints + Beta_Surface)
        self.parts = []
        for ni, n in enumerate(g.nodes):
            if n.skin is not None and n.mesh is not None:
                for prim in g.meshes[n.mesh].primitives:
                    V = _acc(g, blob, prim.attributes.POSITION)[:, :3]
                    J = _acc(g, blob, prim.attributes.JOINTS_0).astype(int)
                    W = _acc(g, blob, prim.attributes.WEIGHTS_0)
                    F = _acc(g, blob, prim.indices).reshape(-1, 3)
                    self.parts.append((np.hstack([V, np.ones((len(V), 1))]), J, W, F))
        # bind data for retarget
        self.bindR = {nm: self.world0[i][:3, :3] for nm, i in self.name2i.items() if i < len(self.world0)}
        self.bindPos = {nm: self.world0[i][:3, 3] for nm, i in self.name2i.items()}
        self.bindDir = {}
        for bn, cn, _, _ in ENTRIES:
            self.bindDir[bn] = _norm(self.bindPos[cn] - self.bindPos[bn])
        self.bindDir["__up"] = _norm(self.bindPos["Spine"] - self.bindPos["Hips"])
        self.bindDir["__right"] = _norm(self.bindPos["LeftUpLeg"] - self.bindPos["RightUpLeg"])
        self.hipsParentR = self.world0[self.parent[self.name2i["Hips"]]][:3, :3] if self.parent[self.name2i["Hips"]] >= 0 else np.eye(3)
        # robot height (bind) for scaling
        ys = np.array([self.world0[i][1, 3] for i in range(len(self.world0))])
        self.height = float(ys.max() - ys.min())

    def RS0_as_local(self):
        return self.local0

    def _toposort(self):
        order, seen = [], set()
        def visit(i):
            order.append(i); seen.add(i)
            for c in (self.nodes[i].children or []):
                if c not in seen:
                    visit(c)
        for i in range(len(self.nodes)):
            if self.parent[i] == -1:
                visit(i)
        return order

    def _fk(self, locals_):
        world = [None] * len(self.nodes)
        for i in self.order:
            p = self.parent[i]
            world[i] = (world[p] @ locals_[i]) if p >= 0 else locals_[i].copy()
        return world

    def pose(self, P):
        """Return skinned vertices (concatenated parts) for SMPL joints P (22,3)."""
        curR = {}                                                       # node idx -> world rotation (current)
        # hips: align bind up/right to target up/right
        tUp = _norm(P[3] - P[0]); tRight = _norm(P[1] - P[2])
        Q1 = _rot_from_to(self.bindDir["__up"], tUp)
        br = Q1 @ self.bindDir["__right"]; br = _norm(br - tUp * np.dot(br, tUp))
        tr = _norm(tRight - tUp * np.dot(tRight, tUp))
        ang = np.arccos(np.clip(np.dot(br, tr), -1, 1))
        if np.dot(np.cross(br, tr), tUp) < 0:
            ang = -ang
        RwHips = _axis_angle(tUp, ang) @ Q1 @ self.bindR["Hips"]
        hi = self.name2i["Hips"]; curR[hi] = RwHips
        newRS = [m.copy() for m in self.RS0]
        scl = [np.linalg.norm(self.RS0[i], axis=0) for i in range(len(self.nodes))]
        newRS[hi] = (np.linalg.inv(self.hipsParentR) @ RwHips) * scl[hi]
        for bn, _, a, b in ENTRIES:
            i = self.name2i[bn]
            Rwb = _rot_from_to(self.bindDir[bn], _norm(P[b] - P[a])) @ self.bindR[bn]
            curR[i] = Rwb
            pr = curR.get(self.parent[i], self.world0[self.parent[i]][:3, :3])
            newRS[i] = (np.linalg.inv(pr) @ Rwb) * scl[i]
        locals_ = []
        for i in range(len(self.nodes)):
            M = np.eye(4); M[:3, :3] = newRS[i]; M[:3, 3] = self.T0[i]
            locals_.append(M)
        world = self._fk(locals_)
        jm = np.stack([world[self.joints[j]] @ self.ibm[j] for j in range(len(self.joints))])  # (67,4,4)
        out = []
        for V, J, W, F in self.parts:
            sk = (jm[J] * W[:, :, None, None]).sum(1)                   # (nv,4,4)
            v = np.einsum("nij,nj->ni", sk, V)[:, :3]
            out.append(trimesh.Trimesh(v, F, process=False))
        m = trimesh.util.concatenate(out)
        return m


def _norm(v):
    return np.asarray(v, float) / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default="report/fig1_joints.npz")
    ap.add_argument("--glb", default="assets_xbot.glb")
    ap.add_argument("--mode", choices=["montage", "gif"], default="montage")
    ap.add_argument("--res", type=int, default=540)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--azim", type=float, default=-32.0)
    ap.add_argument("--elev", type=float, default=13.0)
    ap.add_argument("--ghosts", type=int, default=6)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    render_mannequin.CLAY[:] = [0.80, 0.82, 0.87]            # robot white-gray
    robot = Robot(a.glb)
    scale = 1.7 / robot.height
    hh = robot.bindPos["Hips"] * scale                       # hips world pos after scaling (to drive the trajectory)
    d = np.load(a.joints, allow_pickle=True)
    J, prompts = d["joints"], list(d["prompts"])
    J = np.stack([smooth_time(J[i], a.smooth) for i in range(J.shape[0])])
    flip = trimesh.transformations.rotation_matrix(np.radians(180.0), [0, 1, 0])[:3, :3]

    for i in range(J.shape[0]):
        if a.only >= 0 and i != a.only:
            continue
        Ji = J[i] @ flip.T
        L = Ji.shape[0]
        floor = Ji[..., 1].min(); cx, cz = Ji[..., 0].mean(), Ji[..., 2].mean()
        shift = np.array([-cx, -floor, -cz]); pts = Ji + shift
        dist, ty = fit_view(pts)
        R = Renderer(a.res, a.azim, a.elev, dist, target_y=ty)
        bg = R.ground()
        slug = "".join(c if c.isalnum() else "_" for c in prompts[i])[:28]

        def layer(f, alpha):
            m = robot.pose(Ji[f]); m.apply_scale(scale)
            m.apply_translation(Ji[f][0] - hh + shift)       # hips -> this frame's pelvis, then floor/center
            rgb, mask = R.body(m); return (rgb, mask, alpha)

        if a.mode == "montage":
            sel = np.linspace(0, L - 1, a.ghosts).astype(int)
            al = 0.14 + 0.86 * (np.linspace(0, 1, len(sel)) ** 1.3)
            img = caption(composite(bg, [layer(f, al[k]) for k, f in enumerate(sel)]), prompts[i])
            imageio.imwrite(os.path.join(a.out, f"robot_{i}_{slug}.png"), img)
            print(f"wrote robot_{i}_{slug}.png | {prompts[i]}")
        else:
            frames = []
            for t in range(L):
                ks = [k for k in range(a.every * a.ghosts, 0, -a.every) if t - k >= 0]
                ly = [layer(t - k, 0.10 + 0.30 * (1 - k / (a.every * a.ghosts))) for k in ks]
                ly.append(layer(t, 1.0))
                frames.append(caption(composite(bg, ly), prompts[i]))
            imageio.mimsave(os.path.join(a.out, f"robot_{i}_{slug}.gif"), frames, fps=a.fps)
            print(f"wrote robot_{i}_{slug}.gif | {prompts[i]}")
        R.r.delete()


if __name__ == "__main__":
    main()
