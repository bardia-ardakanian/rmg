# rmg reproduction (unofficial)

my own reimplementation of RMG ("Riemannian Motion Generation", arXiv:2603.15016), since the authors havent
released their code. text to motion on humanml3d. the method is theirs, i just rebuilt it from the paper, so
if a number is off thats on me, not them.

## what it is

motion lives on a product manifold: R^3 for the root translation and (S^3)^22 for the joint rotations as
unit quaternions (upper hemisphere). thats 91 numbers per frame. no dataset mean/std normalization, the
manifold takes care of scale.

the model is velocity riemannian flow matching. interpolate along geodesics, regress the tangent velocity
Log_{x_t}(x_1)/(1-t), project the network output onto the tangent space, integrate with a manifold euler
step. backbone is a frame-token diffusion transformer (384 dim, 6 layers, 8 heads, ffn x8, adaln). text is
qwen3-embedding-0.6b fused with the time embedding, trained with classifier-free guidance (10% dropout,
guidance around 6.5 at sampling).

the recipe is straight from their table 7: adamw, lr 1e-4, cosine with 0.08 warmup, effective batch 256,
150k steps, grad clip 0.5, ema.

## results

humanml3d test set, official guo evaluators, ema weights, guidance 6.5, 1024 samples:

| metric | mine | paper |
|---|---|---|
| r-precision top-3 | 0.793 | 0.805 |
| diversity | 9.517 | 9.555 |
| mm-dist | 3.102 | 2.930 |
| fid | 0.518 | 0.043 |

r-precision, diversity and mm-dist all land close to the paper. fid is the odd one out, about 0.5 vs their
0.043. short version: a few things the paper doesnt pin down (the translation "canonical length", the prior
covariance, the number of ode steps) plus fid being biased high at this sample count. the full breakdown,
their own table, and the baselines are in [docs/RESULTS.md](docs/RESULTS.md), raw numbers in
`results/metrics_humanml3d.json`.

## layout

```
src/        all the python (model, flow, data, training, eval, viz, shared utils)
scripts/    setup.sh (downloads), run_rmg.sh, tb.sh, watch.sh, fetch_qwen.py
data/       where humanml3d goes (or point HML_DIR at it), downloads land here too
docs/       results writeup
figures/    rendered gifs (ignored apart from the samples)
results/    metrics json
web/        the interactive viewer
```

## setup

```bash
pip install -r requirements.txt
bash scripts/setup.sh
```

setup.sh grabs the guo evaluators and glove, clones the text-to-motion and humanml3d repos, and caches the
qwen encoder. it prints the env vars to export (T2M_EVAL, T2M_REPO, HML3D_REPO). the humanml3d dataset
itself needs amass access and the smpl pipeline so the script cant just pull it, see `data/README.md`. point
`HML_DIR` at it once you have it.

## running it

```bash
bash scripts/run_rmg.sh                                          # train, writes to runs/rmg_base/
bash scripts/tb.sh                                              # tensorboard on 6006 (train + eval curves)
python src/rmg_eval_monitor.py --run runs/rmg_base             # log eval fid/r-precision to tb while training
python src/rmg_eval.py --ckpt runs/rmg_base/model.pth --n 1024 --guidance 6.5 --weights ema   # final eval
```

## visualization

everything renders from the generated joints, 3/4 camera with a ground shadow.

```bash
python src/rmg_gen_smpl.py    --ckpt runs/rmg_base/model.pth --out figures/joints.npz   # generate + export joints
python src/fit_render_mesh.py --joints figures/joints.npz --out figures                 # smpl-x body mesh (needs smplx)
python src/render_body.py     --joints figures/joints.npz --mode both --out figures      # capsule body or joints+skeleton
python src/render_rmg.py      --ckpt runs/rmg_base/model.pth --weights raw               # quick matplotlib stick figure
```

fit_render_mesh fits smpl-x to the joints (the right way to do it, since humanml3d rotations arent actual
smpl pose params) and gives a real body. render_body needs no body model, just a capsule figure or a
joints+skeleton one. all of them smooth the joints over time (`--smooth`, default 9) so it doesnt jitter,
pass `--smooth 0` to turn it off.

### samples

smpl-x body mesh fit to the joints:

| "a person walks forward" | "a person sits down" |
|---|---|
| ![walk mesh](figures/walk_mesh.gif) | ![sit mesh](figures/sit_mesh.gif) |

joints + skeleton, no body model:

| "a person walks forward" | "a person sits down" |
|---|---|
| ![walk skeleton](figures/walk_skeleton.gif) | ![sit skeleton](figures/sit_skeleton.gif) |

## web demo

theres an interactive viewer too. type a prompt, it generates on whatever machine runs the server, and the
browser shows the motion in 3d. orbit with the mouse, play/scrub, switch between body and skeleton, export a
png or a gif.

```bash
python web/app.py        # port 8000
```

if its running on a remote box, tunnel the port and open it locally:

```bash
ssh -L 8000:localhost:8000 <host>
# then open http://localhost:8000
```

set `RMG_CKPT` if your checkpoint isnt at `runs/rmg_base/model.pth`.

## stuff i ran into

- qwen pooling. qwen3-embedding wants last-token pooling with left padding, then an l2 normalize (see
  `src/qwen_text.py`). if you mean-pool, or right-pad and grab the last position (which is then a pad token),
  the embeddings basically collapse to the same vector for every caption and the text conditioning quietly
  dies, r-precision sits at chance. took me a while to spot.
- ema looks broken early. with decay 0.9999 the early ema weights are still dominated by the first few
  thousand bad steps, so evaluating ema partway through reads like garbage (fid near the prior) even though
  the raw weights are fine. monitor with `--weights raw` and switch to ema near the end.
- humanml3d rotations arent smpl pose. the 263-d rotations are recomputed per bone, not the original smpl
  params, so handing them to a body model gives a near t-pose. the joints are fine though, which is why the
  mesh viz fits smpl-x to the joints instead of using the rotations directly.
- things the paper leaves open that move fid: the translation scaling, the prior covariance (i used 1.0),
  and the ode step count (i used 100).

## citation

if you use the method, cite the original paper:

```bibtex
@article{miao2026rmg,
  title  = {Riemannian Motion Generation: A Unified Framework for Human Motion
            Representation and Generation via Riemannian Flow Matching},
  author = {Miao, Fangran and Huang, Jian and Li, Ting},
  journal= {arXiv preprint arXiv:2603.15016},
  year   = {2026}
}
```

if this repo helped, a link back is appreciated:

```bibtex
@misc{ardakanian2026rmgrepro,
  title  = {An Unofficial Reproduction of Riemannian Motion Generation (RMG)},
  author = {Ardakanian, Bardia},
  year   = {2026},
  howpublished = {\url{https://github.com/bardia-ardakanian/rmg-reproduction}}
}
```

## license

MIT, see `LICENSE`.
