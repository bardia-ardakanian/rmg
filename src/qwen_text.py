"""
Text encoder for RMG-base (HumanML3D): Qwen3-Embedding-0.6B -> 1024-d sentence vector.
Last-token pooling (the Qwen3-Embedding recipe) with left padding, L2-normalized. Frozen.
"""
import torch

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIM = 1024

_tok = None
_model = None


def _load(device):
    global _tok, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        _tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        _model = AutoModel.from_pretrained(MODEL).to(device).eval()
    return _tok, _model


@torch.no_grad()
def encode(texts, device="cuda", batch=128, max_len=64):
    """List[str] -> (len(texts), 1024) float32, L2-normalized. Last-token pooling, left padding."""
    tok, model = _load(device)
    out = []
    for s in range(0, len(texts), batch):
        b = tok(texts[s:s + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = model(**b).last_hidden_state               # (B, L, H)
        emb = h[:, -1]                                  # left padding -> last position is the real last token
        emb = torch.nn.functional.normalize(emb, dim=-1)
        out.append(emb.float().cpu())
    return torch.cat(out, 0)


if __name__ == "__main__":
    e = encode(["a person walks forward", "a man performs a taichi movement", "someone sits down slowly"],
               device="cuda" if torch.cuda.is_available() else "cpu")
    print("text embeds:", tuple(e.shape), "dim ok:", e.shape[1] == DIM, "unit-norm:", float(e.norm(dim=-1).mean()))
