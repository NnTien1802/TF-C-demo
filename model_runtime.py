"""Inference for the locally trained original TF-C HAR -> SGR checkpoint.

Training used the source `model.py` with input [B, 1, 206].  The web interface
still accepts three-axis SGR windows; only axis 0 is passed to the model,
matching the user's training/evaluation commands.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

LABELS = ["typing", "teeth", "drinking", "writing", "clapping", "folding"]
TRANSLATIONS = {
    "typing": "Gõ phím", "teeth": "Đánh răng", "drinking": "Uống nước",
    "writing": "Viết chữ", "clapping": "Vỗ tay", "folding": "Gấp đồ",
}


class TFC(nn.Module):
    """Exact tensor layout and layer names of the trained code/TFC/model.py."""
    def __init__(self, length=206):
        super().__init__()
        enc_t = nn.TransformerEncoderLayer(length, dim_feedforward=2 * length, nhead=2)
        self.transformer_encoder_t = nn.TransformerEncoder(enc_t, 2)
        self.projector_t = nn.Sequential(nn.Linear(length, 256), nn.BatchNorm1d(256),
                                         nn.ReLU(), nn.Linear(256, 128))
        enc_f = nn.TransformerEncoderLayer(length, dim_feedforward=2 * length, nhead=2)
        self.transformer_encoder_f = nn.TransformerEncoder(enc_f, 2)
        self.projector_f = nn.Sequential(nn.Linear(length, 256), nn.BatchNorm1d(256),
                                         nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x_t, x_f):
        h_t = self.transformer_encoder_t(x_t)
        h_t = h_t.reshape(h_t.shape[0], -1)
        z_t = self.projector_t(h_t)
        h_f = self.transformer_encoder_f(x_f)
        h_f = h_f.reshape(h_f.shape[0], -1)
        z_f = self.projector_f(h_f)
        return h_t, z_t, h_f, z_f


class Classifier(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.logits = nn.Linear(256, 64)
        self.logits_simple = nn.Linear(64, classes)

    def forward(self, embedding):
        return self.logits_simple(torch.sigmoid(self.logits(embedding)))


def load_model(checkpoint_path):
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint: {p}")
    ck = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or not {"model_state_dict", "classifier_state_dict"} <= ck.keys():
        raise ValueError("Checkpoint cần model_state_dict và classifier_state_dict từ tfc_sgr_demo.pt")
    if int(ck.get("num_classes", 6)) != 6 or int(ck.get("input_length", 206)) != 206 or int(ck.get("input_channels", 1)) != 1:
        raise ValueError("Checkpoint không khớp cấu hình đã train: 6 lớp, 1 kênh, 206 điểm")
    model, classifier = TFC(), Classifier()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    classifier.load_state_dict(ck["classifier_state_dict"], strict=True)
    model.eval()
    classifier.eval()
    return model, classifier, {"channels": 1, "length": 206, "labels": LABELS}


def predict(model, classifier, samples, batch_size=6):
    """Softmax [N,6] from normalized SGR [N,3,206]. Select first axis as trained."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (3, 206):
        raise ValueError(f"SGR samples phải có shape [N,3,206], thực tế {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Dữ liệu chứa NaN hoặc Inf")
    results = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xt = torch.from_numpy(x[start:start + batch_size, :1, :].copy())
            xf = torch.fft.fft(xt, dim=-1).abs()
            _, zt, _, zf = model(xt, xf)
            logits = classifier(torch.cat([zt, zf], dim=1))
            results.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(results, axis=0) if results else np.empty((0, 6), dtype=np.float32)
