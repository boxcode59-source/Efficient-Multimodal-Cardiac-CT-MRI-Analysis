"""
================================================================================
 MULTIMODAL CT-MRI CARDIAC DIAGNOSIS NETWORK
 Full working implementation of the proposed architecture:

   Step 1: Patient-Level CT-MRI Preparation & Cardiac Region Extraction
   Step 2: Lightweight Patch Embedding & Token Generation
   Step 3: [Novelty 1] Dynamic Token Importance Estimation & Routing (DTIE/DTR)
   Step 4: [Novelty 2] Heterogeneous Mixture-of-Experts (Local/Global/Efficient)
   Step 5: Modality-Specific Feature Refinement (Channel Attn / Spatial Attn)
   Step 6: [Novelty 3] Cross-Modality Alignment & Adaptive Fusion (AMEG)
   Step 7: Progressive Token Compression (Adaptive Token Merging + CLS token)
   Step 8: Cardiac Classification & Training

 Also included: full metric suite (Accuracy, Precision, Recall, Specificity,
 F1, MCC, Cohen's Kappa, ROC-AUC, FPR/FNR, Confusion Matrix), K-Fold CV,
 ANOVA significance testing, ablation-study runner (reproduces Tables 1-3),
 and diagnostic plots (training curves, ROC, confusion matrix, token-routing
 analysis, expert-utilization analysis, CT/MRI contribution analysis).

 Dependencies:
   pip install torch numpy scipy scikit-learn matplotlib --break-system-packages

 Usage:
   python cardiac_multimodal_net.py                 # runs on synthetic data
   python cardiac_multimodal_net.py --data_root DIR  # runs on real dataset
   python cardiac_multimodal_net.py --ablation       # reproduces ablation tables
   python cardiac_multimodal_net.py --kfold 5        # k-fold cross validation

 Wire in the real dataset (e.g. the Kaggle "Multimodal Cardiac CT-MRI
 Diagnosis Dataset") by pointing --data_root at a directory laid out as:
   data_root/
     patient_0001/ct.npy   patient_0001/mri.npy   patient_0001/label.txt
     patient_0002/ct.npy   patient_0002/mri.npy   patient_0002/label.txt
     ...
 ct.npy / mri.npy: single-channel 2D arrays (H, W), any size (auto-resized).
 label.txt: single integer class id.
 If --data_root is not given, a synthetic dataset is generated automatically
 so every part of the pipeline is exercised end-to-end.
================================================================================
"""

import os
import math
import json
import random
import argparse
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from scipy.ndimage import gaussian_filter
from scipy.stats import f_oneway
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. CONFIG
# ==============================================================================

@dataclass
class Config:
    # data
    data_root: Optional[str] = None
    image_size: int = 128
    patch_size: int = 8
    num_classes: int = 4                      # e.g. ACDC-style diagnostic classes
    num_synthetic_patients: int = 240

    # model dims
    embed_dim: int = 96
    num_heads: int = 4
    window_size: int = 4                      # in tokens, for local (window) expert
    mlp_ratio: float = 4.0
    depth_refine: int = 2                      # residual transformer blocks per modality

    # DTIE / DTR (Novelty 1)
    use_dtie: bool = True
    use_gated_mlp: bool = True
    use_dtr: bool = True
    use_topk_routing: bool = True
    use_adaptive_pruning: bool = True
    keep_ratio_high: float = 0.35              # fraction of tokens routed "high"
    keep_ratio_medium: float = 0.35            # fraction routed "medium"
    # remainder -> "low" (redundant) group

    # Heterogeneous MoE (Novelty 2)
    use_window_mhsa: bool = True               # Local Expert
    use_global_mhsa: bool = True                # Global Expert
    use_linear_attn: bool = True                # Efficient Expert
    use_expert_gating: bool = True
    use_adaptive_fusion: bool = True

    # Cross-modality fusion (Novelty 3)
    use_cross_proj: bool = True
    use_cross_attn: bool = True
    use_ameg: bool = True

    # token compression
    merge_ratio: float = 0.5                   # fraction of tokens kept after merge

    # training
    batch_size: int = 8
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_consistency: float = 0.1
    lambda_routing: float = 0.05
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir: str = "./outputs"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# STEP 1: Patient-level CT/MRI preparation & cardiac-region extraction
# ==============================================================================

def ct_window_normalize(img: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """CT intensity-window normalization (typical cardiac soft-tissue window)."""
    lo, hi = wl - ww / 2, wl + ww / 2
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-8)


def mri_percentile_normalize(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """MRI percentile-based intensity normalization (robust to scanner scaling)."""
    p_lo, p_hi = np.percentile(img, [low, high])
    img = np.clip(img, p_lo, p_hi)
    return (img - p_lo) / (p_hi - p_lo + 1e-8)


def denoise(img: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Gaussian filtering for noise reduction."""
    return gaussian_filter(img, sigma=sigma)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest/bilinear resize via torch (keeps this file dependency-light)."""
    t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def random_augment(img: np.ndarray, training: bool) -> np.ndarray:
    """Controlled training augmentation: flips + small rotations."""
    if not training:
        return img
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
    if random.random() < 0.5:
        img = np.flipud(img).copy()
    k = random.choice([0, 1, 2, 3])
    if k:
        img = np.rot90(img, k).copy()
    return img


class SpatialAttentionModule(nn.Module):
    """SAM: suppresses irrelevant background, emphasizes the cardiac region."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):  # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn, attn


# ==============================================================================
# DATASET  (real dataset loader + synthetic fallback for a runnable demo)
# ==============================================================================

class CardiacCTMRIDataset(Dataset):
    """
    Patient-level paired CT/MRI dataset.
    Expects <root>/<patient_id>/{ct.npy, mri.npy, label.txt}. If data_root is
    None, generates class-separable synthetic CT/MRI pairs so the full
    pipeline (including Novelty 1-3 modules) can be exercised without the
    real dataset present.
    """

    def __init__(self, cfg: Config, patient_ids: List[str], training: bool,
                 synthetic: bool = False, synth_seed: int = 0):
        self.cfg = cfg
        self.patient_ids = patient_ids
        self.training = training
        self.synthetic = synthetic
        self.synth_seed = synth_seed

    def __len__(self):
        return len(self.patient_ids)

    def _load_real(self, pid: str):
        pdir = os.path.join(self.cfg.data_root, pid)
        ct = np.load(os.path.join(pdir, "ct.npy")).astype(np.float32)
        mri = np.load(os.path.join(pdir, "mri.npy")).astype(np.float32)
        with open(os.path.join(pdir, "label.txt")) as f:
            label = int(f.read().strip())
        return ct, mri, label

    def _load_synthetic(self, pid: str):
        rng = np.random.RandomState(abs(hash((pid, self.synth_seed))) % (2 ** 31))
        label = rng.randint(0, self.cfg.num_classes)
        size = self.cfg.image_size
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = size / 2 + rng.uniform(-5, 5), size / 2 + rng.uniform(-5, 5)
        radius = size * (0.18 + 0.03 * label) + rng.uniform(-3, 3)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2)))
        ct = blob * (0.6 + 0.1 * label) + rng.normal(0, 0.05, (size, size))
        mri = blob * (0.5 + 0.08 * ((label + 1) % self.cfg.num_classes)) + rng.normal(0, 0.06, (size, size))
        ct = ct * 400 - 40      # fake HU-like scale for the CT windowing step
        mri = mri * 1000        # fake MRI intensity scale for percentile norm
        return ct.astype(np.float32), mri.astype(np.float32), label

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        if self.synthetic:
            ct, mri, label = self._load_synthetic(pid)
        else:
            ct, mri, label = self._load_real(pid)

        ct = ct_window_normalize(ct)
        mri = mri_percentile_normalize(mri)
        ct = denoise(ct)
        mri = denoise(mri)
        ct = resize_image(ct, self.cfg.image_size)
        mri = resize_image(mri, self.cfg.image_size)
        ct = random_augment(ct, self.training)
        mri = random_augment(mri, self.training)

        ct_t = torch.from_numpy(ct).float().unsqueeze(0)     # (1,H,W)
        mri_t = torch.from_numpy(mri).float().unsqueeze(0)   # (1,H,W)
        return ct_t, mri_t, label


def get_patient_ids(cfg: Config) -> List[str]:
    if cfg.data_root is not None:
        return sorted([d for d in os.listdir(cfg.data_root)
                       if os.path.isdir(os.path.join(cfg.data_root, d))])
    return [f"synth_{i:05d}" for i in range(cfg.num_synthetic_patients)]


def _safe_stratify(ids: List[str], labels: List[int]):
    """Returns `labels` for stratification, or None if any class has < 2 members
    (train_test_split requires >= 2 members per class to stratify)."""
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    if len(ids) < 2 or min(counts.values()) < 2:
        return None
    return labels


def patient_level_split(patient_ids: List[str], cfg: Config, val_frac=0.15, test_frac=0.15):
    """Patient-level train/val/test split (no patient leakage across splits)."""
    labels_proxy = [abs(hash((pid, 0))) % cfg.num_classes for pid in patient_ids]
    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=val_frac + test_frac, random_state=cfg.seed,
        stratify=_safe_stratify(patient_ids, labels_proxy))
    temp_labels = [abs(hash((pid, 0))) % cfg.num_classes for pid in temp_ids]
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=test_frac / (val_frac + test_frac), random_state=cfg.seed,
        stratify=_safe_stratify(temp_ids, temp_labels))
    return train_ids, val_ids, test_ids


# ==============================================================================
# STEP 2: Lightweight patch embedding & token generation
# ==============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class PatchEmbedding(nn.Module):
    """DSC-based patch embedding + learnable positional embeddings."""

    def __init__(self, cfg: Config, in_ch: int = 1):
        super().__init__()
        self.cfg = cfg
        self.proj = DepthwiseSeparableConv(
            in_ch, cfg.embed_dim, kernel_size=cfg.patch_size,
            stride=cfg.patch_size, padding=0)
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, cfg.embed_dim) * 0.02)
        self.n_patches_side = cfg.image_size // cfg.patch_size

    def forward(self, x):  # x: (B,1,H,W)
        x = self.proj(x)                              # (B, D, h, w)
        B, D, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, N, D)
        tokens = tokens + self.pos_embed
        return tokens, (h, w)


# ==============================================================================
# STEP 3 [Novelty 1]: Dynamic Token Importance Estimation & Routing
# ==============================================================================

class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_mult=2.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        h = F.gelu(self.fc1(x)) * torch.sigmoid(self.gate(x))
        return self.fc2(h)


class DTIE(nn.Module):
    """Dynamic Token Importance Estimator: scores each token's informativeness."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.use_gated_mlp:
            self.scorer = nn.Sequential(GatedMLP(cfg.embed_dim), nn.Linear(cfg.embed_dim, 1))
        else:
            self.scorer = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim // 2), nn.ReLU(),
                nn.Linear(cfg.embed_dim // 2, 1))

    def forward(self, tokens):  # (B,N,D) -> (B,N) importance in [0,1]
        if not self.cfg.use_dtie:
            return torch.ones(tokens.shape[:2], device=tokens.device)
        score = self.scorer(tokens).squeeze(-1)
        return torch.sigmoid(score)


class DTR(nn.Module):
    """
    Dynamic Token Router: top-k routing that splits tokens into
    high / medium / low information groups and applies adaptive pruning
    (soft, differentiable gating rather than hard index removal, so batches
    keep a uniform tensor shape while still behaving like true routing).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, tokens, importance):
        # tokens: (B,N,D), importance: (B,N)
        B, N, D = tokens.shape
        if not self.cfg.use_dtr:
            group = torch.zeros(B, N, dtype=torch.long, device=tokens.device)  # all "high"
            route_weight = torch.ones(B, N, device=tokens.device)
            return tokens, group, route_weight

        n_high = max(1, int(N * self.cfg.keep_ratio_high))
        n_med = max(1, int(N * self.cfg.keep_ratio_medium))

        if self.cfg.use_topk_routing:
            order = torch.argsort(importance, dim=1, descending=True)  # (B,N)
        else:
            order = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)

        group = torch.full((B, N), 2, dtype=torch.long, device=tokens.device)  # 2 = low
        rank = torch.argsort(order, dim=1)  # rank[b, n] = position of token n in sorted order
        group = torch.where(rank < n_high, torch.zeros_like(group), group)
        group = torch.where((rank >= n_high) & (rank < n_high + n_med), torch.ones_like(group), group)

        # adaptive pruning: low-information tokens get down-weighted (cheap
        # path) rather than physically dropped, preserving batchability while
        # still reducing their effective contribution / compute weight.
        if self.cfg.use_adaptive_pruning:
            route_weight = torch.ones(B, N, device=tokens.device)
            route_weight = torch.where(group == 2, torch.full_like(route_weight, 0.25), route_weight)
            route_weight = torch.where(group == 1, torch.full_like(route_weight, 0.65), route_weight)
        else:
            route_weight = torch.ones(B, N, device=tokens.device)

        return tokens, group, route_weight

    @staticmethod
    def stats(group: torch.Tensor) -> Dict[str, float]:
        total = group.numel()
        high = (group == 0).float().mean().item()
        med = (group == 1).float().mean().item()
        low = (group == 2).float().mean().item()
        return {"high_frac": high, "medium_frac": med, "low_frac": low}


# ==============================================================================
# STEP 4 [Novelty 2]: Heterogeneous Mixture-of-Experts
# ==============================================================================

class WindowMHSA(nn.Module):
    """Local Expert: window-based multi-head self-attention over token grid."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.ws = cfg.window_size

    def forward(self, tokens, grid_hw):
        h, w = grid_hw
        B, N, D = tokens.shape
        ws = min(self.ws, h, w)
        x = tokens.view(B, h, w, D)
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        H2, W2 = x.shape[1], x.shape[2]
        x = x.view(B, H2 // ws, ws, W2 // ws, ws, D).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, ws * ws, D)                      # (B*num_windows, ws*ws, D)
        out, _ = self.attn(x, x, x)
        out = out.view(B, H2 // ws, W2 // ws, ws, ws, D).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H2, W2, D)
        out = out[:, :h, :w, :].reshape(B, N, D)
        return out


class GlobalMHSA(nn.Module):
    """Global Expert: full self-attention capturing long-range dependencies."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)

    def forward(self, tokens, grid_hw=None):
        out, _ = self.attn(tokens, tokens, tokens)
        return out


class LinearAttention(nn.Module):
    """Efficient Expert: linear-complexity attention via kernel feature maps."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, tokens, grid_hw=None):
        B, N, D = tokens.shape
        qkv = self.qkv(tokens).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B,H,N,d)
        q, k = self._phi(q), self._phi(k)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)           # (B,H,d,d)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class ExpertGatingNetwork(nn.Module):
    """Produces per-token softmax weights over {local, global, efficient} experts."""

    def __init__(self, cfg: Config, n_experts: int = 3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.embed_dim // 2), nn.GELU(),
                                 nn.Linear(cfg.embed_dim // 2, n_experts))

    def forward(self, tokens):
        return F.softmax(self.fc(tokens), dim=-1)  # (B,N,n_experts)


class HeterogeneousMoEBlock(nn.Module):
    """
    Combines Local (window MHSA), Global (MHSA) and Efficient (linear attn)
    experts. DTR's route_weight modulates how much each token contributes to
    the expensive experts (redundant/low tokens lean on the cheap expert).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.local_expert = WindowMHSA(cfg) if cfg.use_window_mhsa else None
        self.global_expert = GlobalMHSA(cfg) if cfg.use_global_mhsa else None
        self.efficient_expert = LinearAttention(cfg) if cfg.use_linear_attn else None
        n_active = sum([cfg.use_window_mhsa, cfg.use_global_mhsa, cfg.use_linear_attn]) or 1
        self.gate = ExpertGatingNetwork(cfg, n_experts=max(n_active, 1)) if cfg.use_expert_gating else None
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio)), nn.GELU(),
            nn.Linear(int(cfg.embed_dim * cfg.mlp_ratio), cfg.embed_dim))
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, tokens, grid_hw, route_weight):
        x = self.norm(tokens)
        outs, names = [], []
        if self.local_expert is not None:
            outs.append(self.local_expert(x, grid_hw)); names.append("local")
        if self.global_expert is not None:
            # global expert gets extra weight on high-importance tokens via route_weight
            gx = x * route_weight.unsqueeze(-1) + x * (1 - route_weight.unsqueeze(-1)) * 0.0
            outs.append(self.global_expert(gx, grid_hw)); names.append("global")
        if self.efficient_expert is not None:
            outs.append(self.efficient_expert(x, grid_hw)); names.append("efficient")
        if len(outs) == 0:
            fused, weights = x, None
        elif len(outs) == 1 or not self.cfg.use_adaptive_fusion:
            fused = sum(outs) / len(outs)
            weights = None
        else:
            stacked = torch.stack(outs, dim=2)                     # (B,N,E,D)
            if self.gate is not None:
                gate_w = self.gate(x)                               # (B,N,E)
                # redundant tokens (low route_weight) are biased toward the
                # cheap "efficient" expert (assumed last in the stack)
                bias = (1 - route_weight).unsqueeze(-1)
                bias_vec = torch.zeros_like(gate_w)
                bias_vec[..., -1] = 1.0
                gate_w = F.softmax(gate_w + bias * bias_vec * 2.0, dim=-1)
            else:
                gate_w = torch.full((x.shape[0], x.shape[1], len(outs)), 1.0 / len(outs), device=x.device)
            fused = (stacked * gate_w.unsqueeze(-1)).sum(dim=2)
            weights = gate_w
        tokens = tokens + fused
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens, weights, names


# ==============================================================================
# STEP 5: Modality-specific feature refinement
# ==============================================================================

class ResidualTransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio)), nn.GELU(),
            nn.Linear(int(cfg.embed_dim * cfg.mlp_ratio), cfg.embed_dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ChannelAttention(nn.Module):
    """SE-style channel attention, used to strengthen informative CT features."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction), nn.ReLU(),
            nn.Linear(dim // reduction, dim), nn.Sigmoid())

    def forward(self, tokens):  # (B,N,D)
        pooled = tokens.mean(dim=1)             # (B,D)
        gate = self.fc(pooled).unsqueeze(1)     # (B,1,D)
        return tokens * gate


class LocalSpatialAttention(nn.Module):
    """Preserves important local MRI structures via a lightweight token-grid conv."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

    def forward(self, tokens, grid_hw):
        B, N, D = tokens.shape
        h, w = grid_hw
        grid = tokens.transpose(1, 2).reshape(B, D, h, w)
        attn = torch.sigmoid(self.conv(grid))               # (B,1,h,w)
        grid = grid * attn
        return grid.flatten(2).transpose(1, 2)


class ModalityRefinement(nn.Module):
    def __init__(self, cfg: Config, modality: str):
        super().__init__()
        self.modality = modality
        self.blocks = nn.ModuleList([ResidualTransformerBlock(cfg) for _ in range(cfg.depth_refine)])
        if modality == "ct":
            self.attn = ChannelAttention(cfg.embed_dim)
        else:
            self.attn = LocalSpatialAttention(cfg.embed_dim)

    def forward(self, tokens, grid_hw):
        for blk in self.blocks:
            tokens = blk(tokens)
        if self.modality == "ct":
            tokens = self.attn(tokens)
        else:
            tokens = self.attn(tokens, grid_hw)
        return tokens


# ==============================================================================
# STEP 6 [Novelty 3]: Cross-modality alignment & adaptive fusion (AMEG)
# ==============================================================================

class CrossAttentionBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(cfg.embed_dim)
        self.norm_kv = nn.LayerNorm(cfg.embed_dim)

    def forward(self, query_tokens, kv_tokens):
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(kv_tokens)
        out, attn_w = self.attn(q, kv, kv)
        return query_tokens + out, attn_w


class AMEG(nn.Module):
    """Adaptive Modality Evidence Gate: patient-specific CT vs MRI weighting."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim), nn.GELU(),
            nn.Linear(cfg.embed_dim, 1))

    def forward(self, ct_pooled, mri_pooled):
        alpha = torch.sigmoid(self.fc(torch.cat([ct_pooled, mri_pooled], dim=-1)))  # (B,1) CT weight
        return alpha


class CrossModalityFusion(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.use_cross_proj:
            self.proj_ct = nn.Linear(cfg.embed_dim, cfg.embed_dim)
            self.proj_mri = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        else:
            self.proj_ct = self.proj_mri = nn.Identity()
        if cfg.use_cross_attn:
            self.ct_to_mri = CrossAttentionBlock(cfg)
            self.mri_to_ct = CrossAttentionBlock(cfg)
        self.ameg = AMEG(cfg) if cfg.use_ameg else None

    def forward(self, ct_tokens, mri_tokens):
        ct_p = self.proj_ct(ct_tokens)
        mri_p = self.proj_mri(mri_tokens)

        if self.cfg.use_cross_attn:
            ct_fused, _ = self.ct_to_mri(ct_p, mri_p)
            mri_fused, _ = self.mri_to_ct(mri_p, ct_p)
        else:
            ct_fused, mri_fused = ct_p, mri_p

        ct_pooled = ct_fused.mean(dim=1)
        mri_pooled = mri_fused.mean(dim=1)

        if self.ameg is not None:
            alpha = self.ameg(ct_pooled, mri_pooled)          # (B,1) CT contribution
        else:
            alpha = torch.full((ct_pooled.shape[0], 1), 0.5, device=ct_pooled.device)

        fused_tokens = torch.cat(
            [ct_fused * alpha.unsqueeze(1), mri_fused * (1 - alpha.unsqueeze(1))], dim=1)
        return fused_tokens, alpha.squeeze(-1)


# ==============================================================================
# STEP 7: Progressive token compression (adaptive token merging + class token)
# ==============================================================================

class AdaptiveTokenMerging(nn.Module):
    """
    Bipartite similarity-based token merging (ToMe-style): splits tokens into
    two sets, matches each token in set A to its most similar token in set B,
    and merges the top-similarity pairs by averaging - reducing token count
    while retaining the most important multimodal information.
    """

    def __init__(self, merge_ratio: float):
        super().__init__()
        self.merge_ratio = merge_ratio

    def forward(self, tokens):
        B, N, D = tokens.shape
        n_keep = max(1, int(N * self.merge_ratio))
        n_merge = N - n_keep
        if n_merge <= 0:
            return tokens

        a, b = tokens[:, 0::2, :], tokens[:, 1::2, :]
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        sim = torch.bmm(a_n, b_n.transpose(1, 2))              # (B, |A|, |B|)
        best_sim, best_idx = sim.max(dim=-1)                    # match each A-token to best B-token

        n_a = a.shape[1]
        n_merge = min(n_merge, n_a)
        merge_order = torch.argsort(best_sim, dim=1, descending=True)
        merge_mask = torch.zeros(B, n_a, dtype=torch.bool, device=tokens.device)
        for bi in range(B):
            merge_mask[bi, merge_order[bi, :n_merge]] = True

        merged_b = b.clone()
        out_a_list = []
        for bi in range(B):
            idx_to_merge = merge_mask[bi].nonzero(as_tuple=True)[0]
            keep_idx = (~merge_mask[bi]).nonzero(as_tuple=True)[0]
            if len(idx_to_merge) > 0:
                targets = best_idx[bi, idx_to_merge]
                merged_b[bi, targets] = (merged_b[bi, targets] + a[bi, idx_to_merge]) / 2.0
            out_a_list.append(a[bi, keep_idx])

        max_keep = max(x.shape[0] for x in out_a_list) if out_a_list else 0
        if max_keep > 0:
            padded_a = torch.zeros(B, max_keep, D, device=tokens.device)
            for bi, t in enumerate(out_a_list):
                padded_a[bi, :t.shape[0]] = t
            out = torch.cat([padded_a, merged_b], dim=1)
        else:
            out = merged_b
        return out


# ==============================================================================
# STEP 8: Full model + classification head
# ==============================================================================

class BottleneckMLPHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(cfg.embed_dim // 2, cfg.num_classes))

    def forward(self, x):
        return self.net(x)


class CardiacMultimodalNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_embed_ct = PatchEmbedding(cfg)
        self.patch_embed_mri = PatchEmbedding(cfg)
        self.sam_ct = SpatialAttentionModule()
        self.sam_mri = SpatialAttentionModule()

        self.dtie_ct = DTIE(cfg)
        self.dtie_mri = DTIE(cfg)
        self.dtr_ct = DTR(cfg)
        self.dtr_mri = DTR(cfg)

        self.moe_ct = HeterogeneousMoEBlock(cfg)
        self.moe_mri = HeterogeneousMoEBlock(cfg)

        self.refine_ct = ModalityRefinement(cfg, "ct")
        self.refine_mri = ModalityRefinement(cfg, "mri")

        self.cross_fusion = CrossModalityFusion(cfg)
        self.token_merge = AdaptiveTokenMerging(cfg.merge_ratio)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim) * 0.02)
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.head = BottleneckMLPHead(cfg)

    def forward(self, ct_img, mri_img):
        # Step 1 (SAM applied on raw single-channel image before patching)
        ct_img, _ = self.sam_ct(ct_img)
        mri_img, _ = self.sam_mri(mri_img)

        # Step 2
        ct_tok, ct_grid = self.patch_embed_ct(ct_img)
        mri_tok, mri_grid = self.patch_embed_mri(mri_img)

        # Step 3 (Novelty 1)
        ct_imp = self.dtie_ct(ct_tok)
        mri_imp = self.dtie_mri(mri_tok)
        ct_tok, ct_group, ct_route_w = self.dtr_ct(ct_tok, ct_imp)
        mri_tok, mri_group, mri_route_w = self.dtr_mri(mri_tok, mri_imp)

        # Step 4 (Novelty 2)
        ct_tok, ct_expert_w, expert_names = self.moe_ct(ct_tok, ct_grid, ct_route_w)
        mri_tok, mri_expert_w, _ = self.moe_mri(mri_tok, mri_grid, mri_route_w)

        # Step 5
        ct_tok = self.refine_ct(ct_tok, ct_grid)
        mri_tok = self.refine_mri(mri_tok, mri_grid)

        # Step 6 (Novelty 3)
        fused_tokens, alpha_ct = self.cross_fusion(ct_tok, mri_tok)

        # Step 7
        fused_tokens = self.token_merge(fused_tokens)
        cls = self.cls_token.expand(fused_tokens.shape[0], -1, -1)
        fused_tokens = torch.cat([cls, fused_tokens], dim=1)
        fused_tokens = self.final_norm(fused_tokens)
        pooled = fused_tokens[:, 0]                      # class-token representation

        # Step 8
        logits = self.head(pooled)

        aux = {
            "ct_importance": ct_imp, "mri_importance": mri_imp,
            "ct_group": ct_group, "mri_group": mri_group,
            "ct_route_weight": ct_route_w, "mri_route_weight": mri_route_w,
            "ct_expert_weights": ct_expert_w, "mri_expert_weights": mri_expert_w,
            "expert_names": expert_names,
            "alpha_ct": alpha_ct,                          # per-patient CT contribution
            "ct_pooled_for_consistency": ct_tok.mean(dim=1),
            "mri_pooled_for_consistency": mri_tok.mean(dim=1),
            "n_tokens_after_merge": fused_tokens.shape[1] - 1,
            "n_tokens_before_merge": ct_tok.shape[1] + mri_tok.shape[1],
        }
        return logits, aux


# ==============================================================================
# LOSSES
# ==============================================================================

class CombinedLoss(nn.Module):
    """Classification + modality-consistency + routing load-balance losses."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, aux):
        loss_cls = self.ce(logits, labels)

        # modality-consistency: encourage CT/MRI pooled reps to agree on
        # patient identity information (simple cosine-alignment proxy)
        ct_p = F.normalize(aux["ct_pooled_for_consistency"], dim=-1)
        mri_p = F.normalize(aux["mri_pooled_for_consistency"], dim=-1)
        loss_consistency = (1 - (ct_p * mri_p).sum(dim=-1)).mean()

        # routing load-balance: encourage roughly the target high/med/low split
        cfg = self.cfg
        target = torch.tensor(
            [cfg.keep_ratio_high, cfg.keep_ratio_medium,
             max(1e-6, 1 - cfg.keep_ratio_high - cfg.keep_ratio_medium)],
            device=logits.device)
        actual_fracs = []
        for g in [aux["ct_group"], aux["mri_group"]]:
            fracs = torch.stack([(g == i).float().mean() for i in range(3)])
            actual_fracs.append(fracs)
        actual = torch.stack(actual_fracs).mean(dim=0)
        loss_routing = F.mse_loss(actual, target)

        total = (loss_cls
                 + self.cfg.lambda_consistency * loss_consistency
                 + self.cfg.lambda_routing * loss_routing)
        return total, {"cls": loss_cls.item(), "consistency": loss_consistency.item(),
                        "routing": loss_routing.item()}


# ==============================================================================
# METRICS
# ==============================================================================

def compute_metrics(y_true, y_pred, y_prob, num_classes) -> Dict[str, float]:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    fpr_list, fnr_list, spec_list = [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_list.append(fp / (fp + tn + 1e-8))
        fnr_list.append(fn / (fn + tp + 1e-8))
        spec_list.append(tn / (tn + fp + 1e-8))

    try:
        y_prob = np.array(y_prob)
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": float(np.mean(spec_list)),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": auc,
        "fpr_macro": float(np.mean(fpr_list)),
        "fnr_macro": float(np.mean(fnr_list)),
        "confusion_matrix": cm.tolist(),
    }


def run_anova(score_groups: List[List[float]]) -> Dict[str, float]:
    """One-way ANOVA across e.g. K-fold scores of different ablation variants."""
    f_stat, p_val = f_oneway(*score_groups)
    return {"F_statistic": float(f_stat), "p_value": float(p_val)}


# ==============================================================================
# TRAIN / EVAL LOOPS
# ==============================================================================

def run_epoch(model, loader, cfg, optimizer=None, criterion=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_true, all_pred, all_prob = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for ct, mri, labels in loader:
            ct, mri, labels = ct.to(cfg.device), mri.to(cfg.device), labels.to(cfg.device)
            logits, aux = model(ct, mri)
            if criterion is not None:
                loss, _ = criterion(logits, labels, aux)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * ct.size(0)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = compute_metrics(all_true, all_pred, all_prob, cfg.num_classes)
    return avg_loss, metrics


def make_loaders(cfg: Config, train_ids, val_ids, test_ids, synthetic: bool):
    train_ds = CardiacCTMRIDataset(cfg, train_ids, training=True, synthetic=synthetic)
    val_ds = CardiacCTMRIDataset(cfg, val_ids, training=False, synthetic=synthetic)
    test_ds = CardiacCTMRIDataset(cfg, test_ids, training=False, synthetic=synthetic)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(cfg: Config, train_loader, val_loader) -> Tuple[nn.Module, Dict[str, list]]:
    model = CardiacMultimodalNet(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = CombinedLoss(cfg)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        tr_loss, tr_metrics = run_epoch(model, train_loader, cfg, optimizer, criterion)
        val_loss, val_metrics = run_epoch(model, val_loader, cfg, optimizer=None, criterion=criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        print(f"[Epoch {epoch+1:02d}/{cfg.epochs}] "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={tr_metrics['accuracy']:.3f} val_acc={val_metrics['accuracy']:.3f}")
    return model, history


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close(fig)


def plot_confusion_matrix(cm, out_path, class_names=None):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    n = cm.shape[0]
    class_names = class_names or [str(i) for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close(fig)


def plot_token_routing_analysis(stats: Dict[str, float], out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["high_frac", "medium_frac", "low_frac"]
    ax.bar(keys, [stats.get(k, 0) for k in keys], color=["#2b6cb0", "#63b3ed", "#cbd5e0"])
    ax.set_title("Dynamic Token Routing: group distribution")
    ax.set_ylabel("fraction of tokens")
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close(fig)


def plot_expert_utilization(mean_weights: np.ndarray, expert_names: List[str], out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(expert_names, mean_weights, color=["#2f855a", "#d69e2e", "#c53030"][:len(expert_names)])
    ax.set_title("Expert Utilization Analysis")
    ax.set_ylabel("mean gating weight")
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close(fig)


def plot_modality_contribution(alphas: np.ndarray, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(alphas, bins=20, color="#805ad5", alpha=0.85)
    ax.axvline(alphas.mean(), color="black", linestyle="--", label=f"mean={alphas.mean():.2f}")
    ax.set_title("Adaptive CT-MRI Fusion: patient-wise CT contribution weight")
    ax.set_xlabel("alpha (CT weight)"); ax.set_ylabel("# patients"); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close(fig)


# ==============================================================================
# ABLATION STUDY (reproduces the three ablation tables in the concept doc)
# ==============================================================================

def ablation_variants() -> Dict[str, Dict[str, Dict[str, bool]]]:
    return {
        "Table1_DTIR": {
            "Without DTIE": {"use_dtie": False},
            "Without Gated MLP": {"use_gated_mlp": False},
            "Without DTR": {"use_dtr": False},
            "Without Top-k routing": {"use_topk_routing": False},
            "Without Adaptive token pruning": {"use_adaptive_pruning": False},
            "Full Proposed Model": {},
        },
        "Table2_HMoE": {
            "Without Window-based MHSA": {"use_window_mhsa": False},
            "Without MHSA": {"use_global_mhsa": False},
            "Without Linear Attention": {"use_linear_attn": False},
            "Without Expert Gating": {"use_expert_gating": False},
            "Without Adaptive expert fusion": {"use_adaptive_fusion": False},
            "Full Proposed Model": {},
        },
        "Table3_CMAF": {
            "Without Cross-Modality Projection Layer": {"use_cross_proj": False},
            "Without Cross-Attention": {"use_cross_attn": False},
            "Without AMEG": {"use_ameg": False},
            "Full Proposed Model": {},
        },
    }


def run_ablation_study(base_cfg: Config, train_ids, val_ids, test_ids, synthetic: bool):
    os.makedirs(base_cfg.out_dir, exist_ok=True)
    results = {}
    for table_name, variants in ablation_variants().items():
        print(f"\n===== {table_name} =====")
        results[table_name] = {}
        for variant_name, overrides in variants.items():
            cfg = Config(**{**asdict(base_cfg), **overrides})
            set_seed(cfg.seed)
            train_loader, val_loader, test_loader = make_loaders(cfg, train_ids, val_ids, test_ids, synthetic)
            model, _ = train_model(cfg, train_loader, val_loader)
            _, test_metrics = run_epoch(model, test_loader, cfg, optimizer=None, criterion=CombinedLoss(cfg))
            row = {k: test_metrics[k] for k in ["accuracy", "precision", "recall_sensitivity", "f1_score"]}
            results[table_name][variant_name] = row
            print(f"  {variant_name:38s} -> {row}")
    with open(os.path.join(base_cfg.out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ==============================================================================
# K-FOLD CROSS VALIDATION
# ==============================================================================

def run_kfold(cfg: Config, patient_ids: List[str], synthetic: bool, k: int = 5):
    labels_proxy = [abs(hash((pid, 0))) % cfg.num_classes for pid in patient_ids]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    fold_accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels_proxy)):
        print(f"\n===== Fold {fold + 1}/{k} =====")
        train_ids_full = [patient_ids[i] for i in train_idx]
        test_ids = [patient_ids[i] for i in test_idx]
        train_ids, val_ids = train_test_split(train_ids_full, test_size=0.15, random_state=cfg.seed)
        set_seed(cfg.seed + fold)
        train_loader, val_loader, test_loader = make_loaders(cfg, train_ids, val_ids, test_ids, synthetic)
        model, _ = train_model(cfg, train_loader, val_loader)
        _, test_metrics = run_epoch(model, test_loader, cfg, optimizer=None, criterion=CombinedLoss(cfg))
        fold_accs.append(test_metrics["accuracy"])
        print(f"Fold {fold + 1} test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nK-Fold mean accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    return fold_accs


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to real dataset; omit to use synthetic demo data.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ablation", action="store_true", help="Run the ablation study (Tables 1-3).")
    parser.add_argument("--kfold", type=int, default=0, help="If >0, run k-fold CV with this many folds.")
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.epochs:
        cfg.epochs = args.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    synthetic = cfg.data_root is None
    if synthetic:
        print("[INFO] No --data_root given: using synthetic CT/MRI demo data "
              "(replace with the Kaggle multimodal cardiac dataset for real results).")

    patient_ids = get_patient_ids(cfg)

    if args.kfold and args.kfold > 1:
        run_kfold(cfg, patient_ids, synthetic, k=args.kfold)
        return

    train_ids, val_ids, test_ids = patient_level_split(patient_ids, cfg)

    if args.ablation:
        run_ablation_study(cfg, train_ids, val_ids, test_ids, synthetic)
        return

    train_loader, val_loader, test_loader = make_loaders(cfg, train_ids, val_ids, test_ids, synthetic)
    model, history = train_model(cfg, train_loader, val_loader)

    print("\n===== Final Test Evaluation =====")
    test_loss, test_metrics = run_epoch(model, test_loader, cfg, optimizer=None, criterion=CombinedLoss(cfg))
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    # ---- collect diagnostic stats for the "Additional Graph" plots ----
    model.eval()
    all_ct_group, all_mri_group, all_alpha = [], [], []
    ct_expert_w_list, mri_expert_w_list, expert_names = [], [], ["local", "global", "efficient"]
    with torch.no_grad():
        for ct, mri, labels in test_loader:
            ct, mri = ct.to(cfg.device), mri.to(cfg.device)
            _, aux = model(ct, mri)
            all_ct_group.append(aux["ct_group"].cpu())
            all_mri_group.append(aux["mri_group"].cpu())
            all_alpha.append(aux["alpha_ct"].cpu())
            if aux["ct_expert_weights"] is not None:
                ct_expert_w_list.append(aux["ct_expert_weights"].reshape(-1, aux["ct_expert_weights"].shape[-1]).cpu())
            if aux["expert_names"]:
                expert_names = aux["expert_names"]

    routing_stats = DTR.stats(torch.cat(all_ct_group + all_mri_group))
    alphas = torch.cat(all_alpha).numpy()

    plot_training_curves(history, os.path.join(cfg.out_dir, "training_curves.png"))
    plot_confusion_matrix(test_metrics["confusion_matrix"], os.path.join(cfg.out_dir, "confusion_matrix.png"))
    plot_token_routing_analysis(routing_stats, os.path.join(cfg.out_dir, "token_routing_analysis.png"))
    plot_modality_contribution(alphas, os.path.join(cfg.out_dir, "ct_mri_contribution.png"))
    if ct_expert_w_list:
        mean_w = torch.cat(ct_expert_w_list, dim=0).mean(dim=0).numpy()
        plot_expert_utilization(mean_w, expert_names[:len(mean_w)],
                                 os.path.join(cfg.out_dir, "expert_utilization.png"))

    with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in test_metrics.items()}, f, indent=2)

    print(f"\nAll plots and metrics saved to: {cfg.out_dir}/")


if __name__ == "__main__":
    main()