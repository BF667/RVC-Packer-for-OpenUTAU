"""Export RVC → ONNX vocoder model (DiffSinger-compatible).

Wraps an RVC v2 model (SynthesizerTrnMs768NSFsid) as a DiffSinger vocoder:
  Input:  "mel" [1,T,768] (actually HuBERT from S2H) + f0 [1,T] Hz
  Output: waveform [1, T_audio]

Bakes in:
  - Index retrieval (brute-force KNN from constant big_npy tensor)
  - 50fps→100fps upsampling
  - F0 Hz → coarse pitch encoding (mel-scale 0-255)
  - Stochastic inference (randn * 0.66666, matching RVC infer())

Usage:
  python export_rvc.py --model <rvc.pth> --output vocoder.onnx [--index <.index>] [--index-rate 0.75]
"""

import os
import sys
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RVC vocoder ONNX wrapper
# ---------------------------------------------------------------------------

class RVCVocoderOnnx(nn.Module):
    """Wraps RVC net_g for ONNX export as DiffSinger vocoder.

    DiffSinger vocoder protocol: mel [1,T,bins] + f0 [1,T] → waveform [1,N].
    Here "mel" is actually 768-dim HuBERT features from S2H.
    """

    F0_MIN_HZ = 50.0
    F0_MAX_HZ = 1100.0

    def __init__(self, net_g_onnx, big_npy=None, index_rate=0.75,
                 sample_rate=40000, has_f0=True):
        super().__init__()
        self.net_g = net_g_onnx
        self.index_rate = index_rate
        self.sample_rate = sample_rate
        self.has_f0 = has_f0

        if big_npy is not None:
            npy = big_npy.astype(np.float32)
            norms = np.linalg.norm(npy, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            npy_normed = npy / norms
            self.register_buffer("big_npy", torch.from_numpy(npy))
            self.register_buffer("big_npy_norm", torch.from_numpy(npy_normed))
            self.has_index = True
        else:
            self.has_index = False

    def _index_retrieve(self, feats: torch.Tensor) -> torch.Tensor:
        """Brute-force KNN index retrieval baked as matrix ops.

        feats: [1, T, 768] → blended [1, T, 768]
        """
        if not self.has_index or self.index_rate <= 0:
            return feats

        f = feats[0]                                         # [T, 768]
        # Manual L2 norm instead of F.normalize — avoids ONNX ReduceL2 op
        # which has compatibility issues with Microsoft.ML.OnnxRuntime
        # (axes attribute vs input mismatch across opset versions)
        f_norm = f / f.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)  # [T, 768]
        sim = f_norm @ self.big_npy_norm.t()                 # [T, N_idx]

        topk_sim, topk_idx = sim.topk(8, dim=-1)             # [T, 8]

        # Match FAISS L2 inverse-square weighting:
        # L2^2 = 2*(1-cos) for normalized vectors, weight = 1/L2^4
        dist_sq = 2.0 * (1.0 - topk_sim + 1e-6)             # [T, 8]
        weights = 1.0 / (dist_sq * dist_sq)                  # [T, 8]
        weights = weights / weights.sum(dim=-1, keepdim=True) # [T, 8]

        gathered = self.big_npy[topk_idx]                     # [T, 8, 768]
        retrieved = (gathered * weights.unsqueeze(-1)).sum(1)  # [T, 768]

        blended = retrieved * self.index_rate + f * (1.0 - self.index_rate)
        return blended.unsqueeze(0)                           # [1, T, 768]

    def _f0_to_coarse(self, f0: torch.Tensor) -> torch.Tensor:
        """Convert F0 Hz → mel-scale coarse pitch [0-255].

        f0: [1, T] → pitch: [1, T] int64
        """
        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel_min = 1127.0 * math.log(1.0 + self.F0_MIN_HZ / 700.0)
        f0_mel_max = 1127.0 * math.log(1.0 + self.F0_MAX_HZ / 700.0)
        f0_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
        coarse = (f0_norm * 254.0 + 1.0).round().long().clamp(1, 255)
        coarse = coarse * (f0 > self.F0_MIN_HZ).long()
        return coarse

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        mel: [1, T_50fps, 768]  HuBERT features from S2H at 50fps
        f0:  [1, T_50fps]       F0 in Hz at 50fps (ignored for f0=0 models)
        Returns: waveform [1, T_audio]
        """
        mel = self._index_retrieve(mel)

        # upsample 50fps -> 100fps
        feat = mel.transpose(1, 2)                             # [1, 768, T_50]
        feat_2x = F.interpolate(feat, scale_factor=2.0,
                                mode="linear",
                                align_corners=False)           # [1, 768, T_100]
        feat_2x = feat_2x.transpose(1, 2)                     # [1, T_100, 768]

        T_100 = feat_2x.shape[1]
        p_len = torch.tensor([T_100], dtype=torch.int64)
        sid = torch.zeros(1, dtype=torch.int64)

        if self.has_f0:
            f0_2x = F.interpolate(f0.unsqueeze(1), scale_factor=2.0,
                                   mode="linear",
                                   align_corners=False).squeeze(1) # [1, T_100]
            pitch = self._f0_to_coarse(f0_2x)                     # [1, T_100]
            audio = self.net_g(feat_2x, p_len, pitch, f0_2x, sid)
        else:
            # f0=0 / pure HiFi-GAN: no pitch conditioning
            audio = self.net_g(feat_2x, p_len, sid)

        # audio: [1, 1, T_audio]
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        return audio                                           # [1, T_audio]


# ---------------------------------------------------------------------------
# Fixed-size wrapper (relative attention bakes reshape constants at trace time)
# ---------------------------------------------------------------------------

class FixedLenVocoder(nn.Module):
    """Wraps RVC for ONNX export with dynamic-length support.

    No padding — enc_p runs at actual input length. The relative attention
    exports as dynamic ONNX ops (pad/slice/reshape computed from runtime
    shape), verified to work at any length.

    Final resample to 44100 Hz because OpenUtau hardcodes 44100 in its
    entire audio pipeline (WaveSource, MasterAdapter, ExportAdapter).
    """

    OPENUTAU_SR = 44100

    def __init__(self, voc, max_t, model_sr, has_f0=True):
        super().__init__()
        self.voc = voc
        self.max_t = max_t
        self.resample_ratio = self.OPENUTAU_SR / model_sr
        self.has_f0 = has_f0

    def forward(self, mel, f0):
        # Note: F0 silence masking is handled by patch_f0_silence_mask() at
        # ONNX level after export.  We no longer duplicate it here, avoiding
        # redundant double-masking and ensuring a single consistent threshold.

        mel = self.voc._index_retrieve(mel)

        feat = mel.transpose(1, 2)
        feat_2x = F.interpolate(feat, scale_factor=2.0,
                                mode="linear",
                                align_corners=False)
        feat_2x = feat_2x.transpose(1, 2)

        p_len = torch.tensor([10000], dtype=torch.int64)
        sid = torch.zeros(1, dtype=torch.int64)

        if self.has_f0:
            f0_2x = F.interpolate(f0.unsqueeze(1), scale_factor=2.0,
                                   mode="linear",
                                   align_corners=False).squeeze(1)
            pitch = self.voc._f0_to_coarse(f0_2x)
            audio = self.voc.net_g(feat_2x, p_len, pitch, f0_2x, sid)
        else:
            # f0=0 / pure HiFi-GAN: no pitch conditioning
            audio = self.voc.net_g(feat_2x, p_len, sid)

        if audio.dim() == 3:
            audio = audio.squeeze(1)

        if self.resample_ratio != 1.0:
            audio = F.interpolate(audio.unsqueeze(1),
                                  scale_factor=self.resample_ratio,
                                  mode="linear",
                                  align_corners=False).squeeze(1)

        return audio


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def _patch_net_g_infer(net_g, has_f0=True):
    """Monkey-patch net_g.forward to replicate infer() for ONNX export.

    Uses torch.randn_like for noise (exports as ONNX RandomNormalLike,
    dynamic shape). Matches RVC infer() exactly: noise_scale=0.66666.

    For f0=0 (pure HiFi-GAN) models, the decoder receives no pitch/F0
    input — it is a plain Generator rather than GeneratorNSF.
    """
    if has_f0:
        def patched_forward(phone, phone_lengths, pitch, nsff0, sid,
                            max_len=None):
            g = net_g.emb_g(sid.unsqueeze(0)).transpose(1, 2)
            m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = net_g.flow(z_p, x_mask, g=g, reverse=True)
            o = net_g.dec(z * x_mask, nsff0, g=g)
            return o
    else:
        def patched_forward(phone, phone_lengths, sid,
                            max_len=None):
            g = net_g.emb_g(sid.unsqueeze(0)).transpose(1, 2)
            m_p, logs_p, x_mask = net_g.enc_p(phone, None, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = net_g.flow(z_p, x_mask, g=g, reverse=True)
            o = net_g.dec(z * x_mask, g=g)
            return o

    net_g.forward = patched_forward


def _load_config_json(model_path: str):
    """Load RVC config from config.json next to the checkpoint."""
    import json
    cfg_path = Path(model_path).parent / "config.json"
    if not cfg_path.exists():
        return None, None
    with open(cfg_path, "r") as f:
        j = json.load(f)
    m = j["model"]
    d = j["data"]
    sr = d["sampling_rate"]
    config = [
        d["filter_length"] // 2 + 1,
        32,
        m["inter_channels"], m["hidden_channels"], m["filter_channels"],
        m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"],
        m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"],
        m["upsample_rates"], m["upsample_initial_channel"],
        m["upsample_kernel_sizes"], m["spk_embed_dim"], m["gin_channels"],
    ]
    return config, sr


def load_rvc_model(model_path: str, device: str = "cpu"):
    """Load RVC model and return (net_g_onnx, config_dict).

    Automatically selects the correct model class based on f0 support:
    - f0=1: SynthesizerTrnMs768NSFsid (NSF + HiFi-GAN decoder)
    - f0=0: SynthesizerTrnMs768NSFsid_nono (pure HiFi-GAN decoder)
    """
    rvc_root = Path(os.environ.get("RVC_ROOT", "RVC"))
    sys.path.insert(0, str(rvc_root))
    sys.path.insert(0, str(rvc_root / "infer" / "lib"))

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", None)
    if config is not None:
        version = ckpt.get("version", "v2")
        sr_key = config[-1] if isinstance(config[-1], str) else str(config[-1])
        sr_map = {"32k": 32000, "40k": 40000, "48k": 48000}
        sr = sr_map.get(sr_key, int(config[-1]) if isinstance(config[-1], (int, float)) else 40000)
    else:
        config, sr = _load_config_json(model_path)
        if config is None:
            raise ValueError("No config in checkpoint and no config.json found")
        version = ckpt.get("version", "v2")

    # Detect f0 support from checkpoint metadata
    if_f0 = ckpt.get("f0", 1)

    try:
        if if_f0:
            from infer_pack.models import SynthesizerTrnMs768NSFsid as SynthClass
        else:
            from infer_pack.models import SynthesizerTrnMs768NSFsid_nono as SynthClass
    except ImportError:
        # Fallback: try the unified ONNX model class
        try:
            from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM as SynthClass
        except ImportError:
            from infer_pack.models import SynthesizerTrnMs768NSFsid as SynthClass

    net_g = SynthClass(
        *config[:-1] if ckpt.get("config") else config,
        sr=sr,
        version=version,
        is_half=False,
    )

    for key in ("weight", "model", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        state = ckpt
    model_keys = set(net_g.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in model_keys}
    net_g.load_state_dict(filtered, strict=False)
    net_g.remove_weight_norm()
    net_g.eval()

    return net_g, {"sample_rate": sr, "version": version, "f0": bool(if_f0)}


def load_index(index_path: str):
    """Load faiss index and reconstruct big_npy."""
    try:
        import faiss
        idx = faiss.read_index(index_path)
        big_npy = idx.reconstruct_n(0, idx.ntotal)
        print(f"  index loaded: {idx.ntotal} vectors, shape {big_npy.shape}")
        return big_npy
    except Exception as e:
        print(f"  index load failed ({e}), will export without index")
        return None


def export_onnx(rvc_model, output_path: str, big_npy=None,
                index_rate=0.75, sample_rate=40000, opset=17,
                max_frames=500, as_template=False):
    """Export RVC as DiffSinger-compatible vocoder model.onnx.

    Args:
        max_frames: dummy input length in 50fps frames for tracing.
        as_template: if True, disables constant folding so all weight
            initializers keep their original names (required for the
            packer's weight-patching approach).
    """
    from attn_patch import patch_attention_for_onnx

    # Detect f0 from model info (default True for backward compat)
    has_f0 = info.get("f0", True)

    _patch_net_g_infer(rvc_model, has_f0=has_f0)
    patch_attention_for_onnx(rvc_model)

    wrapper = RVCVocoderOnnx(rvc_model, big_npy=big_npy,
                              index_rate=index_rate,
                              sample_rate=sample_rate,
                              has_f0=has_f0)
    wrapper.eval()

    traceable = FixedLenVocoder(wrapper, max_frames, model_sr=sample_rate,
                                has_f0=has_f0)
    traceable.eval()

    T = max_frames + 10
    dummy_mel = torch.randn(1, T, 768)
    dummy_f0 = torch.full((1, T), 440.0)

    print(f"Exporting RVC vocoder -> {output_path}")
    print(f"  model_sr={sample_rate} -> output_sr=44100 (resample ratio={44100/sample_rate:.4f})")

    torch.onnx.export(
        traceable,
        (dummy_mel, dummy_f0),
        output_path,
        input_names=["mel", "f0"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {1: "n_frames"},
            "f0": {1: "n_frames"},
            "waveform": {1: "n_samples"},
        },
        opset_version=opset,
        do_constant_folding=not as_template,
    )
    print(f"  done -- {Path(output_path).stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export RVC -> vocoder model.onnx")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to RVC .pth model")
    parser.add_argument("--index", type=str, default=None,
                        help="Path to .index file (optional)")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--output", type=str, default="output/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max input frames at 50fps (default 500 = 10s)")
    parser.add_argument("--template", action="store_true",
                        help="Export as packer template (no constant folding)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    net_g, info = load_rvc_model(args.model)
    sr = info["sample_rate"]

    big_npy = None
    if args.index:
        big_npy = load_index(args.index)

    export_onnx(net_g, args.output, big_npy=big_npy,
                index_rate=args.index_rate, sample_rate=sr, opset=args.opset,
                max_frames=args.max_frames, as_template=args.template)
    print(f"\nVoice bank config: sample_rate={sr}, hop_size={sr // 50}")


if __name__ == "__main__":
    main()
