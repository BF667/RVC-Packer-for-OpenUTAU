"""Export Score2HuBERT -> ONNX acoustic model (DiffSinger-compatible).

Wraps the trained S2H model with ONNX-friendly operations:
- Matrix-based LengthRegulator (replaces repeat_interleave)
- F0 Hz -> per-phoneme MIDI derivation (replaces explicit MIDI input)
- SP frame masking -> exact zeros (built-in post-mute signal)

DiffSinger protocol:
  Input:  tokens [1,N], durations [1,N], f0 [1,T] Hz
  Output: "mel" [1,T,768]  (actually HuBERT features)

Usage:
  python export_s2h.py --checkpoint <best.pt> --config <train.yaml> --output acoustic.onnx
"""

import os
import sys
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ONNX-friendly MultiheadAttention (replaces nn.MultiheadAttention for export)
# ---------------------------------------------------------------------------

class OnnxMHA(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention that uses dynamic shapes
    in all reshape ops, avoiding baked-in constants during ONNX trace."""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // mha.num_heads
        self.batch_first = mha.batch_first
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        if self.batch_first:
            B, L, D = query.shape
        else:
            L, B, D = query.shape

        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.batch_first:
            q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.reshape(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k.reshape(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.reshape(L, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scale = self.head_dim ** -0.5
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            attn_w = attn_w.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_w = F.softmax(attn_w, dim=-1)
        out = torch.matmul(attn_w, v)

        if self.batch_first:
            out = out.transpose(1, 2).reshape(B, L, D)
        else:
            out = out.permute(2, 0, 1, 3).reshape(L, B, D)

        out = self.out_proj(out)
        return out, None


def patch_mha_for_onnx(module: nn.Module):
    """Replace all nn.MultiheadAttention with OnnxMHA in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, OnnxMHA(child))
        else:
            patch_mha_for_onnx(child)


# ---------------------------------------------------------------------------
# ONNX-friendly S2H acoustic wrapper
# ---------------------------------------------------------------------------

class S2HAcousticOnnx(nn.Module):
    """Wraps trained S2H for ONNX export as DiffSinger acoustic model.

    Accepts standard DiffSinger inputs (tokens, durations, f0) and returns
    HuBERT features [1, T, 768] disguised as "mel".
    """

    SP_ID = 3
    POS_RESOLUTION = 1000

    def __init__(self, s2h_model):
        super().__init__()
        self.encoder = s2h_model.encoder
        self.f0_cond = s2h_model.f0_cond
        self.det_head = s2h_model.det_head
        self.hidden_dim = s2h_model.hidden_dim

        lr = s2h_model.length_reg
        self.register_buffer("pos_pe", lr.pos_pe)
        self.pos_proj = lr.pos_proj
        self.dur_proj = lr.dur_proj
        self.note_pos_proj = lr.note_pos_proj
        self.note_dur_proj = lr.note_dur_proj
        self.pos_scale = lr.pos_scale

    # -- helpers (all use standard ONNX ops) --------------------------------

    def _build_expansion(self, dur: torch.Tensor, T: int):
        """Build [T, N] binary expansion matrix from durations [N]."""
        N = dur.shape[0]
        cum = torch.cumsum(dur.float(), dim=0)                    # [N]
        lo = torch.cat([torch.zeros(1, device=dur.device), cum[:-1]])  # [N]
        idx = torch.arange(T, device=dur.device, dtype=torch.float32)  # [T]
        exp = ((idx.unsqueeze(1) >= lo.unsqueeze(0)) &
               (idx.unsqueeze(1) < cum.unsqueeze(0))).float()     # [T, N]
        return exp

    def _f0_to_midi(self, f0: torch.Tensor, exp: torch.Tensor):
        """Derive per-phoneme MIDI [N] from per-frame f0 [T] via expansion [T,N]."""
        voiced = (f0 > 50.0).float()                              # [T]
        exp_t = exp.t()                                            # [N, T]
        f0_sum = (exp_t * (f0 * voiced).unsqueeze(0)).sum(1)       # [N]
        count = (exp_t * voiced.unsqueeze(0)).sum(1)               # [N]
        mean_f0 = f0_sum / count.clamp(min=1.0)
        midi = (12.0 * torch.log2(mean_f0 / 440.0 + 1e-8) + 69.0)
        midi = midi.round().long().clamp(0, 127)
        midi = midi * (count > 0).long()
        return midi                                                # [N]

    def _length_regulate(self, ph: torch.Tensor, exp: torch.Tensor,
                         dur: torch.Tensor, midi: torch.Tensor):
        """ONNX-friendly length regulator using matrix multiply.

        ph:   [N, D]  phone hidden
        exp:  [T, N]  expansion matrix
        dur:  [N]     frame durations
        midi: [N]     MIDI pitch
        Returns: [T, D] frame hidden with position features
        """
        T, N = exp.shape
        device = ph.device
        D = ph.shape[1]

        # expand phone -> frame
        expanded = exp @ ph                                        # [T, D]

        # --- intra-phone position ---
        cum_exp = torch.cumsum(exp, dim=0)                         # [T, N]
        offset = (cum_exp * exp).sum(1) - 1.0                     # [T] local offset
        dur_per_frame = (exp @ dur.float().unsqueeze(1)).squeeze(1).clamp(min=1.0)
        pos_norm = offset / dur_per_frame                          # [0,1)
        pos_idx = (pos_norm * self.POS_RESOLUTION).long().clamp(0, self.POS_RESOLUTION)
        pos_feat = self.pos_proj(self.pos_pe[pos_idx])             # [T, D]

        log_dur = torch.log1p(dur_per_frame).unsqueeze(-1)         # [T, 1]
        dur_feat = self.dur_proj(log_dur)                          # [T, D]

        # --- intra-note position ---
        midi_f = midi.float()
        change = torch.cat([torch.ones(1, device=device),
                            (midi_f[1:] != midi_f[:-1]).float()])  # [N]
        note_idx = torch.cumsum(change, dim=0) - 1.0               # [N] float

        # per-phoneme note duration via N×N same-note matrix
        same = (note_idx.unsqueeze(0) == note_idx.unsqueeze(1)).float()  # [N, N]
        note_dur_per_phone = (same @ dur.float().unsqueeze(1)).squeeze(1)  # [N]

        # per-phoneme cumulative offset within note (lower-triangular mask)
        tri = torch.tril(torch.ones(N, N, device=device))
        cum_in_note = (same * tri) @ dur.float().unsqueeze(1)     # [N, 1]
        phone_note_offset = cum_in_note.squeeze(1) - dur.float()  # [N] start offset

        # expand to frames
        note_dur_frame = (exp @ note_dur_per_phone.unsqueeze(1)).squeeze(1).clamp(min=1.0)
        offset_in_note = (exp @ phone_note_offset.unsqueeze(1)).squeeze(1) + offset
        pos_in_note = offset_in_note / note_dur_frame

        npi = (pos_in_note * self.POS_RESOLUTION).long().clamp(0, self.POS_RESOLUTION)
        note_pos_feat = self.note_pos_proj(self.pos_pe[npi])       # [T, D]
        note_dur_feat = self.note_dur_proj(
            torch.log1p(note_dur_frame).unsqueeze(-1))             # [T, D]

        expanded = expanded + self.pos_scale * (
            pos_feat + dur_feat + note_pos_feat + note_dur_feat)
        return expanded                                            # [T, D]

    # -- main forward -------------------------------------------------------

    def forward(self, tokens: torch.Tensor, durations: torch.Tensor,
                f0: torch.Tensor, speedup: torch.Tensor) -> torch.Tensor:
        """
        tokens:    [1, N] int64  phoneme IDs
        durations: [1, N] int64  frames per phoneme
        f0:        [1, T] float  Hz per frame (from OpenUtau pitch curve)
        speedup:   [1]   int64   DiffSinger compat (ignored by S2H)
        Returns:   [1, T, 768]   "mel" (HuBERT features)
        """
        tok = tokens[0]                                            # [N]
        dur = durations[0]                                         # [N]
        f0_flat = f0[0]                                            # [T]
        T = f0.shape[1]
        _keep = speedup.float().sum() * 0

        exp = self._build_expansion(dur, T)                        # [T, N]
        midi = self._f0_to_midi(f0_flat, exp)                      # [N]
        midi = midi * (tok != self.SP_ID).long()                   # SP -> 0

        # encoder (phone-level)
        ph = self.encoder(tokens, midi.unsqueeze(0), durations)    # [1, N, D]

        # length regulate (frame-level)
        frame_h = self._length_regulate(ph[0], exp, dur, midi)     # [T, D]
        frame_h = frame_h.unsqueeze(0)                             # [1, T, D]

        # f0 conditioning — use OpenUtau's f0 directly
        f0_emb = self.f0_cond(f0)                                  # [1, T, D]

        # det head -> HuBERT
        hubert = self.det_head(frame_h, f0_emb)                   # [1, T, 768]

        # SP frame mask -> exact zeros (post-mute control signal)
        sp_phone = (tok == self.SP_ID).float()                     # [N]
        sp_frame = (exp @ sp_phone.unsqueeze(1)).squeeze(1)        # [T]
        mask = (sp_frame < 0.5).float().unsqueeze(0).unsqueeze(-1) # [1, T, 1]
        hubert = hubert * mask + _keep

        return hubert


# ---------------------------------------------------------------------------
# Export script
# ---------------------------------------------------------------------------

def load_s2h_model(ckpt_path: str, config_path: str, device: str = "cpu"):
    """Load trained S2H model from checkpoint + config."""
    import yaml

    s2h_root = Path(__file__).resolve().parents[1]
    project_root = Path(os.environ.get("S2H_ROOT", "score2hubert_v2"))
    sys.path.insert(0, str(project_root / "src"))

    from model.score2hubert import Score2HuBERT

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    model = Score2HuBERT(
        n_phones=model_cfg.get("n_phones", 80),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        hubert_dim=model_cfg.get("hubert_dim", 768),
        n_enc_layers=model_cfg.get("n_enc_layers", 5),
        n_heads=model_cfg.get("n_heads", 8),
        enc_kernel=model_cfg.get("enc_kernel", 31),
        n_det_blocks=model_cfg.get("n_det_blocks", 6),
        det_kernel=model_cfg.get("det_kernel", 7),
        dropout=0.0,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def export_onnx(s2h_model, output_path: str, opset: int = 17):
    """Export S2H as DiffSinger-compatible acoustic.onnx."""
    wrapper = S2HAcousticOnnx(s2h_model)
    patch_mha_for_onnx(wrapper)
    wrapper.eval()

    N, T = 20, 100
    dummy_tokens = torch.randint(0, 75, (1, N), dtype=torch.int64)
    dummy_dur = torch.randint(3, 10, (1, N), dtype=torch.int64)
    dummy_dur[0, -1] = T - dummy_dur[0, :-1].sum()
    dummy_f0 = torch.full((1, T), 440.0)
    dummy_speedup = torch.tensor([10], dtype=torch.int64)

    print(f"Exporting S2H acoustic -> {output_path}")
    print(f"  dummy: N={N} phonemes, T={T} frames")
    print(f"  patched {sum(1 for m in wrapper.modules() if isinstance(m, OnnxMHA))} MHA modules for ONNX")

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_dur, dummy_f0, dummy_speedup),
        output_path,
        input_names=["tokens", "durations", "f0", "speedup"],
        output_names=["mel"],
        dynamic_axes={
            "tokens": {1: "n_phones"},
            "durations": {1: "n_phones"},
            "f0": {1: "n_frames"},
            "mel": {1: "n_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  done -- {Path(output_path).stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export S2H -> acoustic.onnx")
    parser.add_argument("--checkpoint", type=str,
                        required=True)
    parser.add_argument("--config", type=str,
                        required=True)
    parser.add_argument("--output", type=str, default="output/acoustic.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model = load_s2h_model(args.checkpoint, args.config)
    export_onnx(model, args.output, args.opset)


if __name__ == "__main__":
    main()
