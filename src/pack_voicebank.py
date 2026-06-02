"""Pack an RVC model into an OpenUtau DiffSinger voice bank.

Zero torch dependency. Uses:
  - pth_reader: load .pth as numpy state dict
  - onnx_patcher: merge weight_norm, patch ONNX template

Expects pre-exported assets (shipped with the packer):
  assets/acoustic.onnx         — S2H acoustic model (shared)
  assets/templates/v2_40k.onnx — RVC ONNX template for 40kHz v2
  assets/templates/v2_48k.onnx — RVC ONNX template for 48kHz v2
  phoneme_map/phonemes.txt     — IPA phoneme list
  phoneme_map/dsdict-*.yaml    — phoneme dictionaries
"""

import shutil
from pathlib import Path

import numpy as np

try:
    from pth_reader import load_pth
    from onnx_patcher import (process_state_dict, patch_onnx_template,
                               patch_f0_silence_mask, fix_reduce_l2_nodes,
                               zero_nsf_weights)
except ImportError:
    from .pth_reader import load_pth
    from .onnx_patcher import (process_state_dict, patch_onnx_template,
                                patch_f0_silence_mask, fix_reduce_l2_nodes,
                                zero_nsf_weights)

import sys as _sys
if getattr(_sys, 'frozen', False):
    # sys._MEIPASS points to the _internal directory in PyInstaller 6+
    # or the one-file temp directory in --onefile mode.
    ADAPTER_ROOT = Path(_sys._MEIPASS)
else:
    ADAPTER_ROOT = Path(__file__).resolve().parents[1]

# Ensure ADAPTER_ROOT assets are accessible (PyInstaller 6+ _internal layout)
_assets_dir = ADAPTER_ROOT / "assets"
if not _assets_dir.exists() and getattr(_sys, 'frozen', False):
    _parent_root = Path(_sys._MEIPASS).parent
    if (_parent_root / "assets").exists():
        ADAPTER_ROOT = _parent_root
    elif (Path(_sys.executable).parent / "assets").exists():
        ADAPTER_ROOT = Path(_sys.executable).parent

ASSETS_DIR = ADAPTER_ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
PHONEME_DIR = ADAPTER_ROOT / "phoneme_map"

# RVC config → sample rate mapping
_SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def detect_rvc_config(full_ckpt: dict, state_dict: dict = None) -> dict:
    """Extract RVC model configuration from checkpoint metadata.

    Also detects f0=0 (pure HiFi-GAN) models by inspecting weight keys
    when metadata is ambiguous.  f0=1 models use GeneratorNSF (NSF+HiFi-GAN
    decoder with m_source and noise_convs); f0=0 models use Generator
    (pure HiFi-GAN decoder, no pitch conditioning).
    """
    config = full_ckpt.get("config", [])
    version = full_ckpt.get("version", "v2")

    # sample rate is last element of config list
    sr_raw = config[-1] if config else 40000
    if isinstance(sr_raw, str):
        sr = _SR_MAP.get(sr_raw, 40000)
    else:
        sr = int(sr_raw)

    # f0 support: check metadata first, then fall back to weight keys
    f0 = full_ckpt.get("f0", None)
    if f0 is None and state_dict is not None:
        # Detect from weight keys: NSF models have m_source / noise_convs
        has_nsf = any(k.startswith("dec.m_source.") for k in state_dict)
        has_noise_convs = any(k.startswith("dec.noise_convs.") for k in state_dict)
        has_emb_pitch = "enc_p.emb_pitch.weight" in state_dict
        f0 = 1 if (has_nsf or has_noise_convs or has_emb_pitch) else 0
    elif f0 is None:
        f0 = 1  # default assumption

    # Detect version from weight keys if not in metadata
    if state_dict is not None and version not in ("v1", "v2"):
        emb_phone = state_dict.get("enc_p.emb_phone.weight")
        if emb_phone is not None and hasattr(emb_phone, 'shape'):
            in_dim = emb_phone.shape[1] if len(emb_phone.shape) == 2 else emb_phone.shape[-1]
            version = "v1" if in_dim < 768 else "v2"

    decoder_type = "GeneratorNSF" if f0 else "Generator"

    return {
        "version": version,
        "sample_rate": sr,
        "f0": bool(f0),
        "config": config,
        "decoder_type": decoder_type,
    }


def select_template(rvc_info: dict) -> Path:
    """Select the correct ONNX template for this RVC config.

    f0=1 models (GeneratorNSF/NSF-HiFiGAN) use the standard templates.
    f0=0 models (Generator/pure HiFi-GAN) use '_nono' templates which
    have no NSF source module or noise_convs in the decoder graph.
    """
    v = rvc_info["version"]
    sr = rvc_info["sample_rate"]
    f0 = rvc_info.get("f0", True)
    sr_tag = {32000: "32k", 40000: "40k", 48000: "48k"}.get(sr, f"{sr}")

    # f0=0 (HiFi-GAN only) models need _nono templates
    suffix = "" if f0 else "_nono"
    name = f"{v}_{sr_tag}{suffix}.onnx"
    path = TEMPLATES_DIR / name
    if path.exists():
        return path

    # Fallback: if _nono template doesn't exist, try the standard template.
    # The ONNX patcher will simply skip missing initializers (m_source, noise_convs)
    # and the silence mask layer 1 (F0 masking) will be skipped for f0=0 models.
    if not f0:
        fallback = TEMPLATES_DIR / f"{v}_{sr_tag}.onnx"
        if fallback.exists():
            return fallback

    avail = [f.name for f in TEMPLATES_DIR.glob("*.onnx")] if TEMPLATES_DIR.exists() else []
    raise FileNotFoundError(
        f"No ONNX template for {v}/{sr_tag}/f0={int(f0)}. "
        f"Available: {avail or 'none — run dev/create_templates.py first'}")


def _write_character_txt(out: Path, name: str, author: str, avatar: str):
    lines = [f"name={name}"]
    if author:
        lines.append(f"author={author}")
    if avatar:
        lines.append(f"image={Path(avatar).name}")
    (out / "character.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_character_yaml(out: Path, name: str, lang: str):
    pm = {"zh": "OpenUtau.Core.DiffSinger.DiffSingerChinesePhonemizer",
          "ja": "OpenUtau.Core.DiffSinger.DiffSingerJapanesePhonemizer"}
    (out / "character.yaml").write_text(
        f'Name: {name}\nSingerType: diffsinger\n'
        f'DefaultPhonemizer: {pm.get(lang, "OpenUtau.Core.DiffSinger.DiffSingerPhonemizer")}\n'
        f'Subbanks:\n  - Color: ""\n    Prefix: ""\n    Suffix: ""\n',
        encoding="utf-8")


def _write_dsconfig(out: Path, sr: int, lang: str):
    # OpenUtau hardcodes 44100 Hz in its entire audio pipeline.
    # Vocoder ONNX resamples internally from model_sr to 44100.
    # dsconfig and vocoder.yaml must both say 44100 to match.
    output_sr = 44100
    hop = output_sr // 50  # 882
    dd = f"dsdict-{lang}.yaml" if lang in ("zh", "ja") else "dsdict.yaml"
    (out / "dsconfig.yaml").write_text(
        f"phonemes: phonemes.txt\nacoustic: acoustic.onnx\nvocoder: dsvocoder\n"
        f"sample_rate: {output_sr}\nhop_size: {hop}\n"
        f"win_size: 2048\nfft_size: 2048\n"
        f"num_mel_bins: 768\nmel_fmin: 40\nmel_fmax: 16000\n"
        f'mel_base: "e"\nmel_scale: slaney\n'
        f"predict_dur: false\npredict_pitch: false\n"
        f"predict_voicing: false\npredict_breathiness: false\n"
        f"predict_tension: false\npredict_energy: false\n"
        f"dsdict: {dd}\n",
        encoding="utf-8")


def _write_vocoder_yaml(vdir: Path, sr: int):
    output_sr = 44100
    hop = output_sr // 50  # 882
    (vdir / "vocoder.yaml").write_text(
        f"name: s2h_rvc_vocoder\nmodel: model.onnx\n"
        f"num_mel_bins: 768\nhop_size: {hop}\nwin_size: 2048\nfft_size: 2048\n"
        f"sample_rate: {output_sr}\n"
        f"mel_fmin: 40\nmel_fmax: 16000\n"
        f'mel_base: "e"\nmel_scale: slaney\n',
        encoding="utf-8")


def pack_voicebank(
    rvc_pth_path: str,
    output_dir: str,
    voice_name: str,
    language: str = "ja",
    index_npy_path: str = None,
    index_rate: float = 0.75,
    author: str = "",
    avatar_path: str = "",
    on_progress=None,
) -> Path:
    """Full pipeline: .pth → OpenUtau voice bank directory.

    Args:
        rvc_pth_path: path to RVC .pth model
        output_dir: where to create the voice bank folder
        voice_name: display name for the singer
        language: "ja" or "zh"
        index_npy_path: path to .npy file with index vectors (optional)
        index_rate: blending rate for index retrieval (0-1)
        author: optional author name
        avatar_path: optional avatar image path
        on_progress: optional callback(message: str)

    Returns: Path to the created voice bank directory
    """
    def log(msg):
        if on_progress:
            on_progress(msg)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Read .pth
    log("Reading RVC model...")
    state, full_ckpt = load_pth(rvc_pth_path)
    rvc_info = detect_rvc_config(full_ckpt, state_dict=state)
    sr = rvc_info["sample_rate"]
    log(f"  Detected: {rvc_info['version']}, {sr}Hz, f0={rvc_info['f0']}, "
        f"decoder={rvc_info.get('decoder_type', 'GeneratorNSF')}")

    # 2. Process weights (merge weight_norm, fp32)
    log("Processing weights...")
    processed = process_state_dict(state)
    log(f"  {len(processed)} parameters ready")

    # 3. Select and patch ONNX template
    template = select_template(rvc_info)
    log(f"Using template: {template.name}")

    vocoder_onnx = out / "dsvocoder" / "model.onnx"
    vocoder_onnx.parent.mkdir(parents=True, exist_ok=True)

    # patch weights into template (use weight map for onnx_name -> pth_name)
    weight_map_path = str(template.with_suffix(".json"))
    log("Patching ONNX model...")
    patch_onnx_template(str(template), processed, str(vocoder_onnx),
                        weight_map_path=weight_map_path)

    # 4. Bake index if provided
    if index_npy_path and Path(index_npy_path).exists():
        log("Baking index vectors...")
        try:
            from onnx_patcher import bake_index_into_onnx
        except ImportError:
            from .onnx_patcher import bake_index_into_onnx
        bake_index_into_onnx(str(vocoder_onnx), index_npy_path,
                              index_rate, str(vocoder_onnx))

    # 4b. Inject F0 silence mask into vocoder graph
    log("Injecting F0 silence mask...")
    patch_f0_silence_mask(str(vocoder_onnx), has_f0=rvc_info["f0"])

    # 4c. Fix ReduceL2 nodes for Microsoft.ML.OnnxRuntime compatibility
    log("Fixing ONNX compatibility (ReduceL2)...")
    fix_reduce_l2_nodes(str(vocoder_onnx))

    # 4d. For f0=0 / pure HiFi-GAN models, zero out NSF source weights
    # so the decoder behaves as a plain HiFi-GAN Generator
    if not rvc_info["f0"]:
        log("Zeroing NSF weights for HiFi-GAN model...")
        zero_nsf_weights(str(vocoder_onnx))

    # 5. Write vocoder config
    _write_vocoder_yaml(vocoder_onnx.parent, sr)

    # 6. Copy acoustic model (shared)
    acoustic_src = ASSETS_DIR / "acoustic.onnx"
    if acoustic_src.exists():
        shutil.copy2(acoustic_src, out / "acoustic.onnx")
    else:
        log(f"  WARNING: {acoustic_src} not found!")

    # 7. Copy phoneme files
    ph_src = PHONEME_DIR / "phonemes.txt"
    if ph_src.exists():
        shutil.copy2(ph_src, out / "phonemes.txt")

    for lang_tag in ("zh", "ja"):
        dd_src = PHONEME_DIR / f"dsdict-{lang_tag}.yaml"
        if dd_src.exists():
            shutil.copy2(dd_src, out / f"dsdict-{lang_tag}.yaml")

    # 7b. Copy dsdur (phonemizer duration models)
    dsdur_src = ASSETS_DIR / "dsdur"
    if dsdur_src.is_dir():
        dsdur_dst = out / "dsdur"
        if dsdur_dst.exists():
            shutil.rmtree(dsdur_dst)
        shutil.copytree(dsdur_src, dsdur_dst)
        log(f"  dsdur copied")
    else:
        log(f"  WARNING: {dsdur_src} not found!")

    # 8. Write metadata
    _write_character_txt(out, voice_name, author,
                          avatar_path if avatar_path else "")
    _write_character_yaml(out, voice_name, language)
    _write_dsconfig(out, sr, language)

    # 9. Copy avatar
    if avatar_path and Path(avatar_path).exists():
        shutil.copy2(avatar_path, out / Path(avatar_path).name)

    log(f"\nVoice bank ready: {out}")
    log("Copy to OpenUtau/Singers/ to use.")
    return out
