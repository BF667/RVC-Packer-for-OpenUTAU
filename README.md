# S2H OpenUtau Adapter

Enables [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) singing voice synthesis in [OpenUtau](https://github.com/stakira/OpenUtau) via the DiffSinger protocol.

Uses Score2HuBERT (S2H) as the acoustic model and RVC v2 as the vocoder, wrapped to match OpenUtau's DiffSinger interface. Includes a standalone GUI packer (**no PyTorch required**) that converts any RVC `.pth` into a ready-to-use OpenUtau voice bank.

**Website:** https://s2h.utaisynthesizer.net/ | **QQ Group:** [1097227285](https://qun.qq.com/universal-share/share?ac=1&authKey=uEve5stNVkxDeR8np%2B7T0982%2FR1%2FKswgdbId8%2FEzeLIVaXxinWsTKzGGE3yscUdq&busi_data=eyJncm91cENvZGUiOiIxMDk3MjI3Mjg1IiwidG9rZW4iOiJOZHo5QWJXT0dSbUNyc3NEQy95cTZRSWErTGh6S0FBWHhyUTIycXZYUnpGK29OOUZYYlZlQzU4UlJSNDZMSVJkIiwidWluIjoiMzU1Mjk1ODA4MCJ9&data=uEQD2Yzi1YQ2yYvRP8bgDKa1jlp8i3ifEaiVyoZ8YPimOoqOK6nfkFyKFpuW0mbWBjR2sFoIX64NCAvbkRkd9w&svctype=4&tempid=h5_group_info) | **Discord:** [Join](https://discord.com/invite/nK4KSEWMPU)

## Architecture

```
OpenUtau DiffSinger protocol
─────────────────────────────────────────────────────────────────

  tokens [1,N] + durations [1,N] + f0 [1,T] + speedup [1]
      │
      ▼
  ┌─────────────────────────────────────────────────────────┐
  │  acoustic.onnx  (S2H — shared across all voice banks)  │
  │  Score → HuBERT feature predictor                      │
  │  Matrix-based length regulator, F0→MIDI derivation     │
  └─────────────────────┬───────────────────────────────────┘
                        │  HuBERT features [1, T, 768]
                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  dsvocoder/model.onnx  (RVC — per-voice identity)      │
  │  Index retrieval → 50→100fps upsample → F0→pitch       │
  │  → enc_p → flow → decoder → resample to 44100 Hz      │
  └─────────────────────┬───────────────────────────────────┘
                        │  waveform [1, T×882] at 44100 Hz
                        ▼
                    OpenUtau playback
```

**S2H acoustic model** — voice-independent, predicts 768-dim HuBERT features from musical score input. Derives MIDI pitch from the F0 curve (DiffSinger protocol doesn't provide MIDI). Exported once, shared by all voice banks.

**RVC vocoder** — encodes the singer's voice identity. Bakes KNN index retrieval as matrix ops (replaces FAISS), upsamples 50fps→100fps, resamples output to 44100 Hz (OpenUtau hardcodes this).

## Quick Start (GUI Packer)

Download `RVC_VoiceBank_Packer.exe` from [Releases](#). No Python or PyTorch required.

1. Select an RVC `.pth` model
2. *(Optional)* Select a `.index` file for voice timbre retrieval
3. Set voice name, language (日本語 / 中文), output directory
4. Click **Export**
5. Copy the output folder to `OpenUtau/Singers/`

The GUI supports three languages (English / 日本語 / 中文).

## Developer Usage

Requires Python 3.10+, PyTorch 2.x, numpy, onnx. See [Requirements](#requirements) for details.

### Export S2H acoustic model (one-time)

```bash
python src/export_s2h.py \
  --checkpoint path/to/best.pt \
  --config path/to/train.yaml \
  --output assets/acoustic.onnx
```

### Export RVC vocoder template (one-time per architecture)

```bash
python src/export_rvc.py \
  --model path/to/rvc_v2_40k.pth \
  --index path/to/file.index \
  --output assets/templates/v2_40k.onnx \
  --template
```

The `--template` flag disables constant folding, preserving weight initializer names for the packer's weight-patching approach.

### Export RVC vocoder directly (per-voice, for testing)

```bash
python src/export_rvc.py \
  --model path/to/rvc.pth \
  --index path/to/file.index \
  --index-rate 0.75 \
  --output output/model.onnx
```

### Generate phoneme maps

```bash
python src/generate_phoneme_map.py
```

Generates `phonemes.txt`, `dsdict-ja.yaml`, and `dsdict-zh.yaml` from the S2H IPA vocabulary.

### Build standalone packer (PyInstaller)

```bash
pyinstaller packer.spec
```

Produces `dist/RVC_VoiceBank_Packer/RVC_VoiceBank_Packer.exe`. Bundles numpy + onnx + tkinter; excludes PyTorch, scipy, and CUDA.

## Project Structure

```
S2H_OpenUtau_Adapter/
├── src/
│   ├── export_s2h.py           # S2H → acoustic.onnx (DiffSinger acoustic)
│   ├── export_rvc.py           # RVC → model.onnx (DiffSinger vocoder)
│   ├── pack_voicebank.py       # .pth → complete OpenUtau voice bank
│   ├── onnx_patcher.py         # Patch weights into ONNX templates (numpy only)
│   ├── pth_reader.py           # Read .pth files without torch dependency
│   ├── attn_patch.py           # Dynamic attention shapes for ONNX export
│   ├── generate_phoneme_map.py # IPA phoneme map generator
│   └── packer_gui.py           # Tkinter GUI (EN/JA/ZH)
├── assets/
│   ├── acoustic.onnx           # Pre-exported S2H acoustic (shared)
│   ├── dsdur/                  # Duration model for DiffSinger phonemizer
│   │   ├── dsconfig.yaml       # Phonemizer config (40000 Hz / hop 800)
│   │   ├── dur.onnx            # Duration predictor
│   │   └── linguistic.onnx     # Linguistic encoder
│   └── templates/
│       ├── v2_40k.onnx + .json # RVC v2 40 kHz template + weight map
│       └── v2_48k.onnx + .json # RVC v2 48 kHz template + weight map
├── phoneme_map/
│   ├── phonemes.txt            # IPA token list (line number = token ID)
│   ├── dsdict-ja.yaml          # Japanese romaji → IPA
│   └── dsdict-zh.yaml          # Chinese pinyin → IPA
├── packer.spec                 # PyInstaller build config
└── dist/                       # Built executable
```

## Technical Details

### OpenUtau 44100 Hz Constraint

OpenUtau hardcodes 44100 Hz in its entire audio pipeline (`WaveSource.cs`, `MasterAdapter.cs`, `ExportAdapter.cs`, `RenderEngine.cs`, `Renderers.cs`). `RenderResult` has no `sample_rate` field — all audio is treated as 44100 Hz regardless.

The vocoder wraps RVC output in `FixedLenVocoder`, which resamples from the model's native sample rate (40k/48k) to 44100 via `F.interpolate(mode="linear")`. Both `dsconfig.yaml` and `vocoder.yaml` declare `sample_rate: 44100`, `hop_size: 882` (= 44100 / 50 fps).

The phonemizer's `dsdur/dsconfig.yaml` uses its own `sample_rate: 40000`, `hop_size: 800` — OpenUtau loads these separately.

### Template-Based Packing (Zero-Torch)

To eliminate the PyTorch dependency for end users:

1. **Developer** exports ONNX templates once with `do_constant_folding=False`, preserving all weight initializer names
2. **Packer** (numpy + onnx only) reads `.pth` via a custom unpickler (`pth_reader.py`), merges `weight_norm` pairs (`weight_g × weight_v / ||weight_v||`), and patches template initializers via a JSON weight map (`onnx_name → pth_name`)

This keeps the packer exe under 50 MB (vs ~2 GB with bundled PyTorch).

### Index Retrieval

RVC's FAISS index is replaced with brute-force matrix KNN baked into the ONNX graph:

- Pre-normalized `big_npy_norm` buffer avoids ONNX Expand ops with baked vector counts
- Top-8 cosine similarity → L2-equivalent inverse-square weighting: `w = 1 / (2(1-cos))²`
- Blended: `result = retrieved × index_rate + original × (1 - index_rate)`

### Attention ONNX Export

RVC's relative attention uses `.view()` with runtime-computed sizes that get baked as constants during ONNX tracing. `attn_patch.py` replaces `rel_to_abs` / `abs_to_rel` with `@torch.jit.script` versions that export as dynamic Shape/Gather/Sub chains, verified functional at T=50–300+.

## Requirements

| Component | Dependencies |
|-----------|-------------|
| **GUI Packer (exe)** | None — self-contained |
| **ONNX export scripts** | Python 3.10+, PyTorch 2.x, numpy, onnx, pyyaml |
| **Index support** | faiss-cpu (optional, for `.index` files) |
| **Building the exe** | pyinstaller |
| **OpenUtau runtime** | OpenUtau with DiffSinger renderer (ONNX Runtime bundled) |

## Supported Models

- **RVC v2** with NSF-HiFiGAN decoder
- Sample rates: **40 kHz**, **48 kHz**
- **f0 (pitch) models only** — non-f0 models are not supported
- Standard 768-dim speaker embedding

## Voice Bank Output

A packed voice bank contains:

```
VoiceName/
├── acoustic.onnx          # S2H acoustic model (shared)
├── dsconfig.yaml          # DiffSinger config (44100 Hz, hop 882)
├── phonemes.txt           # IPA token vocabulary
├── dsdict-ja.yaml         # Japanese phoneme dictionary
├── dsdict-zh.yaml         # Chinese phoneme dictionary
├── character.txt          # Voice metadata
├── character.yaml         # DiffSinger singer type + phonemizer
├── dsdur/                 # Phonemizer duration model
│   ├── dsconfig.yaml
│   ├── dur.onnx
│   └── linguistic.onnx
└── dsvocoder/
    ├── model.onnx         # RVC vocoder (per-voice)
    └── vocoder.yaml       # Vocoder config (44100 Hz, hop 882)
```

## Known Limitations

- OpenUtau phonemizer auto-selection sometimes requires manually selecting `DiffSingerJapanesePhonemizer` once for new singers
- S2H derives MIDI from F0 (no MIDI input in DiffSinger protocol) — may affect notes with strong vibrato
- Linear interpolation for sample rate conversion (acceptable quality, not ideal for audiophile use)
- RVC model quality varies by training data — articulation depends on the source model

## Credits

- [OpenUtau](https://github.com/stakira/OpenUtau) — host application with DiffSinger renderer
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — vocoder architecture
- [DiffSinger](https://github.com/openvpi/DiffSinger) — protocol specification

## License

Apache-2.0
