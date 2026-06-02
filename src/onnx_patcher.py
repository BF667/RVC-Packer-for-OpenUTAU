"""Patch weights from a PyTorch state dict into a pre-made ONNX template.

The ONNX template contains the correct computation graph (exported once by
the developer using torch). This module replaces the placeholder weights
with actual model weights, applying weight_norm merging as needed.

Only dependencies: onnx, numpy.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    onnx = None
    numpy_helper = None


# ── Weight-norm merging (replaces torch.nn.utils.remove_weight_norm) ─────

def merge_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray,
                      dim: int = 0) -> np.ndarray:
    """Merge weight_norm params: weight = weight_v * (weight_g / ||weight_v||).

    Same math as torch.nn.utils.weight_norm with dim=0.
    """
    # compute norm over all dims except `dim`
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.sqrt(np.sum(weight_v.astype(np.float32) ** 2, axis=axes,
                          keepdims=True))
    norm = np.maximum(norm, 1e-12)
    weight = weight_v.astype(np.float32) * (
        weight_g.astype(np.float32) / norm)
    return weight


def process_state_dict(state_dict: dict) -> dict:
    """Process a raw .pth state dict:
    1. Merge all weight_norm pairs (weight_g + weight_v → weight)
    2. Convert fp16 → fp32
    3. Strip enc_q (posterior encoder, not used in inference)

    Returns: {name: np.ndarray(fp32)} ready for ONNX patching.
    """
    processed = {}

    # collect weight_norm pairs
    wn_bases = set()
    for key in state_dict:
        if key.endswith(".weight_g"):
            wn_bases.add(key[:-len(".weight_g")])
        elif key.endswith(".weight_v"):
            wn_bases.add(key[:-len(".weight_v")])

    for key, val in state_dict.items():
        if not isinstance(val, np.ndarray):
            continue

        # skip posterior encoder (not in inference ONNX)
        if key.startswith("enc_q."):
            continue

        base = None
        if key.endswith(".weight_g"):
            base = key[:-len(".weight_g")]
        elif key.endswith(".weight_v"):
            base = key[:-len(".weight_v")]

        if base is not None and base in wn_bases:
            # only process once per pair (when we hit weight_v)
            if key.endswith(".weight_v"):
                g_key = base + ".weight_g"
                v_key = base + ".weight_v"
                if g_key in state_dict and v_key in state_dict:
                    merged = merge_weight_norm(
                        state_dict[g_key], state_dict[v_key])
                    processed[base + ".weight"] = merged
            continue  # skip individual _g/_v keys

        # regular parameter: convert to fp32
        processed[key] = val.astype(np.float32)

    return processed


# ── ONNX template patching ───────────────────────────────────────────────

def patch_onnx_template(template_path: str, state_dict: dict,
                        output_path: str,
                        weight_map_path: Optional[str] = None,
                        extra_initializers: Optional[dict] = None):
    """Replace weights in an ONNX template with values from state_dict.

    Args:
        template_path: path to ONNX template file
        state_dict: {param_name: np.ndarray} processed state dict
        output_path: where to save the patched ONNX model
        weight_map_path: optional JSON mapping {onnx_name: pth_name}
                         (if None, assumes names match directly)
        extra_initializers: optional {name: np.ndarray} to add as new
                            constant tensors (e.g., baked index vectors)
    """
    if onnx is None:
        raise ImportError("onnx package required. Install: pip install onnx")

    model = onnx.load(template_path)

    # build mapping
    if weight_map_path and Path(weight_map_path).exists():
        with open(weight_map_path, "r") as f:
            weight_map = json.load(f)
    else:
        weight_map = None

    # patch existing initializers
    patched = 0
    skipped = []
    for init in model.graph.initializer:
        # resolve the state dict key
        if weight_map:
            pth_key = weight_map.get(init.name, init.name)
        else:
            pth_key = init.name

        if pth_key in state_dict:
            arr = state_dict[pth_key]
            orig_shape = list(init.dims)
            if list(arr.shape) != orig_shape:
                if (arr.ndim == 2 and len(orig_shape) == 2
                        and list(arr.shape) == orig_shape[::-1]):
                    arr = arr.T
                elif arr.size == np.prod(orig_shape):
                    arr = arr.reshape(orig_shape)
                else:
                    skipped.append((init.name, list(arr.shape), orig_shape))
                    continue
            # replace data
            init.CopyFrom(numpy_helper.from_array(arr.astype(np.float32),
                                                   name=init.name))
            patched += 1

    # add extra initializers (e.g., baked index)
    if extra_initializers:
        for name, arr in extra_initializers.items():
            tensor = numpy_helper.from_array(arr.astype(np.float32), name=name)
            model.graph.initializer.append(tensor)

    onnx.save(model, output_path)

    print(f"  Patched {patched}/{len(model.graph.initializer)} initializers")
    if skipped:
        print(f"  Skipped {len(skipped)} (shape mismatch):")
        for name, got, exp in skipped[:5]:
            print(f"    {name}: got {got}, expected {exp}")

    return patched


def fix_reduce_l2_nodes(model_path: str, output_path: Optional[str] = None):
    """Replace ReduceL2 ops with Pow+ReduceSum+Sqrt for runtime compatibility.

    ReduceL2 with 'axes' attribute is rejected by Microsoft.ML.OnnxRuntime
    (used by OpenUTAU) because at opset 13+ axes should be an input, not an
    attribute.  PyTorch's ONNX exporter sometimes generates ReduceL2 with
    axes as attribute even at opset 17.  Replacing with equivalent
    Pow+ReduceSum+Sqrt avoids this issue entirely.
    """
    if onnx is None:
        raise ImportError("onnx package required")
    from onnx import helper

    if output_path is None:
        output_path = model_path

    model = onnx.load(model_path)
    graph = model.graph

    nodes_to_remove = []
    new_nodes = []
    new_initializers = []
    fix_count = 0

    for node in graph.node:
        if node.op_type != "ReduceL2":
            continue

        # Extract axes from attribute (old-style) or input (opset 13+)
        axes_val = None
        keepdims = 1
        for attr in node.attribute:
            if attr.name == "axes":
                axes_val = list(attr.ints)
            elif attr.name == "keepdims":
                keepdims = attr.i

        input_name = node.input[0]
        output_name = node.output[0]

        # Unique suffix for intermediate tensors
        suffix = f"_rl2fix_{fix_count}"
        pow_out = f"_rl2_pow{suffix}"
        sum_out = f"_rl2_sum{suffix}"
        sqrt_out = f"_rl2_sqrt{suffix}"
        const_2_name = f"_rl2_two{suffix}"

        # Constant 2.0 for Pow
        new_initializers.append(
            numpy_helper.from_array(
                np.array(2.0, dtype=np.float32), name=const_2_name))

        # Axes as input for ReduceSum
        if axes_val is not None:
            axes_name = f"_rl2_axes{suffix}"
            new_initializers.append(
                numpy_helper.from_array(
                    np.array(axes_val, dtype=np.int64), name=axes_name))
        elif len(node.input) > 1 and node.input[1]:
            axes_name = node.input[1]
        else:
            axes_name = None

        # Pow(input, 2)
        new_nodes.append(helper.make_node(
            "Pow", [input_name, const_2_name], [pow_out]))

        # ReduceSum(pow_out, axes, keepdims)
        if axes_name is not None:
            new_nodes.append(helper.make_node(
                "ReduceSum", [pow_out, axes_name], [sum_out],
                keepdims=keepdims))
        else:
            new_nodes.append(helper.make_node(
                "ReduceSum", [pow_out], [sum_out],
                keepdims=keepdims))

        # Sqrt(sum_out)
        new_nodes.append(helper.make_node(
            "Sqrt", [sum_out], [sqrt_out]))

        # Redirect all consumers of ReduceL2 output to Sqrt output
        for other in graph.node:
            for i, inp in enumerate(other.input):
                if inp == output_name:
                    other.input[i] = sqrt_out

        # Fix graph outputs too
        for out in graph.output:
            if out.name == output_name:
                out.name = sqrt_out

        nodes_to_remove.append(node)
        fix_count += 1

    if fix_count == 0:
        return  # Nothing to fix, don't re-save

    # Remove old ReduceL2 nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    # Add new initializers and nodes
    for init in new_initializers:
        graph.initializer.append(init)
    for node in new_nodes:
        graph.node.append(node)

    onnx.save(model, output_path)
    print(f"  Fixed {fix_count} ReduceL2 node(s) → Pow+ReduceSum+Sqrt")


def zero_nsf_weights(model_path: str, output_path: Optional[str] = None):
    """Zero out NSF source module weights for f0=0 / HiFi-GAN-only models.

    When a pure HiFi-GAN model (f0=0) is packed using a standard template
    (which includes NSF/GeneratorNSF source nodes), the m_source and
    noise_convs weights won't have real values from the checkpoint.
    Zeroing them ensures the NSF path produces zero output, effectively
    making the decoder behave as a plain HiFi-GAN Generator.
    """
    if onnx is None:
        raise ImportError("onnx package required")

    if output_path is None:
        output_path = model_path

    model = onnx.load(model_path)
    zeroed = 0
    # Match both bare (dec.m_source.*) and wrapped (voc.net_g.dec.m_source.*) names
    nsf_patterns = ("dec.m_source.", "dec.noise_convs.", "dec.noise_res.")

    for init in model.graph.initializer:
        if any(p in init.name for p in nsf_patterns):
            arr = numpy_helper.to_array(init)
            zeros = np.zeros_like(arr)
            init.CopyFrom(numpy_helper.from_array(zeros, name=init.name))
            zeroed += 1

    onnx.save(model, output_path)
    if zeroed:
        print(f"  Zeroed {zeroed} NSF weight(s) for f0=0 / HiFi-GAN model")
    else:
        print(f"  No NSF weights found (model may already be f0=0 template)")


def bake_index_into_onnx(model_path: str, npy_path: str, index_rate: float,
                         output_path: str):
    """Replace index vectors in an ONNX model (exported with index support).

    Replaces both 'voc.big_npy' (raw) and 'voc.big_npy_norm' (L2-normalized).
    Pre-normalizing avoids ONNX Expand ops with baked shapes that break when
    the index size changes.
    """
    if onnx is None:
        raise ImportError("onnx package required")

    if npy_path.endswith(".index"):
        import faiss
        idx = faiss.read_index(npy_path)
        big_npy = idx.reconstruct_n(0, idx.ntotal).astype(np.float32)
    else:
        big_npy = np.load(npy_path, allow_pickle=False).astype(np.float32)
    norms = np.linalg.norm(big_npy, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    big_npy_norm = big_npy / norms
    print(f"  Index: {big_npy.shape[0]} vectors, dim={big_npy.shape[1]}")

    model = onnx.load(model_path)

    replacements = {"voc.big_npy": big_npy, "voc.big_npy_norm": big_npy_norm}
    for name, arr in replacements.items():
        for i, init in enumerate(model.graph.initializer):
            if init.name == name:
                model.graph.initializer[i].CopyFrom(
                    numpy_helper.from_array(arr, name=name))
                break

    onnx.save(model, output_path)
    print(f"  Baked index ({big_npy.shape[0]} vectors, rate={index_rate})")


def patch_f0_silence_mask(model_path: str, output_path: Optional[str] = None,
                          has_f0: bool = True):
    """Insert SP silence handling into vocoder ONNX graph.

    Two layers of protection for SP (silence padding) frames:
    1. Zero F0 — prevents NSF harmonic generation (only for f0=1 / GeneratorNSF models)
    2. Zero waveform output — catches residual noise from index retrieval
       leaking non-zero HuBERT into enc_p even when F0 is zero

    For f0=0 (pure HiFi-GAN / Generator) models, layer 1 is skipped since
    there is no F0 input to mask.  Layer 2 (waveform masking) is always
    applied because the acoustic model can still produce silent SP frames.

    Uses Resize with `scales` (not `sizes`) for opset 11-17 compatibility.
    The voiced mask is upsampled from frame rate to audio sample rate via
    a computed scale factor, ensuring correct temporal alignment even for
    fast syllable transitions.
    """
    if onnx is None:
        raise ImportError("onnx package required")
    from onnx import helper, TensorProto

    if output_path is None:
        output_path = model_path

    model = onnx.load(model_path)
    graph = model.graph

    if any(n.output[0] == "_sp_f0_masked" for n in graph.node):
        print("  SP silence mask already present, skipping")
        return

    # Silence threshold — raised slightly from 1e-6 to avoid incorrectly
    # zeroing out low-energy consonant frames that are still voiced.
    _SP_EPS = 1e-4

    for name, val in [
        ("_sp_pow2",  np.array(2.0, dtype=np.float32)),
        ("_sp_eps",   np.array(_SP_EPS, dtype=np.float32)),
        ("_sp_one",   np.array(1.0, dtype=np.float32)),
        ("_sp_axes",  np.array([-1], dtype=np.int64)),
        ("_sp_ax1",   np.array([1], dtype=np.int64)),
        ("_sp_idx1",  np.array(1, dtype=np.int64)),
        ("_sp_sh0",   np.array([0], dtype=np.int64)),
        ("_sp_f1",    np.array(1.0, dtype=np.float32)),
        ("_sp_roi",   np.array([], dtype=np.float32)),
    ]:
        graph.initializer.append(numpy_helper.from_array(val, name=name))

    # Compute voiced mask from mel energy (used by both layers)
    f0_nodes = [
        helper.make_node("Pow", ["mel", "_sp_pow2"], ["_sp_mel_sq"]),
        helper.make_node("ReduceSum", ["_sp_mel_sq", "_sp_axes"],
                         ["_sp_mel_energy"], keepdims=0),
        helper.make_node("Less", ["_sp_mel_energy", "_sp_eps"],
                         ["_sp_is_silent"]),
        helper.make_node("Cast", ["_sp_is_silent"], ["_sp_silent_f"],
                         to=TensorProto.FLOAT),
        helper.make_node("Sub", ["_sp_one", "_sp_silent_f"],
                         ["_sp_voiced"]),
    ]

    # -- Layer 1: F0 masking (only for f0=1 / GeneratorNSF models) --
    if has_f0:
        f0_nodes.append(
            helper.make_node("Mul", ["f0", "_sp_voiced"], ["_sp_f0_masked"]))

        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == "f0":
                    node.input[i] = "_sp_f0_masked"
    else:
        print("  F0 masking skipped (f0=0 / pure HiFi-GAN model)")

    for i, node in enumerate(f0_nodes):
        graph.node.insert(i, node)

    # -- Layer 2: waveform output masking --
    # Rename current "waveform" producer → "_sp_wav_raw"
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == "waveform":
                node.output[i] = "_sp_wav_raw"

    # Upsample _sp_voiced [1,T] → [1,1,T] → [1,1,T_audio] via Resize with
    # *scales* (opset 11-17 compatible).  scale = T_audio / T_frames.
    audio_nodes = [
        # _sp_voiced [1,T] → [1,1,T]  (add channel dim for Resize)
        helper.make_node("Unsqueeze", ["_sp_voiced", "_sp_ax1"],
                         ["_sp_v3d"]),
        # Compute scale factor = T_audio / T_frames  (both as float)
        helper.make_node("Shape", ["_sp_wav_raw"], ["_sp_ws"]),
        helper.make_node("Gather", ["_sp_ws", "_sp_idx1"], ["_sp_ta_i"]),
        helper.make_node("Cast", ["_sp_ta_i"], ["_sp_ta_f"],
                         to=TensorProto.FLOAT),
        helper.make_node("Shape", ["_sp_v3d"], ["_sp_vs"]),
        helper.make_node("Gather", ["_sp_vs", "_sp_idx1"], ["_sp_tf_i"]),
        helper.make_node("Cast", ["_sp_tf_i"], ["_sp_tf_f"],
                         to=TensorProto.FLOAT),
        # scales = [1.0, 1.0, T_audio / T_frames]
        helper.make_node("Div", ["_sp_ta_f", "_sp_tf_f"], ["_sp_scl_f"]),
        # Build 3-element scales tensor: [1.0, 1.0, scale]
        # Use Concat to assemble from constants and computed scale
        helper.make_node("Concat", ["_sp_f1", "_sp_f1", "_sp_scl_f"],
                         ["_sp_scales"], axis=0),
        # Resize with scales (3 inputs: X, roi, scales — opset 11-17)
        helper.make_node("Resize",
                         ["_sp_v3d", "_sp_roi", "_sp_scales"],
                         ["_sp_va3"], mode="nearest"),
        # [1,1,T_audio] → [1,T_audio]
        helper.make_node("Squeeze", ["_sp_va3", "_sp_ax1"], ["_sp_va"]),
        helper.make_node("Mul", ["_sp_wav_raw", "_sp_va"], ["waveform"]),
    ]

    graph.node.extend(audio_nodes)

    onnx.save(model, output_path)
    layers = "F0 + waveform" if has_f0 else "waveform only (no F0)"
    print(f"  SP silence mask injected ({layers})")
