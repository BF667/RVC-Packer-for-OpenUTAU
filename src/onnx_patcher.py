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
