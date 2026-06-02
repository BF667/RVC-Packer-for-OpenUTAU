"""Read PyTorch .pth state dicts without any torch dependency.

PyTorch .pth files are ZIP archives:
  archive/data.pkl   — pickle describing the state dict structure
  archive/data/0..N  — raw tensor bytes

This module provides stub classes for torch types and a custom unpickler
that reconstructs tensors as numpy arrays.

Only dependency: numpy.
"""

import io
import pickle
import struct
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ── Dtype mapping: torch storage type → numpy dtype ──────────────────────

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.float32,  # approximate — no native bf16 in numpy
    "IntStorage": np.int32,
    "LongStorage": np.int64,
    "ShortStorage": np.int16,
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "BoolStorage": np.bool_,
}


# ── Stub classes that replace torch internals during unpickling ──────────

class _StorageType:
    """Marker returned by find_class for torch.*Storage types."""
    __slots__ = ("dtype",)
    def __init__(self, dtype):
        self.dtype = dtype


class _NumpyStorage:
    """Stands in for torch.*Storage during unpickling."""
    __slots__ = ("dtype", "key", "location", "numel", "data")

    def __init__(self, dtype, key, location, numel):
        self.dtype = dtype
        self.key = key
        self.location = location
        self.numel = numel
        self.data = None  # filled by persistent_load

    def __repr__(self):
        return f"Storage(key={self.key}, dtype={self.dtype}, n={self.numel})"


def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                       requires_grad=False, backward_hooks=None):
    """Replaces torch._utils._rebuild_tensor_v2 — builds a numpy array."""
    if storage.data is None:
        return np.zeros(size, dtype=storage.dtype)
    flat = storage.data
    if storage_offset > 0:
        flat = flat[storage_offset:]
    # check if contiguous (C-order)
    if _is_contiguous(size, stride):
        total = 1
        for s in size:
            total *= s
        return flat[:total].reshape(size).copy()
    else:
        # non-contiguous: use as_strided equivalent
        return _strided_view(flat, size, stride, storage.dtype)


def _is_contiguous(size, stride):
    """Check if the tensor layout is C-contiguous."""
    if len(size) == 0:
        return True
    expected = 1
    for i in range(len(size) - 1, -1, -1):
        if size[i] != 1 and stride[i] != expected:
            return False
        expected *= size[i]
    return True


def _strided_view(data, size, stride, dtype):
    """Reconstruct a non-contiguous tensor as a contiguous numpy array."""
    result = np.zeros(size, dtype=dtype)
    if len(size) == 0:
        return result
    # iterate and gather (slow but correct for edge cases)
    it = np.nditer(result, flags=["multi_index"], op_flags=["writeonly"])
    while not it.finished:
        idx = it.multi_index
        flat_idx = sum(i * s for i, s in zip(idx, stride))
        if flat_idx < len(data):
            it[0] = data[flat_idx]
        it.iternext()
    return result


# ── Custom unpickler ─────────────────────────────────────────────────────

class _TorchUnpickler(pickle.Unpickler):
    """Unpickler that maps torch types to numpy equivalents."""

    def __init__(self, fp, reader):
        super().__init__(fp)
        self._reader = reader

    def find_class(self, module, name):
        # torch._utils._rebuild_tensor_v2
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2

        # torch.FloatStorage, torch.HalfStorage, etc.
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(_STORAGE_DTYPES[name])

        # torch.nn.modules.* — return as-is for ordered dict wrapping
        if module.startswith("torch.nn"):
            return lambda *args, **kw: OrderedDict()

        # collections.OrderedDict
        if module == "collections" and name == "OrderedDict":
            return OrderedDict

        # torch._utils._rebuild_parameter / _rebuild_parameter_with_state
        if "_rebuild_parameter" in name:
            return lambda data, *args, **kw: data

        # fallback
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return lambda *args, **kw: None

    def persistent_load(self, saved_id):
        """Load tensor storage from ZIP data files."""
        if not isinstance(saved_id, tuple) or len(saved_id) < 5:
            return saved_id

        tag = saved_id[0]
        if tag != "storage":
            return saved_id

        _, storage_type_info, key, location, numel = saved_id[:5]

        # determine dtype from storage class marker or infer from raw bytes
        if isinstance(storage_type_info, _StorageType):
            dtype = storage_type_info.dtype
        elif isinstance(storage_type_info, str):
            dtype = _STORAGE_DTYPES.get(storage_type_info, np.float32)
        else:
            dtype = None  # will infer from raw size

        raw = self._reader.read_data(str(key))
        if raw is not None and dtype is None:
            # infer dtype from raw byte size vs numel
            bpe = len(raw) / max(numel, 1)
            if bpe <= 1:
                dtype = np.uint8
            elif bpe <= 2:
                dtype = np.float16
            elif bpe <= 4:
                dtype = np.float32
            else:
                dtype = np.float64
        dtype = dtype or np.float32

        storage = _NumpyStorage(dtype, str(key), location, numel)
        if raw is not None:
            storage.data = np.frombuffer(raw, dtype=dtype, count=numel)
        return storage


# ── Public API ───────────────────────────────────────────────────────────

class PthReader:
    """Read a PyTorch .pth file and return its state dict as {str: np.ndarray}.

    Usage:
        reader = PthReader("model.pth")
        state_dict = reader.load()
        # state_dict["enc_p.emb_phone.weight"] -> np.ndarray
    """

    def __init__(self, path):
        self.path = Path(path)
        self._zf = None
        self._prefix = ""

    def load(self) -> dict:
        """Load and return the state dict (or full checkpoint dict)."""
        if not self.path.exists():
            raise FileNotFoundError(f"Not found: {self.path}")

        # check if it's a ZIP (modern torch.save format)
        if zipfile.is_zipfile(str(self.path)):
            return self._load_zip()
        else:
            raise ValueError(
                f"Unsupported format: {self.path.name}. "
                "Expected a PyTorch ZIP-based .pth file.")

    def _load_zip(self) -> dict:
        self._zf = zipfile.ZipFile(str(self.path), "r")
        try:
            # find data.pkl
            pkl_name = None
            for name in self._zf.namelist():
                if name.endswith("data.pkl"):
                    pkl_name = name
                    # prefix is everything before "data.pkl"
                    self._prefix = name[: -len("data.pkl")]
                    break

            if pkl_name is None:
                raise ValueError("No data.pkl found in archive")

            pkl_bytes = self._zf.read(pkl_name)
            unpickler = _TorchUnpickler(io.BytesIO(pkl_bytes), self)
            result = unpickler.load()
            return result
        finally:
            self._zf.close()
            self._zf = None

    def read_data(self, key: str) -> bytes | None:
        """Read raw bytes for a storage key from the ZIP archive."""
        if self._zf is None:
            return None
        data_path = f"{self._prefix}data/{key}"
        try:
            return self._zf.read(data_path)
        except KeyError:
            return None


def load_pth(path: str) -> dict:
    """Convenience function: load .pth → state dict {str: np.ndarray}.

    Handles both full checkpoints (returns 'weight' or 'state_dict' sub-key)
    and plain state dicts.
    """
    reader = PthReader(path)
    data = reader.load()

    if isinstance(data, dict):
        # RVC packaged voices use 'weight'; training checkpoints use 'model'
        for key in ("weight", "model", "state_dict", "model_state_dict"):
            if key in data and isinstance(data[key], dict):
                return data[key], data
        return data, data

    return {}, {}
