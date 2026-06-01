# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for S2H Voice Bank Packer
# Bundles: numpy + onnx + tkinter. Excludes: torch, scipy, CUDA.
#
# Compatible with both PyInstaller 5.x (flat layout) and 6.x+ (_internal
# subdirectory layout).  The COLLECT step places all binaries and data
# alongside the exe (or inside _internal, depending on the PyInstaller
# version), ensuring assets/ and phoneme_map/ are always found.

import os
import PyInstaller

SRC = os.path.abspath('src')
ROOT = os.path.abspath('.')

# Detect PyInstaller version for layout compatibility
_pyinst_major = int(PyInstaller.__version__.split('.')[0])

a = Analysis(
    [os.path.join(SRC, 'packer_gui.py')],
    pathex=[SRC],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('phoneme_map', 'phoneme_map'),
    ],
    hiddenimports=[
        'pth_reader',
        'onnx_patcher',
        'pack_voicebank',
        'numpy',
        'onnx',
        'onnx.numpy_helper',
        'google.protobuf',
        'faiss',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'scipy', 'matplotlib', 'pandas',
        'PIL', 'cv2', 'sklearn',
        'IPython', 'jupyter',
        'tensorboard', 'wandb',
        'librosa',
        'pytest', 'sphinx',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RVC_VoiceBank_Packer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(ROOT, 'assets', 'feather.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    a.zipfiles,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RVC_VoiceBank_Packer',
)
