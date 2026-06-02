# -*- mode: python ; coding: utf-8 -*-
# PyInstaller ONEFILE spec for S2H Voice Bank Packer
# Builds a single .exe (no _internal folder needed)
# Use this for Windows releases.

import os
import PyInstaller

SRC = os.path.abspath('src')
ROOT = os.path.abspath('.')

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
    a.binaries,
    a.datas,
    a.zipfiles,
    [],
    name='RVC_VoiceBank_Packer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=os.path.join(ROOT, 'assets', 'feather.ico'),
)
