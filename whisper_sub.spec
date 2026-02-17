# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for WhisperSub GUI application.

Usage:
    macOS:   pyinstaller whisper_sub.spec
    Windows: pyinstaller whisper_sub.spec

Output:
    macOS:   dist/WhisperSub.app
    Windows: dist/WhisperSub/WhisperSub.exe
"""

import sys
from PyInstaller.utils.hooks import collect_all

# ---------------------------------------------------------------------------
# Collect native libraries and data for key dependencies
# ---------------------------------------------------------------------------

ctranslate2_datas, ctranslate2_binaries, ctranslate2_hiddenimports = collect_all(
    "ctranslate2"
)
faster_whisper_datas, faster_whisper_binaries, faster_whisper_hiddenimports = (
    collect_all("faster_whisper")
)

datas = [
    ("icon.png", "."),
    ("icon.ico", "."),
]
datas += ctranslate2_datas
datas += faster_whisper_datas

binaries = []
binaries += ctranslate2_binaries
binaries += faster_whisper_binaries

hiddenimports = [
    "faster_whisper",
    "ctranslate2",
    "huggingface_hub",
    "tokenizers",
]
hiddenimports += ctranslate2_hiddenimports
hiddenimports += faster_whisper_hiddenimports

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

# ---------------------------------------------------------------------------
# Platform-specific packaging
# ---------------------------------------------------------------------------

if sys.platform == "darwin":
    # macOS: create a .app bundle (folder mode for fast startup)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="WhisperSub",
        debug=False,
        strip=False,
        upx=True,
        console=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        name="WhisperSub",
    )
    app = BUNDLE(
        coll,
        name="WhisperSub.app",
        icon="icon.icns",
        bundle_identifier="com.whispersub.app",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleDisplayName": "WhisperSub",
            "NSHighResolutionCapable": True,
        },
    )
else:
    # Windows: create a folder with exe (onedir mode for fast startup)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="WhisperSub",
        debug=False,
        strip=False,
        upx=True,
        console=False,
        icon="icon.ico",
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        name="WhisperSub",
    )
