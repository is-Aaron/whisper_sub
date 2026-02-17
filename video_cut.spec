# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Video Cut GUI application.

Usage:
    macOS:   pyinstaller video_cut.spec
    Windows: pyinstaller video_cut.spec

Output:
    macOS:   dist/Video Cut.app
    Windows: dist/Video Cut/Video Cut.exe
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
        name="Video Cut",
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
        name="Video Cut",
    )
    app = BUNDLE(
        coll,
        name="Video Cut.app",
        icon="icon.icns",
        bundle_identifier="com.videocut.app",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleDisplayName": "Video Cut",
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
        name="Video Cut",
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
        name="Video Cut",
    )
