"""
生成 WhisperSub 应用图标。

设计: 深蓝渐变圆角方形背景 + 声波弧线 + 字幕线条,
传达 "语音识别 (Whisper) → 字幕 (Sub)" 的核心功能。

用法:
    uv run --with pillow generate_icon.py

生成:
    icon.png  — 512x512 PNG   (macOS / Linux / 通用)
    icon.ico  — 多尺寸 ICO    (Windows 任务栏 & 标题栏)
    icon.icns — 多尺寸 ICNS   (macOS .app 应用图标, 仅 macOS 可生成)
"""

import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def create_icon(size: int = 512) -> Image.Image:
    """绘制指定尺寸的应用图标，所有坐标按比例缩放。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size

    # ── 背景: 深蓝色圆角方形 ──
    pad = round(s * 0.04)
    radius = round(s * 0.18)
    draw.rounded_rectangle(
        [(pad, pad), (s - pad, s - pad)],
        radius=radius,
        fill="#0D47A1",
    )

    # ── 声波弧线 (代表 Whisper 语音识别) ──
    cx = s * 0.38
    cy = s * 0.38
    arc_configs = [
        (s * 0.13, max(round(s * 0.045), 2), "#FFFFFF"),
        (s * 0.24, max(round(s * 0.040), 2), "rgba(255,255,255,220)"),
        (s * 0.35, max(round(s * 0.035), 2), "rgba(255,255,255,180)"),
    ]
    for arc_r, arc_w, color in arc_configs:
        bbox = [cx - arc_r, cy - arc_r, cx + arc_r, cy + arc_r]
        draw.arc(bbox, start=-50, end=50, fill=color, width=arc_w)

    # ── 中心圆点 (声源) ──
    dot_r = round(s * 0.045)
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill="white",
    )

    # ── 字幕线条 (代表 Sub 字幕输出) ──
    line_h = max(round(s * 0.042), 2)
    line_r = max(round(s * 0.021), 1)

    # 第一行: 宽，白色
    y1 = round(s * 0.62)
    draw.rounded_rectangle(
        [(round(s * 0.14), y1), (round(s * 0.86), y1 + line_h)],
        radius=line_r,
        fill="white",
    )

    # 第二行: 中等宽度，浅蓝
    y2 = round(s * 0.71)
    draw.rounded_rectangle(
        [(round(s * 0.20), y2), (round(s * 0.80), y2 + line_h)],
        radius=line_r,
        fill="#64B5F6",
    )

    # 第三行: 窄，更浅蓝 (渐隐效果)
    y3 = round(s * 0.80)
    draw.rounded_rectangle(
        [(round(s * 0.28), y3), (round(s * 0.72), y3 + line_h)],
        radius=line_r,
        fill="#90CAF9",
    )

    return img


def _generate_icns(output_dir: Path) -> None:
    """使用 macOS iconutil 生成 .icns 图标文件。"""
    iconset = output_dir / "icon.iconset"
    iconset.mkdir(exist_ok=True)

    entries = [
        (16, 1, 16),
        (16, 2, 32),
        (32, 1, 32),
        (32, 2, 64),
        (128, 1, 128),
        (128, 2, 256),
        (256, 1, 256),
        (256, 2, 512),
        (512, 1, 512),
        (512, 2, 1024),
    ]

    for base, scale, px in entries:
        icon = create_icon(px)
        suffix = f"@{scale}x" if scale > 1 else ""
        icon.save(iconset / f"icon_{base}x{base}{suffix}.png", "PNG")

    icns_path = output_dir / "icon.icns"
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)],
            check=True,
        )
        print(f"  icon.icns  (16~1024px, macOS)")
    except FileNotFoundError:
        print("  icon.icns  (跳过, 未找到 iconutil 命令)")
    except subprocess.CalledProcessError as exc:
        print(f"  icon.icns  (生成失败: {exc})")
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


def main() -> None:
    output_dir = Path(__file__).parent

    # PNG (512x512)
    icon = create_icon(512)
    png_path = output_dir / "icon.png"
    icon.save(png_path, "PNG")
    print(f"  icon.png   (512x512)")

    # ICO (多尺寸，供 Windows 使用)
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon_256 = create_icon(256)
    ico_path = output_dir / "icon.ico"
    icon_256.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"  icon.ico   ({', '.join(f'{w}x{h}' for w, h in ico_sizes)})")

    # ICNS (macOS .app 图标)
    if sys.platform == "darwin":
        _generate_icns(output_dir)
    else:
        print("  icon.icns  (跳过, 仅 macOS 可生成)")

    print("Done!")


if __name__ == "__main__":
    main()
