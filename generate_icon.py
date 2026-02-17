"""
生成 Video Cut 应用图标。

设计: 蓝色圆角方形背景 + 白色播放三角 + 两行字幕线条,
传达 "视频 → 字幕" 的核心功能。

用法:
    uv run --with pillow generate_icon.py

生成:
    icon.png  — 512x512 PNG   (macOS / Linux / 通用)
    icon.ico  — 多尺寸 ICO    (Windows 任务栏 & 标题栏)
    icon.icns — 多尺寸 ICNS   (macOS .app 应用图标, 仅 macOS 可生成)
"""

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

    # ── 背景: 蓝色圆角方形 ──
    pad = round(s * 0.04)
    radius = round(s * 0.18)
    draw.rounded_rectangle(
        [(pad, pad), (s - pad, s - pad)],
        radius=radius,
        fill="#1565C0",
    )

    # ── 播放按钮: 白色三角形 (▶) ──
    # 质心居中，视觉上偏右一点以平衡三角形的不对称感
    cx = s * 0.52
    cy = s * 0.37
    w = s * 0.28
    h = s * 0.18
    draw.polygon(
        [
            (cx - w / 3, cy - h),
            (cx - w / 3, cy + h),
            (cx + w * 2 / 3, cy),
        ],
        fill="white",
    )

    # ── 字幕线条 ──
    line_h = max(round(s * 0.04), 2)
    line_r = max(round(s * 0.02), 1)

    # 第一行: 宽，白色
    y1 = round(s * 0.64)
    draw.rounded_rectangle(
        [(round(s * 0.16), y1), (round(s * 0.84), y1 + line_h)],
        radius=line_r,
        fill="white",
    )

    # 第二行: 窄，浅蓝
    y2 = round(s * 0.73)
    draw.rounded_rectangle(
        [(round(s * 0.24), y2), (round(s * 0.76), y2 + line_h)],
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
