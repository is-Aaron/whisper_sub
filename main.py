"""
WhisperSub - 从视频生成 SRT 字幕文件

使用 faster-whisper 进行语音识别，支持多语言，自动生成带时间戳的 SRT 字幕。

用法:
    uv run main.py <视频文件路径>
    uv run main.py <视频文件路径> --model small --language zh
    uv run main.py <视频文件路径> --output my_subtitle.srt
"""

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式: HH:MM:SS,mmm"""
    total_ms = round(max(0.0, seconds) * 1000)
    millis = total_ms % 1000
    total_secs = total_ms // 1000
    secs = total_secs % 60
    minutes = (total_secs // 60) % 60
    hours = total_secs // 3600
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_duration(seconds: float) -> str:
    """将秒数转换为可读的时长格式: MM:SS 或 HH:MM:SS"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(processed: float, total: float, elapsed: float, count: int) -> None:
    """打印进度条"""
    if total <= 0:
        return
    pct = min(processed / total, 1.0)
    bar_width = 30
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    eta_str = "--:--"
    if pct > 0 and elapsed > 0:
        eta = elapsed / pct * (1 - pct)
        eta_str = format_duration(eta)

    line = (
        f"\r进度: [{bar}] {pct:>6.1%} | "
        f"音频 {format_duration(processed)}/{format_duration(total)} | "
        f"已用 {format_duration(elapsed)} | "
        f"剩余 ~{eta_str} | "
        f"{count} 条字幕"
    )
    print(line, end="", flush=True)


def collect_segments_with_progress(segments, total_duration: float) -> list:
    """遍历 segments 生成器，边收集边显示进度"""
    segment_list = []
    start_time = time.monotonic()

    for segment in segments:
        if segment.text.strip():
            segment_list.append(segment)
        elapsed = time.monotonic() - start_time
        print_progress(segment.end, total_duration, elapsed, len(segment_list))

    elapsed = time.monotonic() - start_time
    print_progress(total_duration, total_duration, elapsed, len(segment_list))
    print()
    return segment_list


def generate_srt(segments: list) -> str:
    """将 faster-whisper 的识别片段转换为 SRT 格式字符串"""
    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    return "\n".join(srt_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从视频/音频文件生成 SRT 字幕",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run main.py video.mp4
  uv run main.py video.mp4 --model medium --language zh
  uv run main.py audio.mp3 --output subtitle.srt

模型选择 (越大越准，越慢):
  tiny    (~39MB)   - 快速预览
  base    (~74MB)   - 日常使用
  small   (~244MB)  - 推荐平衡 (默认)
  medium  (~769MB)  - 高质量
  large-v3 (~1.5GB) - 最高精度
        """,
    )
    parser.add_argument("input", help="视频或音频文件路径")
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper 模型大小 (默认: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="音频语言代码，如 zh/en/ja (默认: 自动检测)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 SRT 文件路径 (默认: 与输入同名.srt)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="运行设备: auto/cpu/cuda (默认: auto)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search 大小，越大越准但越慢 (默认: 5)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="禁用 VAD 静音过滤 (默认: 启用 VAD)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批量推理大小，越大越快但越耗内存 (默认: 8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件 (默认: 提示确认)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 - {input_path}", file=sys.stderr)
        sys.exit(1)
    if not input_path.is_file():
        print(f"错误: 路径不是文件 - {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".srt")

    if not output_path.parent.exists():
        print(f"错误: 输出目录不存在 - {output_path.parent}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and not args.overwrite:
        answer = input(f"输出文件已存在: {output_path}\n是否覆盖? (y/N): ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消")
            sys.exit(0)

    use_vad = not args.no_vad

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"量化: int8 (加速模式)")
    print(f"批量大小: {args.batch_size}")
    print(f"VAD 过滤: {'开启' if use_vad else '关闭'}")
    print()

    try:
        print("正在加载模型 (首次运行需下载，请耐心等待)...")
        model = WhisperModel(args.model, device=args.device, compute_type="int8")
        batched_model = BatchedInferencePipeline(model=model)
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}", file=sys.stderr)
        print("提示: 请检查网络连接，或尝试更小的模型 (--model tiny)", file=sys.stderr)
        sys.exit(1)

    print()

    try:
        print("正在识别语音...")
        segments, info = batched_model.transcribe(
            str(input_path),
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            language=args.language,
            vad_filter=use_vad,
        )

        print(f"检测到语言: {info.language} (置信度: {info.language_probability:.2%})")
        print(f"音频时长: {format_duration(info.duration)}")
        print()

        segment_list = collect_segments_with_progress(segments, info.duration)
    except Exception as e:
        print(f"\n识别失败: {e}", file=sys.stderr)
        print("提示: 请确认输入文件是有效的视频/音频格式", file=sys.stderr)
        sys.exit(1)

    if not segment_list:
        print("警告: 未识别到任何语音内容", file=sys.stderr)
        sys.exit(0)

    print()
    srt_content = generate_srt(segment_list)

    output_path.write_text(srt_content, encoding="utf-8")

    total_time = segment_list[-1].end - segment_list[0].start
    print(f"字幕生成完成! 共 {len(segment_list)} 条字幕，覆盖 {format_duration(total_time)} 语音内容")
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()
