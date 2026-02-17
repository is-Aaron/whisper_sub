# WhisperSub — 视频字幕生成器

从视频/音频文件自动生成 SRT 字幕，基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 语音识别引擎。

## 功能特性

- 支持主流视频/音频格式 (mp4, mkv, avi, mov, mp3, wav, flac 等)
- 多种 Whisper 模型可选 (tiny → large-v3)
- 自动语言检测，也可手动指定
- VAD 静音过滤，提升识别准确率
- **CLI 模式**: 带进度条的命令行工具
- **GUI 模式**: 图形界面，支持批量处理和并发任务

## 安装

需要 Python 3.12+，推荐使用 [uv](https://github.com/astral-sh/uv) 管理项目：

```bash
uv sync
```

## 使用方式

### 命令行 (CLI)

```bash
# 基本用法
uv run main.py video.mp4

# 指定模型和语言
uv run main.py video.mp4 --model medium --language zh

# 指定输出文件
uv run main.py audio.mp3 --output subtitle.srt
```

#### CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | (必填) | 视频或音频文件路径 |
| `--model` | `small` | 模型大小: tiny / base / small / medium / large-v3 |
| `--language` | 自动检测 | 语言代码: zh / en / ja 等 |
| `--output` | 与输入同名.srt | 输出 SRT 文件路径 |
| `--device` | `auto` | 运行设备: auto / cpu / cuda |
| `--beam-size` | `5` | Beam search 大小 |
| `--batch-size` | `8` | 批量推理大小 |
| `--no-vad` | - | 禁用 VAD 静音过滤 |
| `--overwrite` | - | 覆盖已存在的输出文件 |

### 图形界面 (GUI)

```bash
uv run gui.py
```

支持批量添加文件/文件夹、选择模型、设置并发数，实时查看进度和日志。

## 模型选择

| 模型 | 大小 | 适用场景 |
|------|------|----------|
| tiny | ~39MB | 快速预览 |
| base | ~74MB | 日常使用 |
| **small** | **~244MB** | **推荐 (默认)** |
| medium | ~769MB | 高质量 |
| large-v3 | ~1.5GB | 最高精度 |

首次使用时模型会自动下载，也可在 GUI 中手动指定本地模型路径。

## 打包为桌面应用

可使用 PyInstaller 将 GUI 打包为独立的桌面应用程序，无需用户安装 Python 环境。

### 安装打包依赖

```bash
uv sync --extra build
```

### macOS 打包 (.app)

```bash
uv run pyinstaller whisper_sub.spec
```

打包完成后，应用位于 `dist/WhisperSub.app`，可直接双击运行或拖入「应用程序」文件夹。

### Windows 打包 (.exe)

```bash
uv run pyinstaller whisper_sub.spec
```

打包完成后，应用位于 `dist/WhisperSub/WhisperSub.exe`，可将整个 `WhisperSub` 文件夹分发给用户。

> **注意**: 打包必须在目标平台上执行 — macOS 上打包生成 `.app`，Windows 上打包生成 `.exe`，不能交叉编译。

## 生成图标

```bash
uv run --with pillow generate_icon.py
```
