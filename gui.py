"""
Video Cut GUI — 视频字幕生成器图形界面

跨平台桌面应用 (macOS / Windows)，支持：
  - 批量添加视频/音频文件或整个文件夹
  - 选择模型大小 / 自定义模型路径
  - 设置并发任务数
  - 实时进度显示与日志

用法:
    uv run gui.py
"""

import sys
import threading
import tkinter as tk
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk

from faster_whisper import BatchedInferencePipeline, WhisperModel

from main import format_duration, generate_srt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
DEVICES = ["auto", "cpu", "cuda"]

LANGUAGES: list[tuple[str, str | None]] = [
    ("自动检测", None),
    ("中文", "zh"),
    ("英语", "en"),
    ("日语", "ja"),
    ("韩语", "ko"),
    ("法语", "fr"),
    ("德语", "de"),
    ("西班牙语", "es"),
    ("俄语", "ru"),
    ("葡萄牙语", "pt"),
    ("意大利语", "it"),
]

LANG_MAP: dict[str, str | None] = {name: code for name, code in LANGUAGES}

MEDIA_EXTS = frozenset(
    ".mp4 .mkv .avi .mov .wmv .flv .webm "
    ".mp3 .wav .flac .aac .ogg .m4a .wma".split()
)

FILE_TYPES = [
    ("视频/音频文件", " ".join(f"*{e}" for e in sorted(MEDIA_EXTS))),
    ("所有文件", "*.*"),
]

# Task status constants
ST_PENDING = "等待中"
ST_LOADING = "加载模型"
ST_RUNNING = "识别中"
ST_DONE = "已完成"
ST_FAIL = "失败"
ST_CANCEL = "已取消"

# ---------------------------------------------------------------------------
# Thread-local model cache
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_or_load_model(
    model_id: str, device: str
) -> BatchedInferencePipeline:
    """Return a BatchedInferencePipeline cached per-thread.

    Each thread in the ThreadPoolExecutor keeps its own model instance so that
    truly concurrent transcription is possible when concurrency > 1.
    When the model settings change, the old instance is replaced.
    """
    key = f"{model_id}|{device}"
    if getattr(_thread_local, "model_key", None) != key:
        model = WhisperModel(model_id, device=device, compute_type="int8")
        _thread_local.pipeline = BatchedInferencePipeline(model=model)
        _thread_local.model_key = key
    return _thread_local.pipeline


# ---------------------------------------------------------------------------
# Worker function (runs inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def _worker(
    task_id: str,
    input_path: str,
    output_path: str,
    model_id: str,
    device: str,
    language: str | None,
    beam_size: int,
    batch_size: int,
    use_vad: bool,
    msg_queue: Queue,
    cancel_event: threading.Event,
) -> None:
    """Transcribe a single file and write the SRT output."""

    def _send(kind: str, **kwargs: object) -> None:
        msg_queue.put({"task_id": task_id, "kind": kind, **kwargs})

    try:
        if cancel_event.is_set():
            _send("status", value=ST_CANCEL)
            return

        # --- Load model (may be cached in this thread) ---
        _send("status", value=ST_LOADING)
        pipeline = _get_or_load_model(model_id, device)

        if cancel_event.is_set():
            _send("status", value=ST_CANCEL)
            return

        # --- Transcribe ---
        _send("status", value=ST_RUNNING)
        segments_gen, info = pipeline.transcribe(
            input_path,
            batch_size=batch_size,
            beam_size=beam_size,
            language=language,
            vad_filter=use_vad,
        )

        _send(
            "meta",
            language=info.language,
            probability=info.language_probability,
            duration=info.duration,
        )

        collected: list = []
        for segment in segments_gen:
            if cancel_event.is_set():
                _send("status", value=ST_CANCEL)
                return
            if segment.text.strip():
                collected.append(segment)
            progress = (
                min(segment.end / info.duration, 1.0) if info.duration > 0 else 0.0
            )
            _send("progress", value=progress, count=len(collected))

        if not collected:
            _send("status", value=ST_DONE, detail="未识别到语音内容")
            return

        # --- Write SRT ---
        srt_content = generate_srt(collected)
        Path(output_path).write_text(srt_content, encoding="utf-8")

        total_speech = collected[-1].end - collected[0].start
        _send(
            "status",
            value=ST_DONE,
            detail=f"{len(collected)} 条字幕, 覆盖 {format_duration(total_speech)}",
        )

    except Exception as exc:
        tb = traceback.format_exc()
        _send("status", value=ST_FAIL, detail=str(exc), traceback=tb)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class App:
    """Main GUI application."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Video Cut — 视频字幕生成器")
        self.root.geometry("960x720")
        self.root.minsize(760, 520)

        # Internal state
        self.tasks: dict[str, dict] = {}
        self.executor: ThreadPoolExecutor | None = None
        self.cancel_event = threading.Event()
        self.msg_queue: Queue = Queue()
        self.running = False
        self._task_counter = 0

        # Tk variables
        self.var_model = tk.StringVar(value="small")
        self.var_model_path = tk.StringVar()
        self.var_device = tk.StringVar(value="auto")
        self.var_language = tk.StringVar(value="自动检测")
        self.var_beam_size = tk.IntVar(value=5)
        self.var_batch_size = tk.IntVar(value=8)
        self.var_concurrency = tk.IntVar(value=1)
        self.var_vad = tk.BooleanVar(value=True)

        self._apply_icon()
        self._build_ui()
        self._poll_messages()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Icon ──────────────────────────────────────────────────────────────

    def _apply_icon(self) -> None:
        """Set the window icon. Falls back silently if icon files are missing."""
        if getattr(sys, "frozen", False):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent
        try:
            if sys.platform == "win32":
                ico_path = base / "icon.ico"
                if ico_path.exists():
                    self.root.iconbitmap(str(ico_path))
                    return
            png_path = base / "icon.png"
            if png_path.exists():
                photo = tk.PhotoImage(file=str(png_path))
                self.root.iconphoto(True, photo)
                self._icon_photo = photo  # prevent garbage collection
        except tk.TclError:
            pass

    # ── UI Construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._build_settings(main_frame)
        self._build_buttons(main_frame)
        self._build_content(main_frame)

    def _build_settings(self, parent: ttk.Frame) -> None:
        settings = ttk.LabelFrame(parent, text=" 设置 ", padding=8)
        settings.pack(fill=tk.X, pady=(0, 6))

        # Row 1: Model, Device, Language, Concurrency
        row1 = ttk.Frame(settings)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="模型:").pack(side=tk.LEFT)
        ttk.Combobox(
            row1,
            textvariable=self.var_model,
            values=MODEL_SIZES,
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(row1, text="设备:").pack(side=tk.LEFT)
        ttk.Combobox(
            row1,
            textvariable=self.var_device,
            values=DEVICES,
            width=7,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(row1, text="语言:").pack(side=tk.LEFT)
        ttk.Combobox(
            row1,
            textvariable=self.var_language,
            values=[name for name, _ in LANGUAGES],
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(row1, text="并发任务数:").pack(side=tk.LEFT)
        ttk.Spinbox(
            row1, from_=1, to=16, textvariable=self.var_concurrency, width=4
        ).pack(side=tk.LEFT, padx=2)

        # Row 2: Model path
        row2 = ttk.Frame(settings)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="模型路径 (留空则自动下载):").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_model_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(
            row2, text="浏览", command=self._browse_model_path, width=6
        ).pack(side=tk.LEFT)

        # Row 3: Beam size, Batch size, VAD
        row3 = ttk.Frame(settings)
        row3.pack(fill=tk.X, pady=2)

        ttk.Label(row3, text="Beam Size:").pack(side=tk.LEFT)
        ttk.Spinbox(
            row3, from_=1, to=20, textvariable=self.var_beam_size, width=4
        ).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(row3, text="批量大小:").pack(side=tk.LEFT)
        ttk.Spinbox(
            row3, from_=1, to=32, textvariable=self.var_batch_size, width=4
        ).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Checkbutton(
            row3, text="启用 VAD 静音过滤", variable=self.var_vad
        ).pack(side=tk.LEFT, padx=(14, 0))

    def _build_buttons(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(bar, text="添加文件", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="添加文件夹", command=self._add_folder).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="移除选中", command=self._remove_selected).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="清空列表", command=self._clear_tasks).pack(
            side=tk.LEFT
        )

        self.btn_start = ttk.Button(
            bar, text="开始处理", command=self._start_processing
        )
        self.btn_start.pack(side=tk.RIGHT, padx=(4, 0))

        self.btn_stop = ttk.Button(
            bar, text="停止", command=self._stop_processing, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.RIGHT)

    def _build_content(self, parent: ttk.Frame) -> None:
        paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # --- Task list (Treeview) ---
        tree_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=3)

        columns = ("file", "output", "status", "progress")
        self.tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", selectmode="extended"
        )

        self.tree.heading("file", text="文件")
        self.tree.heading("output", text="输出")
        self.tree.heading("status", text="状态")
        self.tree.heading("progress", text="进度")

        self.tree.column("file", width=260, minwidth=100)
        self.tree.column("output", width=260, minwidth=100)
        self.tree.column("status", width=80, minwidth=60, anchor=tk.CENTER)
        self.tree.column("progress", width=220, minwidth=100, anchor=tk.CENTER)

        self.tree.tag_configure("done", foreground="#2e7d32")
        self.tree.tag_configure("fail", foreground="#c62828")
        self.tree.tag_configure("run", foreground="#1565c0")
        self.tree.tag_configure("cancel", foreground="#e65100")

        tree_scroll = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Log area ---
        log_frame = ttk.LabelFrame(paned, text=" 日志 ", padding=4)
        paned.add(log_frame, weight=1)

        self.log_text = tk.Text(
            log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED
        )
        log_scroll = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ── File / Task Management ────────────────────────────────────────────

    def _browse_model_path(self) -> None:
        directory = filedialog.askdirectory(title="选择 CTranslate2 模型目录")
        if directory:
            self.var_model_path.set(directory)

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择视频/音频文件", filetypes=FILE_TYPES
        )
        for p in paths:
            self._insert_task(Path(p))

    def _add_folder(self) -> None:
        directory = filedialog.askdirectory(title="选择文件夹")
        if not directory:
            return
        found = 0
        added = 0
        for f in sorted(Path(directory).rglob("*")):
            if f.is_file() and f.suffix.lower() in MEDIA_EXTS:
                found += 1
                if self._insert_task(f):
                    added += 1
        if added:
            self._log(f"从文件夹添加了 {added} 个文件")
        elif found:
            messagebox.showinfo("提示", "所选文件夹中的视频/音频文件均已在列表中")
        else:
            messagebox.showinfo("提示", "所选文件夹中未找到视频/音频文件")

    def _insert_task(self, path: Path) -> bool:
        """Insert a task for the given file. Returns True if added, False if duplicate."""
        path = path.resolve()
        input_str = str(path)
        if any(t["input"] == input_str for t in self.tasks.values()):
            return False

        self._task_counter += 1
        task_id = f"task_{self._task_counter}"
        output_path = path.with_suffix(".srt")

        self.tasks[task_id] = {
            "input": str(path),
            "output": str(output_path),
            "status": ST_PENDING,
            "progress": 0.0,
            "count": 0,
            "detail": "",
        }
        self.tree.insert(
            "",
            tk.END,
            iid=task_id,
            values=(path.name, output_path.name, ST_PENDING, ""),
        )
        return True

    def _remove_selected(self) -> None:
        skipped = 0
        for item_id in self.tree.selection():
            task = self.tasks.get(item_id)
            if task and task["status"] in (ST_LOADING, ST_RUNNING):
                skipped += 1
                continue
            self.tree.delete(item_id)
            self.tasks.pop(item_id, None)
        if skipped:
            messagebox.showinfo("提示", f"{skipped} 个正在处理的任务无法移除")

    def _clear_tasks(self) -> None:
        if self.running:
            messagebox.showwarning("提示", "请先停止处理再清空列表")
            return
        for item_id in list(self.tasks):
            self.tree.delete(item_id)
        self.tasks.clear()

    # ── Processing Control ────────────────────────────────────────────────

    def _start_processing(self) -> None:
        pending_ids = [
            tid
            for tid, task in self.tasks.items()
            if task["status"] in (ST_PENDING, ST_FAIL, ST_CANCEL)
        ]
        if not pending_ids:
            messagebox.showinfo("提示", "没有待处理的任务")
            return

        # Validate numeric parameters
        try:
            concurrency = self.var_concurrency.get()
            beam_size = self.var_beam_size.get()
            batch_size = self.var_batch_size.get()
        except tk.TclError:
            messagebox.showerror(
                "参数错误", "并发任务数、Beam Size 和批量大小必须为整数"
            )
            return

        if concurrency < 1 or beam_size < 1 or batch_size < 1:
            messagebox.showerror(
                "参数错误", "并发任务数、Beam Size 和批量大小必须大于 0"
            )
            return

        # Warn about existing output files
        existing = [
            Path(self.tasks[tid]["output"]).name
            for tid in pending_ids
            if Path(self.tasks[tid]["output"]).exists()
        ]
        if existing:
            names = "\n".join(f"  • {n}" for n in existing[:10])
            if len(existing) > 10:
                names += f"\n  … 等共 {len(existing)} 个文件"
            if not messagebox.askokcancel(
                "确认覆盖",
                f"以下 {len(existing)} 个输出文件已存在，将被覆盖:\n{names}",
            ):
                return

        self.running = True
        self.cancel_event.clear()
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

        model_id = self.var_model_path.get().strip() or self.var_model.get()
        device = self.var_device.get()
        language = LANG_MAP.get(self.var_language.get())
        use_vad = self.var_vad.get()

        self._log(
            f"开始处理 {len(pending_ids)} 个任务 "
            f"(并发={concurrency}, 模型={model_id}, 设备={device})"
        )
        if concurrency > 1:
            self._log(
                f"提示: 并发={concurrency} 会在每个线程各加载一份模型，请确保内存充足"
            )

        self.executor = ThreadPoolExecutor(max_workers=concurrency)

        for tid in pending_ids:
            task = self.tasks[tid]
            task["status"] = ST_PENDING
            task["progress"] = 0.0
            task["count"] = 0
            task["detail"] = ""
            self._update_row(tid)

            self.executor.submit(
                _worker,
                task_id=tid,
                input_path=task["input"],
                output_path=task["output"],
                model_id=model_id,
                device=device,
                language=language,
                beam_size=beam_size,
                batch_size=batch_size,
                use_vad=use_vad,
                msg_queue=self.msg_queue,
                cancel_event=self.cancel_event,
            )

        threading.Thread(target=self._wait_for_completion, daemon=True).start()

    def _wait_for_completion(self) -> None:
        """Block until all submitted futures finish, then notify UI."""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.msg_queue.put({"kind": "all_done"})

    def _stop_processing(self) -> None:
        self.cancel_event.set()
        self.btn_stop.configure(state=tk.DISABLED)
        self._log("正在停止，等待当前片段处理完成…")

    # ── Message Polling (UI thread) ───────────────────────────────────────

    def _poll_messages(self) -> None:
        """Drain the message queue and update UI. Called every 80 ms."""
        for _ in range(300):
            try:
                msg = self.msg_queue.get_nowait()
            except Empty:
                break
            try:
                self._handle_message(msg)
            except Exception as exc:
                self._log(f"[内部错误] {exc}")
        self.root.after(80, self._poll_messages)

    def _handle_message(self, msg: dict) -> None:
        kind = msg["kind"]

        if kind == "all_done":
            self.running = False
            self.btn_start.configure(state=tk.NORMAL)
            self.btn_stop.configure(state=tk.DISABLED)
            self._log("全部任务处理完毕")
            return

        task_id = msg.get("task_id")
        task = self.tasks.get(task_id)
        if not task:
            return

        if kind == "status":
            status = msg["value"]
            detail = msg.get("detail", "")
            task["status"] = status
            task["detail"] = detail

            if status == ST_DONE:
                task["progress"] = 1.0
                self._update_row(task_id, tag="done")
                self._log(f"[完成] {Path(task['input']).name}  {detail}")
            elif status == ST_FAIL:
                self._update_row(task_id, tag="fail")
                self._log(f"[失败] {Path(task['input']).name}  {detail}")
                tb = msg.get("traceback")
                if tb:
                    for line in tb.strip().splitlines():
                        self._log(f"  {line}")
            elif status == ST_CANCEL:
                self._update_row(task_id, tag="cancel")
                self._log(f"[取消] {Path(task['input']).name}")
            elif status == ST_LOADING:
                self._update_row(task_id, tag="run")
                self._log(f"[加载模型] {Path(task['input']).name}")
            elif status == ST_RUNNING:
                self._update_row(task_id, tag="run")

        elif kind == "progress":
            task["progress"] = msg["value"]
            task["count"] = msg["count"]
            self._update_row(task_id, tag="run")

        elif kind == "meta":
            lang = msg["language"]
            prob = msg["probability"]
            dur = msg["duration"]
            self._log(
                f"  语言={lang} ({prob:.0%}), "
                f"时长={format_duration(dur)} — {Path(task['input']).name}"
            )

    def _update_row(self, task_id: str, tag: str = "") -> None:
        """Refresh one row in the Treeview."""
        task = self.tasks[task_id]
        status = task["status"]
        progress = task["progress"]
        count = task.get("count", 0)
        detail = task.get("detail", "")

        if status == ST_RUNNING:
            bar_width = 20
            filled = int(bar_width * progress)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
            progress_text = f"{bar} {progress:.0%} ({count} 条)"
        elif status == ST_DONE:
            progress_text = f"100%  {detail}"
        elif status == ST_FAIL:
            progress_text = detail[:50]
        else:
            progress_text = ""

        self.tree.item(
            task_id,
            values=(
                Path(task["input"]).name,
                Path(task["output"]).name,
                status,
                progress_text,
            ),
            tags=(tag,) if tag else (),
        )

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ── Window Lifecycle ──────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askokcancel(
                "确认退出", "任务正在处理中，确定要退出吗？"
            ):
                return
            self.cancel_event.set()
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
