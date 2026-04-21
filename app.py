

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import os
import subprocess
import sys
from pathlib import Path

from transcribe import get_transcript_at
from llm_generator import ManimCodeGenerator
from renderer import render

import whisper
import subprocess as sp


def get_audio_duration(path: str) -> float:
    """Use ffprobe to get audio duration in seconds."""
    try:
        result = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


class SlateMateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SlateMate")
        self.root.geometry("540x500")
        self.root.configure(bg="#0D1B2A")

        self.audio_path       = None
        self.audio_duration   = 0.0
        self.start_time       = None
        self.elapsed_before   = 0.0
        self.is_playing       = False
        self.is_paused        = False
        self.audio_process    = None
        self.whisper_model    = None
        self.generator        = None
        self._slider_dragging = False

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#0D1B2A")
        header.pack(pady=16)

        tk.Label(
            header, text="Slate",
            font=("Arial", 26, "bold"), fg="white", bg="#0D1B2A"
        ).pack(side="left")
        tk.Label(
            header, text="Mate",
            font=("Arial", 26, "bold"), fg="#00C2A8", bg="#0D1B2A"
        ).pack(side="left")

        tk.Label(
            self.root, text="AI-Powered Lecture Visualizer",
            font=("Arial", 10), fg="#8BAFC7", bg="#0D1B2A"
        ).pack()

        # Divider
        tk.Frame(self.root, bg="#162436", height=1).pack(fill="x", padx=20, pady=10)

        # Status
        self.status_var = tk.StringVar(value="Load a lecture audio file to begin")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Arial", 10), fg="#8BAFC7", bg="#0D1B2A",
            wraplength=490
        ).pack(pady=4)

        # Time display
        time_frame = tk.Frame(self.root, bg="#0D1B2A")
        time_frame.pack(fill="x", padx=30)

        self.current_time_var = tk.StringVar(value="00:00")
        tk.Label(
            time_frame, textvariable=self.current_time_var,
            font=("Arial", 10), fg="#00C2A8", bg="#0D1B2A"
        ).pack(side="left")

        self.total_time_var = tk.StringVar(value="00:00")
        tk.Label(
            time_frame, textvariable=self.total_time_var,
            font=("Arial", 10), fg="#8BAFC7", bg="#0D1B2A"
        ).pack(side="right")

        # Slider
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = tk.Scale(
            self.root,
            variable=self.slider_var,
            from_=0, to=100,
            orient="horizontal",
            length=480,
            showvalue=False,
            bg="#162436", fg="#00C2A8",
            troughcolor="#162436",
            highlightthickness=0,
            sliderlength=16,
            bd=0,
            command=self._on_slider_move
        )
        self.slider.pack(padx=20, pady=4)
        self.slider.bind("<ButtonPress-1>", self._slider_press)
        self.slider.bind("<ButtonRelease-1>", self._slider_release)

        # Load button
        tk.Button(
            self.root, text="📂  Load Lecture Audio",
            command=self._load_audio,
            font=("Arial", 11), bg="#162436", fg="white",
            relief="flat", padx=15, pady=7
        ).pack(pady=6)

        # Playback controls
        controls_frame = tk.Frame(self.root, bg="#0D1B2A")
        controls_frame.pack(pady=4)

        tk.Button(
            controls_frame, text="⏮ -10s",
            command=lambda: self._seek_relative(-10),
            font=("Arial", 10), bg="#162436", fg="white",
            relief="flat", padx=10, pady=6
        ).pack(side="left", padx=4)

        self.pause_btn = tk.Button(
            controls_frame, text="⏸  Pause",
            command=self._toggle_pause,
            font=("Arial", 11), bg="#162436", fg="white",
            relief="flat", padx=12, pady=6,
            state="disabled"
        )
        self.pause_btn.pack(side="left", padx=4)

        self.stop_btn = tk.Button(
            controls_frame, text="⏹  Stop",
            command=self._stop_audio,
            font=("Arial", 11), bg="#8B2020", fg="white",
            relief="flat", padx=12, pady=6,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)

        tk.Button(
            controls_frame, text="+10s ⏭",
            command=lambda: self._seek_relative(10),
            font=("Arial", 10), bg="#162436", fg="white",
            relief="flat", padx=10, pady=6
        ).pack(side="left", padx=4)

        # Divider
        tk.Frame(self.root, bg="#162436", height=1).pack(fill="x", padx=20, pady=10)

        # Animate button
        self.animate_btn = tk.Button(
            self.root,
            text="⚡  Visualize Last 30s",
            command=self._trigger_animation,
            font=("Arial", 14, "bold"),
            bg="#00C2A8", fg="#0D1B2A",
            relief="flat", padx=20, pady=12,
            state="disabled"
        )
        self.animate_btn.pack(pady=6)

        tk.Label(
            self.root,
            text="Press at any moment to generate an animation from the last 30 seconds",
            font=("Arial", 9), fg="#8BAFC7", bg="#0D1B2A"
        ).pack()

    # ── Audio controls ────────────────────────────────────────────────────────

    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select Lecture Audio/Video File",
            filetypes=[("Audio/Video files", "*.mp3 *.mp4 *.wav *.m4a *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return

        self.audio_path     = path
        self.audio_duration = get_audio_duration(path)
        self.slider.config(to=max(self.audio_duration, 1))
        self.total_time_var.set(self._fmt(self.audio_duration))
        self.status_var.set(f"Loaded: {Path(path).name}")

        self.animate_btn.config(state="normal")
        self.pause_btn.config(state="normal")
        self.stop_btn.config(state="normal")

        threading.Thread(target=self._load_models, daemon=True).start()
        self._start_playback(seek=0)

    def _load_models(self):
        self.status_var.set("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        self.status_var.set("Initializing SlateMate AI...")
        try:
            self.generator = ManimCodeGenerator()
            self.status_var.set("SlateMate ready! Press 'Visualize Last 30s' anytime.")
        except Exception as e:
            self.status_var.set(f"AI error: {str(e)[:60]}")

    def _start_playback(self, seek: float = 0):
        seek = max(0, min(seek, self.audio_duration))
        self.elapsed_before = seek
        self.start_time     = time.time()
        self.is_playing     = True
        self.is_paused      = False

        cmd = ["ffplay", "-nodisp", "-autoexit", "-ss", str(seek), self.audio_path]
        self.audio_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self.pause_btn.config(text="⏸  Pause")
        threading.Thread(target=self._update_clock, daemon=True).start()

    def _toggle_pause(self):
        if not self.is_playing and not self.is_paused:
            return
        if not self.is_paused:
            self.elapsed_before = self._current_elapsed()
            self.is_playing = False
            self.is_paused  = True
            if self.audio_process:
                self.audio_process.terminate()
            self.pause_btn.config(text="▶  Resume")
            self.status_var.set(f"Paused at {self._fmt(self.elapsed_before)}")
        else:
            self._start_playback(seek=self.elapsed_before)
            self.pause_btn.config(text="⏸  Pause")
            self.status_var.set("SlateMate ready! Press 'Visualize Last 30s' anytime.")

    def _stop_audio(self):
        self.is_playing = False
        self.is_paused  = False
        self.elapsed_before = 0.0
        self.start_time = None
        if self.audio_process:
            self.audio_process.terminate()
            self.audio_process = None
        self.pause_btn.config(text="⏸  Pause", state="disabled")
        self.stop_btn.config(state="disabled")
        self.animate_btn.config(state="disabled")
        self.current_time_var.set("00:00")
        self.slider_var.set(0)
        self.status_var.set("Stopped. Load a file to begin again.")

    def _seek_relative(self, delta: float):
        self._seek_to(self._current_elapsed() + delta)

    def _seek_to(self, position: float):
        position = max(0, min(position, self.audio_duration))
        was_paused = self.is_paused
        if self.audio_process:
            self.audio_process.terminate()
        if was_paused:
            self.elapsed_before = position
            self.is_paused  = True
            self.is_playing = False
            self.current_time_var.set(self._fmt(position))
            self.slider_var.set(position)
        else:
            self._start_playback(seek=position)

    # ── Slider ────────────────────────────────────────────────────────────────

    def _slider_press(self, event):
        self._slider_dragging = True

    def _slider_release(self, event):
        self._slider_dragging = False
        self._seek_to(self.slider_var.get())

    def _on_slider_move(self, value):
        if self._slider_dragging:
            self.current_time_var.set(self._fmt(float(value)))

    # ── Clock ─────────────────────────────────────────────────────────────────

    def _update_clock(self):
        while self.is_playing:
            elapsed = self._current_elapsed()
            self.current_time_var.set(self._fmt(elapsed))
            if not self._slider_dragging:
                self.slider_var.set(elapsed)
            if self.audio_duration > 0 and elapsed >= self.audio_duration:
                self.is_playing = False
                break
            time.sleep(0.5)

    def _current_elapsed(self) -> float:
        if self.is_paused or self.start_time is None:
            return self.elapsed_before
        return self.elapsed_before + (time.time() - self.start_time)

    def _fmt(self, seconds: float) -> str:
        s = int(seconds)
        return f"{s // 60:02d}:{s % 60:02d}"

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _trigger_animation(self):
        if not self.audio_path:
            messagebox.showwarning("No audio", "Please load a lecture audio file first.")
            return
        if self.generator is None:
            messagebox.showwarning("Not ready", "SlateMate is still loading, please wait.")
            return

        was_playing = self.is_playing and not self.is_paused
        if was_playing:
            self._toggle_pause()

        timestamp = self._current_elapsed()
        self.animate_btn.config(state="disabled")
        self.status_var.set(f"Captured at {self._fmt(timestamp)} — generating visualization...")

        threading.Thread(
            target=self._run_pipeline,
            args=(timestamp, was_playing),
            daemon=True
        ).start()

    def _run_pipeline(self, timestamp: float, resume_after: bool):
        try:
            self._set_status("Step 1/3 — Transcribing last 30s...")
            transcript = get_transcript_at(
                self.audio_path, timestamp, model=self.whisper_model
            )
            print(f"\n[SlateMate] Transcript:\n{transcript}\n")

            self._set_status("Step 2/3 — SlateMate AI generating animation...")
            result = self.generator.generate(transcript)

            if not result.success:
                self._show_error(f"Code generation failed: {result.error}")
                return

            print(f"[SlateMate] Code:\n{result.code}\n")

            self._set_status("Step 3/3 — Rendering visualization...")
            render_result = render(result.code)

            if not render_result.success:
                self._set_status("Retrying with error feedback...")
                retry_prompt = (
                    f"{transcript}\n\n"
                    f"Previous Manim code failed:\n{render_result.error}\n\n"
                    f"Fix it. Only use Arrow, Text, MathTex, Circle, Square, Axes. "
                    f"Use ORIGIN, UP, DOWN, LEFT, RIGHT for positions. No numpy."
                )
                result = self.generator.generate(retry_prompt)
                if not result.success:
                    self._show_error(f"Retry failed: {result.error}")
                    return
                render_result = render(result.code)
                if not render_result.success:
                    self._show_error(f"Render failed after retry:\n{render_result.error}")
                    return

            self._set_status("Visualization ready!")
            self._play_video(render_result.video_path)

        except Exception as e:
            self._show_error(str(e))
        finally:
            self.root.after(0, lambda: self.animate_btn.config(state="normal"))
            if resume_after:
                self.root.after(500, self._toggle_pause)
            else:
                self.root.after(0, lambda: self._set_status(
                    "SlateMate ready! Press 'Visualize Last 30s' anytime."
                ))

    def _play_video(self, video_path: str):
        print(f"[SlateMate] Playing: {video_path}")
        if sys.platform == "win32":
            os.startfile(video_path)
        else:
            subprocess.Popen(["ffplay", video_path])

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _show_error(self, message: str):
        print(f"[SlateMate] Error: {message}")
        self._set_status("Error — check console for details")
        self.root.after(0, lambda: messagebox.showerror("SlateMate Error", message))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = SlateMateApp(root)
    root.mainloop()