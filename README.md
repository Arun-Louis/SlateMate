# SlateMate

**AI-powered pedagogical assistant that generates real-time Manim animations from STEM lectures.**

SlateMate listens to a lecture audio file and lets the professor (or student) press a button at any moment to generate an animation of the concept being explained. It captures the last 30 seconds of audio, transcribes it with Whisper, sends it to Claude AI, and renders a Manim animation — all automatically.

---

## Pipeline

```
Audio File
    │
    ▼
Whisper (transcription)
    │
    ▼
Claude AI (concept extraction + Manim code generation)
    │
    ▼
Code Sanitizer (auto-fixes common LLM Manim mistakes)
    │
    ▼
Manim CLI (renders animation)
    │
    ▼
Animation plays automatically
```

---

## Requirements

- Python 3.14
- ffmpeg
- MiKTeX (LaTeX — required for MathTex rendering)
- Anthropic API key (Claude)

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/SlateMate.git
cd SlateMate
```

### 2. Install Python dependencies
```bash
pip install manim openai-whisper anthropic requests
```

### 3. Install ffmpeg

**Windows (winget):**
```bash
winget install ffmpeg
```
Then add ffmpeg to your PATH. The binary will be at:
```
C:\Users\YOUR_USERNAME\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-...\bin
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 4. Install MiKTeX (Windows only — for LaTeX rendering)

Download and install from **https://miktex.org/download**, then add to PATH:
```
C:\Users\YOUR_USERNAME\AppData\Local\Programs\MiKTeX\miktex\bin\x64
```

### 5. Set your Anthropic API key

**Windows (temporary — current session only):**
```bash
set ANTHROPIC_API_KEY=your_key_here
```

**Windows (permanent):**
1. Search "Environment Variables" in Start Menu
2. Under User Variables → New
3. Variable name: `ANTHROPIC_API_KEY`, Value: your key

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Get a key at **https://console.anthropic.com**

---

## Running SlateMate

```bash
python src/app.py
```

> **Windows note:** If you have the Microsoft Store Python, use the full path:
> ```bash
> C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python314\python.exe src\app.py
> ```

---

## How to Use

1. Click **Load Lecture Audio** and select an `.mp3`, `.mp4`, `.wav`, or `.m4a` file
2. Audio starts playing automatically
3. Use the **slider** to jump to any point in the audio
4. Use **⏮ -10s** / **+10s ⏭** to skip backward or forward
5. Press **⏸ Pause** / **▶ Resume** to pause and resume
6. At any moment, press **⚡ Visualize Last 30s** to generate an animation
7. The animation opens automatically when ready (~10–20 seconds)

---

## Project Structure

```
SlateMate/
│
├── src/
│   ├── app.py              # Main UI — audio player + pipeline controller
│   ├── transcribe.py       # Extracts 30s clip and transcribes with Whisper
│   ├── llm_generator.py    # Sends transcript to Claude, returns Manim code
│   ├── renderer.py         # Runs Manim CLI and returns video path
│   └── code_sanitizer.py   # Auto-fixes common LLM-generated Manim errors
│
├── temp/                   # Temporary files (auto-created)
├── output/                 # Rendered animations (auto-created)
├── data/
│   └── raw_videos/         # Place lecture audio/video files here
│
├── requirements.txt
└── README.md
```

---

## Configuration

**`src/llm_generator.py`**
```python
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # swap to "claude-sonnet-4-6" for better quality
```

**`src/transcribe.py`**
```python
CLIP_DURATION = 30      # seconds to capture on button press
WHISPER_MODEL = "base"  # swap to "large-v3" for better accuracy
```

**`src/renderer.py`**
```python
PYTHON_EXE = r"C:\Users\...\Python314\python.exe"  # update to your Python path
```

---

## Troubleshooting

**`manim` not recognized**
Use `python -m manim` instead. Verify with:
```bash
python -m manim --version
```

**`ffmpeg` not found**
Make sure ffmpeg's `bin` folder is in PATH and restart your terminal.

**LaTeX / MathTex errors**
Install MiKTeX from https://miktex.org and add to PATH:
```
C:\Users\YOUR_USERNAME\AppData\Local\Programs\MiKTeX\miktex\bin\x64
```

**Render keeps failing**
SlateMate auto-retries once with error feedback to Claude. If it fails twice, press the button again at a slightly different timestamp.

**FP16 warning from Whisper**
Normal on machines without a GPU — transcription still works correctly.

---

## Contributing

Each team member should work on their own branch:
```bash
git checkout -b feature/your-name
git add .
git commit -m "description of changes"
git push origin feature/your-name
```
Then open a pull request to merge into `main`.

---

## Team

Built as part of the ASU Capstone Project.
