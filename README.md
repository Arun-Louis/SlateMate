# SlateMate

**AI-powered pedagogical assistant that generates real-time Manim animations from STEM lectures.**

SlateMate captures a 30-second window of lecture audio, transcribes it with Whisper, classifies whether it needs a visualization using a fine-tuned DistilBERT model, and if so, generates and renders a Manim animation using Claude AI — all from a simple web interface.

---

## How It Works

```
Upload lecture audio/video
         │
         ▼
Whisper transcribes last 30 seconds
         │
         ▼
DistilBERT classifies: VISUAL or NO_VISUAL
         │
    VISUAL only
         │
         ▼
Claude AI generates Manim Python code
         │
         ▼
Code Sanitizer fixes common errors
         │
         ▼
Manim renders the animation
         │
         ▼
Animation displays in browser
```

If the classifier returns **NO_VISUAL**, the pipeline stops early — no LLM call is made.

---

## Requirements

- Python 3.10 or higher
- ffmpeg
- MiKTeX (Windows) or a LaTeX distribution (Mac/Linux) — required for math rendering
- Anthropic API key

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/SlateMate.git
cd SlateMate
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

**Windows:**
```bash
winget install ffmpeg
```
Add ffmpeg to PATH after installing. It will be located at:
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

### 4. Install LaTeX

**Windows — MiKTeX:**
Download from **https://miktex.org/download** and install. Then add to PATH:
```
C:\Users\YOUR_USERNAME\AppData\Local\Programs\MiKTeX\miktex\bin\x64
```

**Mac:**
```bash
brew install --cask mactex
```

**Linux:**
```bash
sudo apt install texlive
```

### 5. Set your Anthropic API key

**Windows:**
```bash
set ANTHROPIC_API_KEY=your_key_here
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Get a key at **https://console.anthropic.com**

### 6. Add the DistilBERT model

The fine-tuned DistilBERT classifier must be placed at:
```
SlateMate/
└── distilbert_binary_outputs/
    └── best_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer_config.json
        └── vocab.txt
```

Download the model from the team shared drive and place it in the folder above.

---

## Running SlateMate

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

> The `--server.fileWatcherType none` flag is required — without it, Streamlit conflicts with the Transformers library and shows a blank page.

Then open your browser at **http://localhost:8501**

---

## How to Use

1. Upload a lecture audio or video file (`.mp3`, `.mp4`, `.wav`, `.m4a`, `.mkv`)
2. Enter the timestamp in seconds where you want to analyze (e.g. `90` for the 1:30 mark)
3. Click **⚡ Visualize Last 30 Seconds**
4. SlateMate will:
   - Transcribe the 30 seconds ending at that timestamp
   - Run the DistilBERT classifier
   - If VISUAL: generate and render a Manim animation
   - If NO_VISUAL: show a message and stop (no animation needed)

---

## Project Structure

```
SlateMate/
│
├── streamlit_app.py          # Main web interface
├── transcribe.py             # Extracts audio clip and transcribes with Whisper
├── visual_classifier.py      # Loads DistilBERT model and predicts VISUAL/NO_VISUAL
├── llm_generator.py          # Sends transcript to Claude, returns Manim code
├── renderer.py               # Runs Manim CLI and returns video path
├── code_sanitizer.py         # Auto-fixes common LLM-generated Manim errors
│
├── train_distilbert_binary_from_excel_folder.py   # Training script for DistilBERT
├── train_tfidf_logreg_binary_from_excel_folder.py # Baseline TF-IDF + LogReg training
│
├── distilbert_binary_outputs/
│   └── best_model/           # Fine-tuned DistilBERT model (add manually)
│
├── data/                     # Uploaded lecture files (auto-created)
├── temp/                     # Temporary audio clips and scene files (auto-created)
├── output/                   # Rendered animation videos (auto-created)
│
└── requirements.txt
```

---

## Configuration

**`llm_generator.py`** — change the Claude model:
```python
CLAUDE_MODEL = "claude-haiku-4-5-20251001"   # fast and cheap
# or
CLAUDE_MODEL = "claude-sonnet-4-6"            # better quality
```

**`transcribe.py`** — change clip duration or Whisper model:
```python
CLIP_DURATION = 30      # seconds to capture
WHISPER_MODEL = "base"  # or "large-v3" for better accuracy
```

**`visual_classifier.py`** — change model path if needed:
```python
MODEL_DIR = Path("distilbert_binary_outputs") / "best_model"
```

---

## Training the Classifier (Optional)

If you want to retrain the DistilBERT classifier on new data:

```bash
python train_distilbert_binary_from_excel_folder.py
```

To run the baseline TF-IDF + Logistic Regression model:
```bash
python train_tfidf_logreg_binary_from_excel_folder.py
```

Both scripts expect ground truth Excel files in the `data/ground_truth_files/` folder (MaViLS format).

---

## Troubleshooting

**Blank page in Streamlit**
Always launch with `--server.fileWatcherType none`:
```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

**`ffmpeg` not found**
Make sure ffmpeg's `bin` folder is in your PATH and restart your terminal.

**LaTeX / MathTex errors**
Make sure MiKTeX is installed and its `bin/x64` folder is in PATH. Restart your terminal after adding it.

**DistilBERT model not found**
Make sure the model folder exists at `distilbert_binary_outputs/best_model/` with all required files inside.

**Render keeps failing**
The code sanitizer automatically fixes common LLM Manim errors. If it still fails, try a different timestamp — some transcript segments are easier for the model to animate than others.

**`torchvision` missing**
```bash
pip install torchvision
```

**FP16 warning from Whisper**
```
FP16 is not supported on CPU; using FP32 instead
```
This is normal on machines without a GPU. Transcription still works correctly.

---

## Team

Built as part of the FSE 570 Capstone Project — Arizona State University, Spring 2025.
