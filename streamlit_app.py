from pathlib import Path

import streamlit as st
import whisper

from transcribe import get_transcript_at
from visual_classifier import predict_visual_need
from llm_generator import ManimCodeGenerator
from renderer import render


st.set_page_config(
    page_title="SlateMate",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 SlateMate")
st.subheader("Button-Triggered Near-Real-Time Lecture Visualization")

st.write(
    "Upload a lecture audio/video file, choose the current timestamp, "
    "and SlateMate will visualize the last 30 seconds if the segment is visually worthy."
)

uploaded_file = st.file_uploader(
    "Upload lecture audio/video",
    type=["mp3", "mp4", "wav", "m4a", "mkv"]
)

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

if "generator" not in st.session_state:
    st.session_state.generator = None

if uploaded_file is not None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    audio_path = data_dir / uploaded_file.name
    audio_path.write_bytes(uploaded_file.read())

    st.success(f"Uploaded: {uploaded_file.name}")
    st.audio(str(audio_path))

    timestamp = st.number_input(
        "Enter current lecture timestamp in seconds",
        min_value=0.0,
        value=60.0,
        step=5.0
    )

    if st.button("⚡ Visualize Last 30 Seconds"):
        with st.spinner("Loading Whisper model..."):
            if st.session_state.whisper_model is None:
                st.session_state.whisper_model = whisper.load_model("base")

        with st.spinner("Step 1/4 — Transcribing last 30 seconds..."):
            transcript = get_transcript_at(
                str(audio_path),
                timestamp,
                model=st.session_state.whisper_model
            )

        st.subheader("Transcript")
        st.write(transcript)

        with st.spinner("Step 2/4 — Checking visual worthiness using DistilBERT..."):
            prediction = predict_visual_need(transcript)

        st.subheader("DistilBERT Prediction")

        col1, col2 = st.columns(2)
        col1.metric("Predicted Label", prediction["label"])
        col2.metric("Confidence", f"{prediction['confidence']:.2f}")

        if prediction["label"] == "NO_VISUAL":
            st.warning("This segment was classified as NO_VISUAL. No animation generated.")
            st.stop()

        st.success("This segment is visually worthy. Generating Manim animation...")

        with st.spinner("Step 3/4 — Generating Manim code with Claude..."):
            if st.session_state.generator is None:
                st.session_state.generator = ManimCodeGenerator()

            generation_result = st.session_state.generator.generate(transcript)

        if not generation_result.success:
            st.error(f"Code generation failed: {generation_result.error}")
            st.stop()

        st.subheader("Generated Manim Code")
        st.code(generation_result.code, language="python")

        with st.spinner("Step 4/4 — Rendering Manim animation..."):
            render_result = render(generation_result.code)

        if not render_result.success:
            st.error(f"Render failed: {render_result.error}")
            st.stop()

        st.success("Animation rendered successfully!")
        st.subheader("Generated Animation")
        st.video(render_result.video_path)

else:
    st.info("Upload a lecture audio or video file to begin.")