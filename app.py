import io
import os
import string
import tempfile
import streamlit as st
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="Whisper Small — Algerian Darja ASR",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (CSS & RTL) ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5rem;
        direction: rtl;
    }
    .darja-output {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.4rem;
        line-height: 1.8;
        direction: rtl;
        text-align: right;
        background-color: #f8f9fa;
        color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    .metric-box {
        background: #f1f3f5;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Normalization Utilities ---
_ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670"
_TATWEEL = "\u0640"
_PUNCT_MAP = {ord(c): None for c in string.punctuation + "\u060C\u061B\u061F\u00AB\u00BB"}

def normalize_darja(text: str) -> str:
    if not text:
        return ""
    text = text.translate({ord(c): None for c in _ARABIC_DIACRITICS})
    text = text.replace(_TATWEEL, "")
    text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064A")
    text = text.translate(_PUNCT_MAP)
    return " ".join(text.split()).strip()

# --- Model Loader (Cached) ---
MODEL_ID = "openai/whisper-small"
ADAPTER_ID = "touati-kamel/whisper-algerian-darja-small"

@st.cache_resource(show_spinner="Loading Whisper Algerian Darja Model...")
def load_asr_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="arabic", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    return processor, model

processor, model = load_asr_model()

# --- Header ---
st.markdown('<div class="main-title">🎙️ Whisper Small — Algerian Darja ASR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">التعرف الآلي على الكلام بالدارجة الجزائرية</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("📌 Model Information")
    st.markdown("""
    - **Base Model**: `openai/whisper-small`
    - **Adapter ID**: [`touati-kamel/whisper-algerian-darja-small`](https://huggingface.co/touati-kamel/whisper-algerian-darja-small)
    - **Architecture**: LoRA (r=64, α=128)
    - **Language**: Algerian Arabic (*Darja / الدارجة*)
    """)
    
    st.subheader("📊 Benchmark WER")
    st.markdown("""
    | Dataset | WER (%) |
    | :--- | :---: |
    | **Loubna Stories** | **14.87%** |
    | **Rawi Storytelling** | **27.54%** |
    | **Kahwa Podcast** | **34.85%** |
    """)
    
    st.info("Trained sequentially over the OddAdmix Algerian speech collection.")

# --- Audio Input Tabs ---
tab_mic, tab_upload = st.tabs(["🎤 Record Microphone", "📁 Upload Audio File"])

audio_bytes = None
audio_source_label = ""

with tab_mic:
    st.write("Click the button below to record your voice in Algerian Darja:")
    recorded_audio = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="darja_mic_recorder",
        use_container_width=True
    )
    if recorded_audio and "bytes" in recorded_audio:
        audio_bytes = recorded_audio["bytes"]
        audio_source_label = "Microphone Recording"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, OGG, M4A, FLAC):",
        type=["wav", "mp3", "ogg", "m4a", "flac"]
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_source_label = f"Uploaded File: {uploaded_file.name}"

# --- Processing & Output ---
if audio_bytes:
    st.divider()
    st.subheader("🔊 Audio Preview")
    st.audio(audio_bytes)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        apply_norm = st.checkbox("Apply Darja Text Normalization (تطبيق التنظيف الإملائي)", value=True)
    
    if st.button("🚀 Transcribe Audio (تحويل الصوت إلى نص)", type="primary", use_container_width=True):
        with st.spinner("⏳ Transcribing speech in Algerian Darja..."):
            try:
                # Save to temporary file for librosa loading
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name

                # Load at 16kHz
                audio_array, sr = librosa.load(tmp_path, sr=16000)
                os.remove(tmp_path)

                # Process features
                input_features = processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features

                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language="arabic",
                    task="transcribe"
                )

                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=225
                    )

                raw_transcription = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

                final_text = normalize_darja(raw_transcription) if apply_norm else raw_transcription

                # Display Results
                st.success("✅ Transcription Complete!")
                st.markdown(f'<div class="darja-output">{final_text}</div>', unsafe_allow_html=True)
                
                # Copy friendly display & Stats
                st.text_area("Text Output (for easy copy):", value=final_text, height=90)
                
                word_count = len(final_text.split())
                char_count = len(final_text)
                st.caption(f"Audio Duration: {len(audio_array)/16000:.2f}s | Words: {word_count} | Characters: {char_count}")

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

# --- Footer ---
st.divider()
st.caption("Built with Whisper Small, LoRA (PEFT), and Streamlit. Model checkpoint hosted on Hugging Face Hub: [touati-kamel/whisper-algerian-darja-small](https://huggingface.co/touati-kamel/whisper-algerian-darja-small).")
