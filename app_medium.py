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
    page_title="Whisper Medium — Algerian Darja ASR",
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
        background: linear-gradient(90deg, #1d8cf8, #3358f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5rem;
        direction: rtl;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .darja-output {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.45rem;
        line-height: 1.9;
        direction: rtl;
        text-align: right;
        background-color: #f8f9fa;
        color: #111827;
        padding: 1.3rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
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

def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decodes audio bytes of any format (WebM, OGG/Opus, MP3, WAV, M4A, FLAC)
    into a 16000Hz mono float32 numpy array.
    """
    # Method 1: pydub
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_frame_rate(16000).set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.sample_width == 2:
            samples /= 32768.0
        elif seg.sample_width == 4:
            samples /= 2147483648.0
        elif seg.sample_width == 1:
            samples = (samples - 128) / 128.0
        if len(samples) > 0:
            return samples
    except Exception:
        pass

    # Method 2: Direct ffmpeg subprocess
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as in_tmp:
            in_tmp.write(audio_bytes)
            in_path = in_tmp.name
        out_path = in_path + "_converted.wav"
        
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        import soundfile as sf
        samples, _ = sf.read(out_path, dtype="float32")
        
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        if len(samples) > 0:
            return samples
    except Exception:
        pass

    # Method 3: Direct soundfile in-memory read
    try:
        import soundfile as sf
        samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        return samples
    except Exception as e:
        raise RuntimeError(f"Audio decoding error: {e}. Please ensure valid audio data is recorded/uploaded.")

# --- Model Configuration ---
MODEL_ID = "openai/whisper-medium"
ADAPTER_ID = "touati-kamel/whisper-algerian-darja-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

@st.cache_resource(show_spinner="Loading Whisper Medium Algerian Darja Model (769M Params)...")
def load_asr_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="arabic", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    return processor, model

processor, model = load_asr_model()

# --- Header ---
st.markdown('<div class="main-title">Whisper Medium — Algerian Darja ASR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">التعرف الآلي عالي الدقة على الكلام بالدارجة الجزائرية</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Model Specifications")
    st.markdown("""
    - **Base Model**: `openai/whisper-medium`
    - **LoRA Adapter**: [`touati-kamel/whisper-algerian-darja-medium`](https://huggingface.co/touati-kamel/whisper-algerian-darja-medium)
    - **Parameters**: 833M total (69.2M trainable LoRA)
    - **Architecture**: 24 Encoder / 24 Decoder layers
    - **Target Dialect**: Algerian Arabic (*Darja / الدارجة*)
    - **Device**: `{}`
    """.format(DEVICE.upper()))
    
    st.subheader("Benchmark WER")
    st.markdown("""
    | Dataset Split | WER (%) |
    | :--- | :---: |
    | **Loubna Stories** | **0.34%** |
    | **Kahwa Podcast** | **0.68%** |
    | **Rawi Storytelling** | **0.95%** |
    """)
    
    st.info("Curriculum Learning: Fine-tuned over 31,661 steps across conversational podcasts, spontaneous storytelling, and cultural narratives.")

# --- Session State Initialization ---
if "active_audio" not in st.session_state:
    st.session_state.active_audio = None
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = None
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = 0.0

# --- Audio Input Tabs ---
tab_mic, tab_upload = st.tabs(["Record Microphone", "Upload Audio File"])

with tab_mic:
    st.write("Click below and speak naturally in Algerian Darja:")
    recorded_audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        key="darja_medium_mic_recorder",
        use_container_width=True
    )
    if recorded_audio and "bytes" in recorded_audio and len(recorded_audio["bytes"]) > 0:
        st.session_state.active_audio = recorded_audio["bytes"]

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an Algerian audio clip (WAV, MP3, OGG, M4A, FLAC):",
        type=["wav", "mp3", "ogg", "m4a", "flac"],
        key="darja_medium_file_uploader"
    )
    if uploaded_file is not None:
        st.session_state.active_audio = uploaded_file.read()

# --- Processing & Output ---
if st.session_state.active_audio:
    st.divider()
    st.subheader("Audio Playback")
    st.audio(st.session_state.active_audio)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        apply_norm = st.checkbox("Apply Darja Text Normalization (تنظيف وتوحيد الحروف)", value=True)
    with col2:
        if st.button("Clear Audio"):
            st.session_state.active_audio = None
            st.session_state.transcription_text = None
            st.rerun()
    
    if st.button("Transcribe Speech (تحويل الصوت إلى نص)", type="primary", use_container_width=True):
        with st.spinner("Transcribing with Whisper Medium Algerian Darja..."):
            try:
                # Robustly decode audio to 16kHz mono float32
                audio_array = decode_audio_bytes(st.session_state.active_audio)

                if len(audio_array) == 0:
                    st.warning("The audio input is empty. Please record or upload a valid audio sample.")
                else:
                    # Feature Extraction
                    input_features = processor(
                        audio_array,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features

                    if torch.cuda.is_available():
                        input_features = input_features.to("cuda", dtype=DTYPE)

                    try:
                        forced_decoder_ids = processor.get_decoder_prompt_ids(
                            language="arabic",
                            task="transcribe"
                        )
                    except Exception:
                        forced_decoder_ids = None

                    with torch.no_grad():
                        if forced_decoder_ids is not None:
                            predicted_ids = model.generate(
                                input_features,
                                forced_decoder_ids=forced_decoder_ids,
                                max_new_tokens=225
                            )
                        else:
                            predicted_ids = model.generate(
                                input_features,
                                language="arabic",
                                task="transcribe",
                                max_new_tokens=225
                            )

                    raw_transcription = processor.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0]

                    st.session_state.transcription_text = normalize_darja(raw_transcription) if apply_norm else raw_transcription
                    st.session_state.audio_duration = len(audio_array) / 16000.0

            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")

# Display Persisted Results
if st.session_state.transcription_text:
    st.success("Transcription Complete.")
    st.markdown(f'<div class="darja-output">{st.session_state.transcription_text}</div>', unsafe_allow_html=True)
    
    # Text area for easy copy
    st.text_area("Text Output (for easy copying):", value=st.session_state.transcription_text, height=100)
    
    word_count = len(st.session_state.transcription_text.split())
    char_count = len(st.session_state.transcription_text)
    st.caption(f"Audio Duration: **{st.session_state.audio_duration:.2f}s** | Words: **{word_count}** | Characters: **{char_count}**")

# --- Footer ---
st.divider()
st.caption("Developed by **Kamel Touati** | Powered by OpenAI Whisper Medium, PEFT (LoRA), and Streamlit. [Hugging Face Model](https://huggingface.co/touati-kamel/whisper-algerian-darja-medium).")
