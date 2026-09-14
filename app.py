import io
import os
import time
import string
import tempfile
import difflib
import streamlit as st
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="Whisper Algerian Darja ASR — Small vs. Medium",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (CSS & RTL) ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: center;
        background: linear-gradient(135deg, #1d8cf8 0%, #3358f4 50%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
        direction: rtl;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .model-card-medium {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .model-card-small {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .darja-output-medium {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.35rem;
        line-height: 1.9;
        direction: rtl;
        text-align: right;
        background-color: #f0fdf4;
        color: #14532d;
        padding: 1.1rem;
        border-radius: 10px;
        border: 1px solid #bbf7d0;
        min-height: 110px;
    }
    .darja-output-small {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.35rem;
        line-height: 1.9;
        direction: rtl;
        text-align: right;
        background-color: #f8fafc;
        color: #1e293b;
        padding: 1.1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        min-height: 110px;
    }
    .badge-medium {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        border-radius: 9999px;
        background-color: #dcfce7;
        color: #15803d;
        margin-bottom: 8px;
    }
    .badge-small {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        border-radius: 9999px;
        background-color: #e2e8f0;
        color: #334155;
        margin-bottom: 8px;
    }
    .diff-box {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.95rem;
        color: #92400e;
        direction: rtl;
        text-align: right;
        margin-top: 0.8rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 700;
        height: 3.2rem;
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
        raise RuntimeError(f"Audio decoding error: {e}. Please ensure valid audio data is recorded or uploaded.")

# --- Model Definitions ---
MODELS_INFO = {
    "medium": {
        "title": "Whisper Medium (833M)",
        "base_model": "openai/whisper-medium",
        "adapter_id": "touati-kamel/whisper-algerian-darja-medium",
        "params": "833M total (69.2M LoRA)",
        "badge_class": "badge-medium",
        "badge_text": "⭐ SOTA — Best WER: 0.34%",
        "card_class": "model-card-medium",
        "output_class": "darja-output-medium",
        "wer": {"Loubna Stories": "0.34%", "Kahwa Podcast": "0.68%", "Rawi Stories": "0.95%"}
    },
    "small": {
        "title": "Whisper Small (267M)",
        "base_model": "openai/whisper-small",
        "adapter_id": "touati-kamel/whisper-algerian-darja-small",
        "params": "267M total (25.9M LoRA)",
        "badge_class": "badge-small",
        "badge_text": "⚡ Lightweight — Best WER: 14.87%",
        "card_class": "model-card-small",
        "output_class": "darja-output-small",
        "wer": {"Loubna Stories": "14.87%", "Rawi Stories": "27.54%", "Kahwa Podcast": "34.85%"}
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Model Loader (Cached per model) ---
@st.cache_resource(show_spinner="Loading Whisper Model...")
def load_whisper_model(model_key: str):
    info = MODELS_INFO[model_key]
    processor = WhisperProcessor.from_pretrained(info["base_model"], language="arabic", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        info["base_model"],
        torch_dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, info["adapter_id"])
    model.eval()
    return processor, model

def run_transcription(model_key: str, audio_array: np.ndarray, apply_norm: bool = True):
    processor, model = load_whisper_model(model_key)
    
    start_time = time.time()
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

    raw_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    latency = time.time() - start_time
    
    final_text = normalize_darja(raw_text) if apply_norm else raw_text
    return final_text, latency

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Evaluation & Comparison")
    
    inference_mode = st.radio(
        "Select Transcription Mode:",
        options=[
            "⚡ Compare Both Models (Side-by-Side)",
            "🚀 Whisper Medium (Recommended)",
            "🔹 Whisper Small (Lightweight)"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📊 Benchmark WER Comparison")
    st.markdown("""
    | Dataset Split | Medium (833M) | Small (267M) | Δ Gain |
    | :--- | :---: | :---: | :---: |
    | **Loubna Stories** | **0.34%** | 14.87% | **+97.7%** |
    | **Kahwa Podcast** | **0.68%** | 34.85% | **+98.0%** |
    | **Rawi Storytelling** | **0.95%** | 27.54% | **+96.5%** |
    """)
    
    st.markdown(f"**Hardware Device**: `{DEVICE.upper()}`")
    st.markdown("---")
    st.markdown("""
    - [Medium Model Card (HF)](https://huggingface.co/touati-kamel/whisper-algerian-darja-medium)
    - [Small Model Card (HF)](https://huggingface.co/touati-kamel/whisper-algerian-darja-small)
    - [OddAdmix Speech Collection](https://huggingface.co/oddadmix)
    """)

# --- Main Page Header ---
st.markdown('<div class="main-title">Whisper Algerian Darja ASR — Comparison Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">مقارنة التعرف الآلي على الكلام بالدارجة الجزائرية بين النموذجين المتوسط والصغير</div>', unsafe_allow_html=True)

# --- Session State Initialization ---
if "active_audio" not in st.session_state:
    st.session_state.active_audio = None
if "transcription_results" not in st.session_state:
    st.session_state.transcription_results = {}
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = 0.0

# --- Audio Input Tabs ---
tab_mic, tab_upload = st.tabs(["🎙️ Record Voice (ميكروفون)", "📁 Upload Audio File (ملف صوتي)"])

with tab_mic:
    st.write("Record your voice in Algerian Darja (any dialect):")
    recorded_audio = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="darja_compare_mic_recorder",
        use_container_width=True
    )
    if recorded_audio and "bytes" in recorded_audio and len(recorded_audio["bytes"]) > 0:
        st.session_state.active_audio = recorded_audio["bytes"]

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an Algerian audio file (WAV, MP3, OGG, M4A, FLAC):",
        type=["wav", "mp3", "ogg", "m4a", "flac"],
        key="darja_compare_file_uploader"
    )
    if uploaded_file is not None:
        st.session_state.active_audio = uploaded_file.read()

# --- Action & Output Section ---
if st.session_state.active_audio:
    st.divider()
    
    col_audio, col_ctrl = st.columns([2, 1])
    with col_audio:
        st.subheader("🎧 Audio Input Preview")
        st.audio(st.session_state.active_audio)
    
    with col_ctrl:
        st.write(" ")
        st.write(" ")
        apply_norm = st.checkbox("Apply Darja Text Normalization (تنظيف وتوحيد الحروف)", value=True)
        if st.button("🗑️ Clear Audio & Results"):
            st.session_state.active_audio = None
            st.session_state.transcription_results = {}
            st.session_state.audio_duration = 0.0
            st.rerun()

    # Transcribe Button
    button_label = "⚡ Run Side-by-Side Model Comparison (مقارنة النموذجين معاً)" if "Compare" in inference_mode else "⚡ Transcribe Audio (تحويل الصوت إلى نص)"
    
    if st.button(button_label, type="primary", use_container_width=True):
        st.session_state.transcription_results = {}
        
        try:
            audio_array = decode_audio_bytes(st.session_state.active_audio)
            
            if len(audio_array) == 0:
                st.warning("The input audio is empty. Please record or upload a valid audio sample.")
            else:
                st.session_state.audio_duration = len(audio_array) / 16000.0
                
                # Determine models to run
                models_to_run = []
                if "Compare" in inference_mode:
                    models_to_run = ["medium", "small"]
                elif "Medium" in inference_mode:
                    models_to_run = ["medium"]
                else:
                    models_to_run = ["small"]
                
                progress_text = "Processing audio with Whisper models..."
                progress_bar = st.progress(0, text=progress_text)
                
                for idx, m_key in enumerate(models_to_run):
                    m_title = MODELS_INFO[m_key]["title"]
                    progress_bar.progress(
                        int((idx / len(models_to_run)) * 100),
                        text=f"Transcribing with {m_title}..."
                    )
                    text_out, latency = run_transcription(m_key, audio_array, apply_norm)
                    st.session_state.transcription_results[m_key] = {
                        "text": text_out,
                        "latency": latency
                    }
                
                progress_bar.progress(100, text="Transcription Complete!")
                time.sleep(0.3)
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

# --- Display Results ---
results = st.session_state.transcription_results
if results:
    st.divider()
    st.subheader("🎯 Transcription Results & Model Comparison")
    
    if len(results) == 2:
        col_med, col_sml = st.columns(2)
        
        # Whisper Medium Column
        with col_med:
            med_res = results.get("medium", {})
            st.markdown(f"""
            <div class="{MODELS_INFO['medium']['card_class']}">
                <span class="{MODELS_INFO['medium']['badge_class']}">{MODELS_INFO['medium']['badge_text']}</span>
                <h3 style="margin-top:0; color:#14532d;">{MODELS_INFO['medium']['title']}</h3>
                <div class="{MODELS_INFO['medium']['output_class']}">{med_res.get('text', '')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area("Medium Output (Copy):", value=med_res.get("text", ""), height=80, key="copy_med")
            w_med = len(med_res.get("text", "").split())
            c_med = len(med_res.get("text", ""))
            st.caption(f"⏱️ Inference Latency: **{med_res.get('latency', 0.0):.2f}s** | Words: **{w_med}** | Chars: **{c_med}**")

        # Whisper Small Column
        with col_sml:
            sml_res = results.get("small", {})
            st.markdown(f"""
            <div class="{MODELS_INFO['small']['card_class']}">
                <span class="{MODELS_INFO['small']['badge_class']}">{MODELS_INFO['small']['badge_text']}</span>
                <h3 style="margin-top:0; color:#1e293b;">{MODELS_INFO['small']['title']}</h3>
                <div class="{MODELS_INFO['small']['output_class']}">{sml_res.get('text', '')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area("Small Output (Copy):", value=sml_res.get("text", ""), height=80, key="copy_sml")
            w_sml = len(sml_res.get("text", "").split())
            c_sml = len(sml_res.get("text", ""))
            st.caption(f"⏱️ Inference Latency: **{sml_res.get('latency', 0.0):.2f}s** | Words: **{w_sml}** | Chars: **{c_sml}**")
        
        # Text Comparison Difference Summary
        med_words = med_res.get("text", "").split()
        sml_words = sml_res.get("text", "").split()
        matcher = difflib.SequenceMatcher(None, sml_words, med_words)
        similarity = matcher.ratio() * 100
        
        st.markdown(f"""
        <div class="diff-box">
            <b>🔍 ملخص المقارنة بين النموذجين:</b><br>
            • نسبة التطابق النصي بين المخرجات: <b>{similarity:.1f}%</b><br>
            • يتميز النموذج <b>Medium (833M)</b> بقدرة فائقة على فهم مخارج الحروف الجزائرية وسرعة الكلام والكلمات المركبة بدقة مضاعفة (WER 0.34% مقابل 14.87%).
        </div>
        """, unsafe_allow_html=True)
        
    else:
        for m_key, m_res in results.items():
            info = MODELS_INFO[m_key]
            st.markdown(f"""
            <div class="{info['card_class']}">
                <span class="{info['badge_class']}">{info['badge_text']}</span>
                <h3 style="margin-top:0;">{info['title']}</h3>
                <div class="{info['output_class']}">{m_res.get('text', '')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area(f"{info['title']} Output (Copy):", value=m_res.get("text", ""), height=90, key=f"copy_{m_key}")
            w_count = len(m_res.get("text", "").split())
            c_count = len(m_res.get("text", ""))
            st.caption(f"⏱️ Latency: **{m_res.get('latency', 0.0):.2f}s** | Duration: **{st.session_state.audio_duration:.2f}s** | Words: **{w_count}** | Chars: **{c_count}**")

# --- Footer ---
st.divider()
st.caption("Developed by **Kamel Touati** | Fine-tuned on OddAdmix Algerian Datasets with PEFT LoRA (QLoRA 4-bit) & Streamlit.")
