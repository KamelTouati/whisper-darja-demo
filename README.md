# Whisper Algerian Arabic (Darja / الدارجة الجزائرية) — Small & Medium ASR Suite

<p align="center">
  <img src="https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/whisper_architecture.png" alt="Whisper Architecture" width="700"/>
</p>

<p align="center">
  <a href="https://huggingface.co/touati-kamel/whisper-algerian-darja-medium"><img src="https://img.shields.io/badge/HF%20Model-Whisper%20Medium-orange?style=flat-square&logo=huggingface" alt="Whisper Medium Model"></a>
  <a href="https://huggingface.co/touati-kamel/whisper-algerian-darja-small"><img src="https://img.shields.io/badge/HF%20Model-Whisper%20Small-orange?style=flat-square&logo=huggingface" alt="Whisper Small Model"></a>
  <a href="https://github.com/huggingface/peft"><img src="https://img.shields.io/badge/PEFT-LoRA%204--bit%20(QLoRA)-purple?style=flat-square" alt="PEFT LoRA"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://wandb.ai/k_touati-estin/whisper-algerian-darja"><img src="https://img.shields.io/badge/W%26B-Training%20Logs-gold?style=flat-square&logo=weightsandbiases" alt="Weights and Biases"></a>
</p>

---

## 📌 Overview

This repository contains the evaluation benchmarks, inference code, and interactive **Streamlit comparison application** for fine-tuned Automatic Speech Recognition (ASR) models for **Algerian Arabic (Darja / الدارجة الجزائرية)**:

1. **`touati-kamel/whisper-algerian-darja-medium`** (SOTA — **833M Parameters**, 69.2M LoRA)
2. **`touati-kamel/whisper-algerian-darja-small`** (Lightweight — **267M Parameters**, 25.9M LoRA)

Both models were adapted from **OpenAI Whisper** via 4-bit quantized Low-Rank Adaptation (**QLoRA**) across a 3-phase sequential streaming curriculum covering conversational podcasts, spontaneous expressive storytelling, and cultural narratives from the **OddAdmix Algerian speech collection**.

---

## 🏆 Comprehensive Benchmark & Performance Matrix

The models were evaluated iteratively across three distinct Algerian dialect audio domains using Word Error Rate (**WER %**) and Cross-Entropy Loss with standardized Arabic text normalization:

| Domain / Benchmark Split | Dataset Identifier | Whisper Small (267M) WER | Whisper Medium (833M) WER | Error Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **Loubna Expressive Stories** | `oddadmix/arabic-audio-collection-algerian-loubna-stories` | 14.87% | **0.34%** | **-97.7%** |
| **Kahwa Conversational Podcast** | `oddadmix/arabic-audio-collection-algerian-kahwa-postcast` | 34.85% | **0.68%** | **-98.0%** |
| **Rawi Cultural Storytelling** | `oddadmix/arabic-audio-collection-algerian-rawi` | 27.54% | **0.95%** | **-96.5%** |
| **Final Evaluation Loss** | — | 0.2548 | **0.00612** | **-97.6%** |

### 🔍 Model Architecture Specifications

| Specification | Whisper Small (`openai/whisper-small`) | Whisper Medium (`openai/whisper-medium`) |
| :--- | :---: | :---: |
| **Total Model Weights** | 267,687,168 | **833,063,936** |
| **Trainable LoRA Weights** | 25,952,256 (9.70%) | **69,206,016 (8.31%)** |
| **Transformer Layers** | 12 Encoder / 12 Decoder | **24 Encoder / 24 Decoder** |
| **Attention Heads** | 12 heads | **16 heads** |
| **LoRA Target Modules** | `q_proj, k_proj, v_proj, out_proj, fc1, fc2` | `q_proj, k_proj, v_proj, out_proj, fc1, fc2` |
| **LoRA Hyperparameters** | $r=64, \alpha=128, \text{dropout}=0.05$ | $r=64, \alpha=128, \text{dropout}=0.05$ |
| **Quantization** | 4-bit NormalFloat4 (NF4) | 4-bit NormalFloat4 (NF4) |
| **Effective Batch Size** | 32 (8 per device × 4 grad accum) | 32 (4 per device × 8 grad accum) |
| **Cumulative Optimization Steps**| 15,829 steps | **31,661 steps** |

---

## 🚀 Sequential Curriculum Learning Methodology

```
┌────────────────────────────────────────────────────────┐
│               OpenAI Whisper (Small / Medium)          │
│                (4-bit NF4 Quantization Base)           │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │    LoRA Adapters (r=64)   │
             │   Target: q, k, v, out,   │
             │           fc1, fc2        │
             └─────────────┬─────────────┘
                           │
       ┌────────────────────┴────────────────────┐
       │     Sequential Curriculum Learning      │
       ├─────────────────────────────────────────┤
       │ Phase 1: Kahwa Podcast (Conversational) │
       │  • Small: 4,942 steps | Med: 9,886 steps│
       │                    ▼                    │
       │ Phase 2: Loubna Stories (Expressive)    │
       │  • Small: 10,324 steps| Med: 20,650 step│
       │                    ▼                    │
       │ Phase 3: Rawi Narratives (Storytelling) │
       │  • Small: 562 steps   | Med: 1,125 steps│
       └─────────────────────────────────────────┘
```

---

## 🎙️ Interactive Streamlit Comparison Application

The repository includes a web application ([`app.py`](app.py)) that allows users to record or upload a **single audio input** and compare the transcription outputs of **both models side-by-side**.

### Features:
- 🔴 **Live Microphone Recording & Audio File Upload** (WAV, MP3, OGG, M4A, FLAC, WebM).
- ⚡ **Side-by-Side Model Comparison**: Real-time evaluation of Whisper Medium vs. Whisper Small on the identical audio stream.
- 🔍 **Output Diff & Similarity Score**: Computes textual alignment and shows vocabulary/phonetic nuances between models.
- 🧹 **Darja Text Normalization Toggle**: Diacritics (harakat), tatweel (kashida), punctuation, and Alef/Yaa normalization.
- 📋 **Copy to Clipboard & Latency / Duration Metrics**.

### Running the App Locally:

```bash
# 1. Clone the repository
git clone https://github.com/KamelTouati/whisper-darja-demo.git
cd whisper-darja-demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
```

---

## 💻 Python Quickstart & Inference

### 1. Installation

```bash
pip install --upgrade transformers peft torch torchaudio soundfile librosa jiwer bitsandbytes accelerate
```

### 2. High-Level Usage with `transformers.pipeline`

```python
import torch
from transformers import pipeline

# Choose model: "whisper-algerian-darja-medium" (recommended) or "whisper-algerian-darja-small"
model_id = "touati-kamel/whisper-algerian-darja-medium"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else "cpu",
    chunk_length_s=30,
)

# Transcribe an audio file (16kHz WAV or MP3)
result = pipe(
    "path/to/algerian_audio.mp3",
    generate_kwargs={"language": "arabic", "task": "transcribe"}
)

print("Transcription (Algerian Darja):", result["text"])
```

### 3. Native PyTorch + `PeftModel`

```python
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_id = "openai/whisper-medium"  # or "openai/whisper-small"
adapter_id = "touati-kamel/whisper-algerian-darja-medium"  # or "touati-kamel/whisper-algerian-darja-small"

# 1. Load Processor
processor = WhisperProcessor.from_pretrained(base_model_id, language="arabic", task="transcribe")

# 2. Load Base Model and Apply LoRA Adapter
base_model = WhisperForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# 3. Preprocess Audio
audio, sr = librosa.load("path/to/audio.mp3", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
if torch.cuda.is_available():
    input_features = input_features.to("cuda", dtype=torch.float16)

# 4. Generate
forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=225
    )

# 5. Decode
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Algerian Darja Output:", transcription)
```

---

## 🧹 Complete Arabic Normalization Function

```python
import re
import string

_ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670"
_TATWEEL = "\u0640"
_PUNCT_MAP = {ord(c): None for c in string.punctuation + "\u060C\u061B\u061F\u00AB\u00BB"}

def normalize_darja_text(text: str) -> str:
    if not text:
        return ""
    # Strip diacritics & tatweel
    text = text.translate({ord(c): None for c in _ARABIC_DIACRITICS})
    text = text.replace(_TATWEEL, "")
    # Normalize Alef and Yaa
    text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064A")
    # Remove punctuation
    text = text.translate(_PUNCT_MAP)
    # Collapse whitespace
    return " ".join(text.split()).strip()
```

---

## 📚 Datasets & Citations

### Training Datasets
- [oddadmix/arabic-audio-collection-algerian-kahwa-postcast](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-algerian-kahwa-postcast)
- [oddadmix/arabic-audio-collection-algerian-loubna-stories](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-algerian-loubna-stories)
- [oddadmix/arabic-audio-collection-algerian-rawi](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-algerian-rawi)

### BibTeX Citations

```bibtex
@misc{touati2026whisper_darja_medium,
  author = {Kamel Touati},
  title = {Whisper Medium Fine-Tuned for Algerian Arabic (Darja)},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/touati-kamel/whisper-algerian-darja-medium}}
}
```

```bibtex
@misc{touati2026whisper_darja_small,
  author = {Kamel Touati},
  title = {Whisper Small Fine-Tuned for Algerian Arabic (Darja)},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/touati-kamel/whisper-algerian-darja-small}}
}
```

```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```
