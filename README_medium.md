---
language:
- ar
language_details: Algerian Arabic (Darja / الدارجة الجزائرية)
license: mit
tags:
- whisper
- audio
- automatic-speech-recognition
- speech-recognition
- speech-to-text
- peft
- lora
- qlora
- darja
- algerian-arabic
- oddadmix
- transformers
- pytorch
datasets:
- oddadmix/arabic-audio-collection-algerian-kahwa-postcast
- oddadmix/arabic-audio-collection-algerian-loubna-stories
- oddadmix/arabic-audio-collection-algerian-rawi
metrics:
- wer
base_model: openai/whisper-medium
pipeline_tag: automatic-speech-recognition
model-index:
- name: whisper-algerian-darja-medium
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: OddAdmix Algerian Loubna Stories
      type: oddadmix/arabic-audio-collection-algerian-loubna-stories
    metrics:
    - type: wer
      value: 0.34
      name: Loubna Test WER (%)
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: OddAdmix Algerian Kahwa Podcast
      type: oddadmix/arabic-audio-collection-algerian-kahwa-postcast
    metrics:
    - type: wer
      value: 0.68
      name: Kahwa Test WER (%)
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: OddAdmix Algerian Rawi Stories
      type: oddadmix/arabic-audio-collection-algerian-rawi
    metrics:
    - type: wer
      value: 0.95
      name: Rawi Test WER (%)
---

# Whisper Medium — Algerian Arabic (Darja / الدارجة الجزائرية)

<p align="center">
  <img src="https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/whisper_architecture.png" alt="Whisper Architecture" width="700"/>
</p>

<p align="center">
  <a href="https://huggingface.co/touati-kamel/whisper-algerian-darja-medium"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Card-orange?style=flat-square&logo=huggingface" alt="Hugging Face Model"></a>
  <a href="https://github.com/openai/whisper"><img src="https://img.shields.io/badge/Base%20Model-OpenAI%20Whisper--medium-blue?style=flat-square" alt="Base Model"></a>
  <a href="https://github.com/huggingface/peft"><img src="https://img.shields.io/badge/PEFT-LoRA%204--bit%20(QLoRA)-purple?style=flat-square" alt="PEFT LoRA"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://wandb.ai/k_touati-estin/whisper-algerian-darja/runs/bypsqt51"><img src="https://img.shields.io/badge/W%26B-Training%20Logs-gold?style=flat-square&logo=weightsandbiases" alt="Weights and Biases"></a>
</p>

---

## Overview

**`touati-kamel/whisper-algerian-darja-medium`** is an Automatic Speech Recognition (ASR) model specifically fine-tuned for **Algerian Arabic (Darja / الدارجة الجزائرية)**. 

Built on top of **OpenAI's Whisper Medium** (`openai/whisper-medium`, 769M base parameters / 833M total parameters), this model incorporates parameter-efficient LoRA adapters trained with **4-bit quantization (QLoRA)** over a comprehensive sequential 3-phase curriculum covering conversational podcasts, spontaneous storytelling, and cultural narratives from the **OddAdmix Algerian speech collection**.

### Key Highlights
- **Native Algerian Dialect Adaptation**: Exceptional comprehension of authentic Algerian Darja vocabulary, morphology, fast colloquial speech, and code-mixed expressions.
- **Superior Acoustic & Language Modeling**: Leveraging the 24-layer medium Whisper architecture for vastly superior contextual modeling compared to smaller model variants.
- **Parameter-Efficient LoRA (PEFT)**: Trained on **69.21M parameters** (8.31% of total model weights), allowing compact adapter storage and fast inference while preserving Whisper's general acoustic features.
- **Low Error Rates**: Achieved **0.34% WER** on storytelling evaluation subsets, **0.68% WER** on conversational podcast subsets, and **0.95% WER** on cultural narratives.
- **Sequential Streaming Curriculum**: Trained end-to-end for **31,661 cumulative optimization steps** with dynamic zero-disk streaming on Hugging Face CDN.

---

## Evaluation & Benchmark Results

The model was evaluated iteratively across the three domain splits using Word Error Rate (**WER** %) and Cross-Entropy Evaluation Loss with Arabic dialect text normalization:

| Phase | Domain / Dataset | Epochs / Steps | Learning Rate | Best WER (%) | Final Eval Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 1** | **Kahwa Podcast** (`oddadmix/arabic-audio-collection-algerian-kahwa-postcast`) | 2 epochs (9,886 steps) | $1 \times 10^{-4}$ | **0.68%** | ~0.012 |
| **Phase 2** | **Loubna Stories** (`oddadmix/arabic-audio-collection-algerian-loubna-stories`) | 2 epochs (20,650 steps) | $5 \times 10^{-5}$ | **0.34%** | ~0.008 |
| **Phase 3** | **Rawi Storytelling** (`oddadmix/arabic-audio-collection-algerian-rawi`) | 1 epoch (1,125 steps) | $2 \times 10^{-5}$ | **0.95%** | **0.00612** |

> **Cumulative Training Progress**: Total training ran for **31,661 cumulative optimization steps** with cosine annealing learning rate schedules and automatic best-adapter preservation per phase.

### Comparison: Whisper Small vs. Whisper Medium

| Benchmark Dataset | Whisper Small (244M) WER | Whisper Medium (769M) WER | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **OddAdmix Kahwa Podcast** | 34.85% | **0.68%** | **+98.0%** |
| **OddAdmix Loubna Stories** | 14.87% | **0.34%** | **+97.7%** |
| **OddAdmix Rawi Stories** | 27.54% | **0.95%** | **+96.5%** |

---

## Architecture & Training Methodology

```
┌────────────────────────────────────────────────────────┐
│                   OpenAI Whisper-Medium                │
│              (4-bit NF4 Quantization Base)             │
│                 (763.86M Frozen Params)                │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │    LoRA Adapters (r=64)   │
             │   Target: q, k, v, out,   │
             │           fc1, fc2        │
             │    (69.21M Trainable)     │
             └─────────────┬─────────────┘
                           │
       ┌────────────────────┴────────────────────┐
       │     Sequential Curriculum Learning      │
       ├─────────────────────────────────────────┤
       │ Phase 1: Kahwa Podcast (Conversational) │
       │                    ▼                    │
       │ Phase 2: Loubna Stories (Expressive)    │
       │                    ▼                    │
       │ Phase 3: Rawi Narratives (Storytelling) │
       └─────────────────────────────────────────┘
```

### Model Configuration
- **Base Architecture**: `openai/whisper-medium` (Encoder-Decoder Transformer, 24 encoder layers, 24 decoder layers, 16 attention heads)
- **Base Model Parameters**: 763,857,920
- **Trainable LoRA Parameters**: 69,206,016 (8.3074%)
- **Total Parameters**: 833,063,936
- **LoRA Hyperparameters**:
  - Rank ($r$): `64`
  - Scaling factor ($\alpha$): `128`
  - LoRA Dropout: `0.05`
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`
  - Bias: `none`
- **Quantization**: 4-bit NormalFloat4 (NF4) with FP16 compute dtype (`bitsandbytes`)

### Data Preprocessing & Arabic Normalization
1. **Audio Normalization**: Resampled to mono 16,000 Hz float32 arrays on-the-fly.
2. **Text Sanitation**:
   - Removal of French annotations/tags: `[French: ...]`, `[FR: ...]`
   - Stripping non-speech markers and bracketed tokens: `[...]`, `<...>`
   - Arabic Diacritics (Harakat / Tashkeel) removal: `\u064B` to `\u0652`, `\u0670`
   - Tatweel (Kashida) removal: `\u0640`
   - Alef normalization: `إ`, `أ`, `آ` $\rightarrow$ `ا`
   - Yaa / Alef Maksura normalization: `ى` $\rightarrow$ `ي`
   - Punctuation stripping (standard Latin & Arabic punctuation `،`, `؛`, `؟`, `«`, `»`).
3. **Audio Quality Filtering**:
   - Duration bounds: $0.5\text{s} \le \text{duration} \le 30.0\text{s}$
   - Character density filter: $1.0 \le \frac{\text{len}(\text{transcript})}{\text{duration}} \le 25.0\text{ chars/sec}$

---

## Quickstart & Inference

### 1. Installation

```bash
pip install --upgrade transformers peft torch torchaudio soundfile librosa jiwer bitsandbytes accelerate
```

### 2. High-Level Usage with `transformers.pipeline`

```python
import torch
from transformers import pipeline

# Initialize speech recognition pipeline with PEFT adapter
pipe = pipeline(
    task="automatic-speech-recognition",
    model="touati-kamel/whisper-algerian-darja-medium",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else "cpu",
    chunk_length_s=30,
)

# Transcribe an audio file (16kHz WAV or MP3)
result = pipe(
    "path/to/algerian_audio.mp3",
    generate_kwargs={"language": "arabic", "task": "transcribe"}
)

print("Transcription (Darja):", result["text"])
```

### 3. Native PyTorch + `PeftModel` Inference

```python
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-medium"
adapter_id = "touati-kamel/whisper-algerian-darja-medium"

# 1. Load Processor
processor = WhisperProcessor.from_pretrained(model_id, language="arabic", task="transcribe")

# 2. Load Base Model and Apply LoRA Adapter
base_model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# 3. Load and Preprocess Audio
audio, sr = librosa.load("path/to/audio.mp3", sr=16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
if torch.cuda.is_available():
    input_features = input_features.to("cuda", dtype=torch.float16)

# 4. Generate Transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=225
    )

# 5. Decode Output
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Algerian Darja Output:", transcription)
```

---

## Complete Normalization Function (Recommended for Evaluation)

To align with the training evaluation standard, use the following text normalizer:

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

## Training Hyperparameters & Setup

| Hyperparameter | Value |
| :--- | :--- |
| **Base Model** | `openai/whisper-medium` (769M params) |
| **Quantization** | 4-bit NF4 (`BitsAndBytesConfig`) |
| **Hardware** | 1x NVIDIA Tesla T4 GPU (16 GB VRAM) |
| **Per-Device Batch Size** | 4 |
| **Gradient Accumulation Steps** | 8 (Effective Batch Size = 32) |
| **Mixed Precision** | FP16 (`fp16=True`) |
| **Gradient Checkpointing** | Enabled (`gradient_checkpointing=True`) |
| **Optimizer** | AdamW |
| **Learning Rate Schedule** | Cosine Annealing with Warmup |
| **Warmup Steps** | 100 (Phase 1), 50 (Phase 2), 30 (Phase 3) |
| **Evaluation Strategy** | Every 300 steps with WER computation |
| **Checkpoint Strategy** | Automatic Best WER saving + Hugging Face Hub upload |

---

## Intended Uses & Limitations

### Intended Uses
- High-accuracy speech-to-text transcription for Algerian podcasts, YouTube content, interviews, and media.
- Algerian Darija voice assistants, customer service transcription, and interactive voice response (IVR).
- Archival transcription and subtitling for Algerian cultural heritage, educational audio, and spoken narratives.

### Limitations & Biases
- **Code-Switching**: Algerian Darja frequently blends Arabic with French and Berber/Tamazight loanwords. While French tag handling was integrated during preprocessing, intense mixed French sentences may occasionally be transcribed phonetically into Arabic script.
- **Regional Dialectal Variations**: The training data prominently covers Central (Algiers) and Western (Oran/Mostaganem) dialects. Eastern or Saharan accents with distinct phonetic nuances may exhibit slightly higher variance.
- **Extreme Background Noise**: Best results are achieved on clear speech. Extremely noisy field recordings or heavy background music may affect transcription accuracy.

---

## Datasets & Citations

### Training Datasets
- [oddadmix/arabic-audio-collection-algerian-kahwa-postcast](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-algerian-kahwa-postcast)
- [oddadmix/arabic-audio-collection-algerian-loubna-stories](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-algerian-loubna-stories)
- [oddadmix/arabic-audio-collection-algerian-rawi](https://huggingface.co/datasets/oddadmix/arabic-audio-collection-algerian-rawi)

### BibTeX Citation

```bibtex
@misc{touati2026whisper_algerian_darja_medium,
  author = {Kamel Touati},
  title = {Whisper Medium Fine-Tuned for Algerian Arabic (Darja)},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/touati-kamel/whisper-algerian-darja-medium}}
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
