# Algerian Arabic (Darja) Speech Processing & Research Suite

An end-to-end speech recognition and scientific research ecosystem tailored for **Algerian Arabic (Darja / الدارجة الجزائرية)**, encompassing Automatic Speech Recognition (ASR) foundation model adaptation and camera-ready academic research.

---

## Project Structure

```text
whisper-darja-demo/
│
├── asr/                           # Automatic Speech Recognition (Whisper Small & Medium)
│   ├── app.py                     # Interactive Side-by-side Streamlit Demo (Small vs. Medium)
│   ├── app_medium.py              # Standalone Whisper Medium inference UI
│   ├── train_sequential.py        # Sequential Streaming QLoRA training script (Kaggle)
│   ├── modal_train.py             # Cloud training script on Modal (A10G GPU)
│   ├── requirements.txt           # Python dependencies for ASR
│   ├── packages.txt               # System dependencies (ffmpeg)
│   ├── README.md                  # Whisper Small model documentation
│   └── README_medium.md           # Whisper Medium SOTA (0.34% WER) model documentation
│
├── paper/                         # Scientific Research Publication (arXiv cs.CL)
│   ├── main.tex                   # XeLaTeX camera-ready paper (XeTeX 2025 compliant)
│   ├── references.bib             # 21 verified academic citations
│   ├── figures_data/              # Loss and WER trajectory logs across curriculum phases
│   ├── generate_figures.py        # High-resolution vector plot generator
│   ├── fig_loss_trajectory.pdf    # Training loss trajectory
│   ├── fig_wer_convergence.pdf    # WER convergence graph
│   └── arxiv_submission.zip       # Pre-packaged submission archive for arXiv
│
├── app.py                         # Root entrypoint for Streamlit / Hugging Face Spaces deployment
├── requirements.txt               # Unified project dependencies
└── .gitignore                     # Git ignore rules for checkpoints and caches
```

---

## 1. Automatic Speech Recognition (ASR)

Adapted multilingual Whisper foundation models to low-resource Algerian Darja using a zero-disk sequential streaming curriculum across conversational podcasts (`kahwa`), expressive storytelling (`loubna`), and cultural narratives (`rawi`).

- **Whisper Medium (SOTA)**: `touati-kamel/whisper-algerian-darja-medium` (0.34% WER on stories, 0.68% on podcasts)
- **Whisper Small**: `touati-kamel/whisper-algerian-darja-small` (8.88% WER baseline)

### Running the ASR Demo Locally
```bash
pip install -r asr/requirements.txt
streamlit run app.py
```

---

## 2. Scientific Paper (arXiv Preprint)

- **Title**: *Adapting Foundation Speech Models to Low-Resource Dialects: A Sequential Streaming QLoRA Approach for Algerian Arabic (Darja)*
- **Author**: Kamel Touati (*Independent AI Researcher, Algiers, Algeria*)
- **Primary Category**: `cs.CL` (Computation and Language)
- **Cross-Lists**: `eess.AS`, `cs.AI`, `cs.LG`
- **XeLaTeX Source**: Located in [`paper/main.tex`](paper/main.tex).

---

## Public Model Links

- **Whisper Medium (SOTA)**: [touati-kamel/whisper-algerian-darja-medium](https://huggingface.co/touati-kamel/whisper-algerian-darja-medium)
- **Whisper Small**: [touati-kamel/whisper-algerian-darja-small](https://huggingface.co/touati-kamel/whisper-algerian-darja-small)
- **GitHub Repository**: [KamelTouati/whisper-darja-demo](https://github.com/KamelTouati/whisper-darja-demo)
