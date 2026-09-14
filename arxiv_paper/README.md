# Research Paper: Adapting Foundation Speech Models to Low-Resource Dialects (Algerian Darja)

This folder contains the complete LaTeX source code and bibliography for the scientific research paper ready for submission to **arXiv** and international speech recognition conferences (e.g., Interspeech, ICASSP, ACL/WANLP).

---

## Files Included

- `paper.tex`: Main scientific LaTeX manuscript containing abstract, introduction, linguistic analysis, mathematical formulation of QLoRA streaming curriculum learning, empirical benchmark tables, scaling analysis, qualitative error analysis, and conclusions.
- `references.bib`: BibTeX bibliography file with complete citations.

---

## How to Compile to PDF

### Method 1: Using Overleaf
1. Compress this folder (`arxiv_paper`) into a `.zip` archive.
2. Go to [Overleaf](https://www.overleaf.com) and click **New Project** -> **Upload Project**.
3. Select the `.zip` file and compile with pdfLaTeX.

### Method 2: Local Command Line (if TeX Live / MiKTeX installed)
```bash
cd arxiv_paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## Submitting to arXiv

1. Ensure `paper.tex` and `references.bib` are in the root of your upload archive.
2. If arXiv requires `.bbl`, compile locally and include `paper.bbl`, or submit `paper.tex` and `references.bib` directly in a `.tar.gz` bundle:
```bash
tar -czvf arxiv_submission.tar.gz paper.tex references.bib
```
3. Submit to category: **cs.CL** (Computation and Language) or **eess.AS** (Audio and Speech Processing).
