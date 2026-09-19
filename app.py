# =============================================================================
# Root Entrypoint — Streamlit & Hugging Face Spaces Deployment
# =============================================================================

import os
import sys
import runpy

# Ensure submodules are in python path
root_dir = os.path.dirname(os.path.abspath(__file__))
asr_dir  = os.path.join(root_dir, "asr")
tts_dir  = os.path.join(root_dir, "tts")

if asr_dir not in sys.path:
    sys.path.insert(0, asr_dir)
if tts_dir not in sys.path:
    sys.path.insert(0, tts_dir)

# Execute the primary ASR interactive application (Side-by-side Whisper evaluation)
asr_app_path = os.path.join(asr_dir, "app.py")
runpy.run_path(asr_app_path, run_name="__main__")
