# =============================================================================
# Fine-tuning Whisper Medium — Darja Algérienne — Modal Cloud (A10G GPU)
# =============================================================================

import os
import modal

# 1. Définition de l'application Modal
app = modal.App("whisper-medium-darja-training")

# 2. Image Docker avec toutes les dépendances audio & deep learning
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "numpy<2.1",
        "scipy",
        "torch",
        "torchaudio",
        "transformers>=4.40.0",
        "datasets<3.0.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.10.0",
        "soundfile",
        "librosa",
        "jiwer",
        "wandb",
        "huggingface_hub",
    )
)

# 3. Fonction distante exécutée sur GPU A10G
@app.function(
    image=training_image,
    gpu="A10G",          # GPU A10G (24 GB VRAM)
    cpu=4,               # 4 vCPUs pour le décodage audio
    memory=16384,        # 16 GiB RAM
    timeout=86400,       # 24h max (s'arrête automatiquement dès la fin)
    secrets=[modal.Secret.from_name("whisper-secrets")],
)
def run_training():
    import os
    import subprocess
    import gc
    import re
    import shutil
    import json
    import string
    import time as _time
    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional

    import torch
    import wandb
    import numpy as np
    import tempfile

    # ── GPU & Variables d'environnement ───────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_DISABLE_XET"]   = "1"

    BASE_TMP = "/tmp"
    HF_CACHE_DIR = os.path.join(BASE_TMP, "hf_home")
    SCRATCH_TMP_DIR = os.path.join(BASE_TMP, "scratch")

    for _d in [HF_CACHE_DIR, SCRATCH_TMP_DIR]:
        os.makedirs(_d, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TMPDIR"]  = SCRATCH_TMP_DIR
    tempfile.tempdir = SCRATCH_TMP_DIR

    from datasets import load_dataset, load_dataset_builder, Audio
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import BitsAndBytesConfig
    from huggingface_hub import HfApi, snapshot_download, login as hf_login
    from transformers.trainer_utils import get_last_checkpoint
    import jiwer

    # =========================================================================
    # ── SECTION 1 : Secrets & Login ──────────────────────────────────────────
    # =========================================================================
    HF_TOKEN = (
        os.environ.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    WANDB_API_KEY = (
        os.environ.get("wandb_api_key")
        or os.environ.get("WANDB_API_KEY")
        or os.environ.get("WANDB_KEY")
    )

    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN non trouvé dans le secret Modal 'whisper-secrets' !")

    hf_login(token=HF_TOKEN)
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    else:
        print("⚠️ WANDB_API_KEY non trouvé, poursuite sans login explicite W&B.")

    # =========================================================================
    # ── SECTION 2 : Configuration ────────────────────────────────────────────
    # =========================================================================
    MODEL_SIZE    = "medium"
    HF_USERNAME   = "touati-kamel"
    HF_REPO_ID    = f"{HF_USERNAME}/whisper-algerian-darja-{MODEL_SIZE}"
    OUTPUT_DIR    = f"./whisper_algerian_darja_{MODEL_SIZE}"
    WANDB_PROJECT = "whisper-algerian-darja"

    MODEL_CONFIGS = {
        "small":    {"base_model_id": "openai/whisper-small",    "batch_size": 8, "grad_accum": 4},
        "medium":   {"base_model_id": "openai/whisper-medium",   "batch_size": 4, "grad_accum": 8},
        "large-v3": {"base_model_id": "openai/whisper-large-v3", "batch_size": 2, "grad_accum": 16},
    }
    cfg = MODEL_CONFIGS[MODEL_SIZE]
    DEFAULT_STEPS_FALLBACK = 4000

    PHASES = [
        {
            "name":       "kahwa",
            "dataset_id": "oddadmix/arabic-audio-collection-algerian-kahwa-postcast",
            "prefix":     "kahwa",
            "epochs":     2,
            "lr":         1e-4,
            "warmup":     100,
        },
        {
            "name":       "loubna",
            "dataset_id": "oddadmix/arabic-audio-collection-algerian-loubna-stories",
            "prefix":     "loubna",
            "epochs":     2,
            "lr":         5e-5,
            "warmup":     50,
        },
        {
            "name":       "rawi",
            "dataset_id": "oddadmix/arabic-audio-collection-algerian-rawi",
            "prefix":     "rawi",
            "epochs":     1,
            "lr":         2e-5,
            "warmup":     30,
        },
    ]

    EVAL_SUBSET_SIZE     = 300
    PHASE_STATE_FILENAME = "phase_state.json"
    ADAPTER_HUB_PATH     = "last-adapter"

    api = HfApi()
    print(f"🎯 whisper-{MODEL_SIZE} | {HF_REPO_ID}")

    # =========================================================================
    # ── SECTION 3 : Utilitaires & Normalisation ──────────────────────────────
    # =========================================================================
    def print_disk_usage(label: str = "") -> None:
        out = subprocess.run(["df", "-h", "/", BASE_TMP], capture_output=True, text=True).stdout
        print(f"💾 Disque {label}:\n{out}")

    def dir_size_gb(path: str) -> float:
        if not os.path.isdir(path):
            return 0.0
        total = 0
        for dp, _, fns in os.walk(path):
            for f in fns:
                try:
                    total += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        return total / 1e9

    def get_dataset_size(dataset_id: str) -> Optional[int]:
        try:
            builder = load_dataset_builder(dataset_id)
            n = builder.info.splits["train"].num_examples
            print(f"   📊 Taille connue : {n:,} exemples")
            return n
        except Exception as e:
            print(f"   ⚠️ Taille inconnue ({e}), fallback : {DEFAULT_STEPS_FALLBACK} steps")
            return None

    def compute_max_steps(n_examples: Optional[int], batch_size: int, grad_accum: int, n_epochs: int) -> int:
        if n_examples is None:
            return DEFAULT_STEPS_FALLBACK
        effective_n = int(n_examples * 0.85)
        steps_per_epoch = max(1, effective_n // batch_size)
        total = steps_per_epoch * n_epochs
        print(f"   🔢 max_steps calculé : {effective_n} × {n_epochs} epochs / batch {batch_size} = {total} steps")
        return total

    def download_hub_checkpoint(phase_out: str, phase_name: str) -> Optional[str]:
        hub_folder = f"last-checkpoint-{phase_name}"
        local_ckpt_dir = os.path.join(phase_out, "hub_checkpoint")
        actual_ckpt = os.path.join(local_ckpt_dir, hub_folder)
        try:
            print(f"🔄 Tentative téléchargement checkpoint depuis HF Hub ({hub_folder})...")
            snapshot_download(
                repo_id=HF_REPO_ID,
                token=HF_TOKEN,
                local_dir=local_ckpt_dir,
                allow_patterns=[f"{hub_folder}/**"],
            )
            has_weights = (
                os.path.isdir(actual_ckpt)
                and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(actual_ckpt))
            )
            if has_weights:
                state_f = os.path.join(actual_ckpt, "trainer_state.json")
                if os.path.isfile(state_f):
                    with open(state_f) as _f:
                        _gs = json.load(_f).get("global_step", "?")
                    print(f"✅ Checkpoint Hub récupéré ({hub_folder}) — global_step={_gs} → {actual_ckpt}")
                else:
                    print(f"✅ Checkpoint Hub récupéré ({hub_folder}) → {actual_ckpt}")
                return actual_ckpt
            return None
        except Exception as e:
            print(f"ℹ️ Aucun checkpoint HF Hub pour {phase_name} ({e})")
            return None

    # ── Normalisation texte arabe ────────────────────────────────────────────
    _FR_TAG_RE  = re.compile(r"\[\s*(?:French|FR)\s*:\s*(.*?)\]", re.IGNORECASE)
    _BRACKET_RE = re.compile(r"\[.*?\]|<.*?>")
    _ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670"
    _TATWEEL           = "\u0640"
    _PUNCT_MAP = {ord(c): None for c in string.punctuation + "\u060C\u061B\u061F\u00AB\u00BB"}

    def normalize_arabic(text: str) -> str:
        if not text:
            return ""
        text = text.translate({ord(c): None for c in _ARABIC_DIACRITICS})
        text = text.replace(_TATWEEL, "")
        text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
        text = text.replace("\u0649", "\u064A")
        text = text.translate(_PUNCT_MAP)
        return " ".join(text.split()).strip()

    def preprocess_transcript(batch: dict) -> dict:
        text = batch.get("transcript_text") or ""
        text = _FR_TAG_RE.sub(r"\1", text)
        text = _BRACKET_RE.sub("", text)
        text = normalize_arabic(text)
        batch["transcript_text"] = text
        return batch

    def is_valid(example: dict) -> bool:
        duration = example.get("duration", 15)
        if not (0.5 <= duration <= 30.0):
            return False
        text = (example.get("transcript_text") or "").strip()
        if not text:
            return False
        return 1.0 <= len(text) / duration <= 25.0

    # ── DataCollator ─────────────────────────────────────────────────────────
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_features = [
                {
                    "input_features": self.processor.feature_extractor(
                        f["audio"]["array"],
                        sampling_rate=f["audio"]["sampling_rate"],
                    ).input_features[0]
                }
                for f in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_ids_list = [
                self.processor.tokenizer(
                    f["transcript_text"], truncation=True, max_length=448
                ).input_ids
                for f in features
            ]
            labels_batch = self.processor.tokenizer.pad(
                [{"input_ids": l} for l in label_ids_list], return_tensors="pt"
            )
            batch["labels"] = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask == 0, -100
            )
            return batch

    # ── Callbacks ────────────────────────────────────────────────────────────
    class HubUploadCallback(TrainerCallback):
        def __init__(self, api: HfApi, repo_id: str, token: str, phase_name: str) -> None:
            self.api = api
            self.repo_id = repo_id
            self.token = token
            self.phase_name = phase_name
            self.hub_folder = f"last-checkpoint-{phase_name}"

        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            last_ckpt = get_last_checkpoint(args.output_dir)
            if not last_ckpt:
                return control
            print(f"\n☁️ Upload checkpoint step {state.global_step} → HF Hub...")
            try:
                self.api.upload_folder(
                    folder_path=last_ckpt,
                    repo_id=self.repo_id,
                    path_in_repo=self.hub_folder,
                    token=self.token,
                    commit_message=f"checkpoint step {state.global_step} (phase {self.phase_name})",
                )
                print("✅ Upload OK.")
            except Exception as e:
                print(f"⚠️ Upload échoué (non fatal) : {e}")
            return control

    class DiskMonitorCallback(TrainerCallback):
        WATCHED = {"hf_home": HF_CACHE_DIR, "scratch": SCRATCH_TMP_DIR, "system_tmp": BASE_TMP}
        def __init__(self, output_dir: str) -> None:
            self.output_dir = output_dir

        def on_log(self, args, state, control, **kwargs):
            payload = {f"disk_gb/{k}": dir_size_gb(v) for k, v in self.WATCHED.items()}
            payload["disk_gb/output_dir"] = dir_size_gb(self.output_dir)
            try:
                payload["disk_gb/root_free"] = shutil.disk_usage("/").free / 1e9
            except OSError:
                pass
            wandb.log(payload, step=state.global_step)
            return control

    class BestWERSaveCallback(TrainerCallback):
        def __init__(self, model, save_dir: str) -> None:
            self.model = model
            self.save_dir = save_dir
            self.best_wer = float("inf")

        def on_evaluate(self, args, state, control, metrics: Dict[str, float] = None, **kwargs):
            wer = metrics.get("eval_wer", float("inf")) if metrics else float("inf")
            if wer < self.best_wer:
                self.best_wer = wer
                best_path = os.path.join(self.save_dir, "best_adapter")
                self.model.save_pretrained(best_path)
                print(f"🏆 Meilleur WER = {wer:.2f}% — adapter sauvegardé dans {best_path}")
            return control

        def reload_best(self):
            best_path = os.path.join(self.save_dir, "best_adapter")
            if os.path.isdir(best_path):
                print(f"🔄 Rechargement du meilleur adapter ({self.best_wer:.2f}% WER)...")
                try:
                    self.model.load_adapter(best_path, adapter_name="default")
                except Exception as e:
                    print(f"⚠️ Rechargement adapter ignoré ({e}) — modèle garde les poids finaux.")

    class TempCleanupCallback(TrainerCallback):
        def __init__(self, tmp_dir: str = SCRATCH_TMP_DIR, min_age: int = 300) -> None:
            self.tmp_dir = tmp_dir
            self.min_age = min_age

        def on_save(self, args, state, control, **kwargs):
            if not os.path.isdir(self.tmp_dir):
                return control
            now, freed = _time.time(), 0
            for dp, _, fns in os.walk(self.tmp_dir):
                for f in fns:
                    fp = os.path.join(dp, f)
                    try:
                        if now - os.path.getmtime(fp) > self.min_age:
                            freed += os.path.getsize(fp)
                            os.remove(fp)
                    except OSError:
                        pass
            if freed:
                print(f"🧹 Scratch nettoyé : {freed/1e9:.3f} GB")
            return control

    # ── Métrique WER ─────────────────────────────────────────────────────────
    def make_compute_metrics(proc: WhisperProcessor):
        def compute_metrics(pred):
            pred_ids  = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = proc.tokenizer.pad_token_id
            pred_str  = proc.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = proc.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            pred_str  = [normalize_arabic(p) for p in pred_str]
            pairs     = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
            if not pairs:
                return {"wer": 0.0}
            preds, refs = zip(*pairs)
            return {"wer": 100 * jiwer.wer(list(refs), list(preds))}
        return compute_metrics

    # =========================================================================
    # ── SECTION 4 : Phase State ──────────────────────────────────────────────
    # =========================================================================
    def load_phase_state() -> dict:
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                token=HF_TOKEN,
                allow_patterns=PHASE_STATE_FILENAME,
                local_dir=OUTPUT_DIR,
            )
            full_path = os.path.join(OUTPUT_DIR, PHASE_STATE_FILENAME)
            if os.path.isfile(full_path):
                with open(full_path, "r") as f:
                    state = json.load(f)
                print(f"🔄 Phase state récupéré : {state}")
                return state
        except Exception:
            pass
        return {"completed": [], "global_step_offset": 0}

    def save_phase_state(state: dict) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        local_path = os.path.join(OUTPUT_DIR, PHASE_STATE_FILENAME)
        with open(local_path, "w") as f:
            json.dump(state, f, indent=2)
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=PHASE_STATE_FILENAME,
                repo_id=HF_REPO_ID,
                token=HF_TOKEN,
                commit_message=f"update phase_state.json: completed={state['completed']}",
            )
        except Exception as e:
            print(f"⚠️ Impossible d'uploader phase_state.json : {e}")

    # =========================================================================
    # ── SECTION 5 : Init Modèle & Processeur ─────────────────────────────────
    # =========================================================================
    print_disk_usage("(démarrage)")
    print(f"\n🔍 GPU : {torch.cuda.device_count()} device(s)")
    for _i in range(torch.cuda.device_count()):
        _f, _t = torch.cuda.mem_get_info(_i)
        print(f"   GPU {_i} — {torch.cuda.get_device_name(_i)} — {_f/1e9:.1f} GB libres / {_t/1e9:.1f} GB")

    if not api.repo_exists(HF_REPO_ID, token=HF_TOKEN):
        api.create_repo(HF_REPO_ID, token=HF_TOKEN, exist_ok=True)

    phase_state        = load_phase_state()
    completed_phases   = set(phase_state.get("completed", []))
    global_step_offset = phase_state.get("global_step_offset", 0)

    processor = WhisperProcessor.from_pretrained(
        cfg["base_model_id"], language="arabic", task="transcribe"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["base_model_id"],
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=False,
    )
    model = prepare_model_for_kbit_training(model)

    if completed_phases:
        print(f"🔄 Phases terminées : {completed_phases} — rechargement adapter...")
        local_adapter = os.path.join(OUTPUT_DIR, ADAPTER_HUB_PATH)
        snapshot_download(
            repo_id=HF_REPO_ID, token=HF_TOKEN,
            local_dir=OUTPUT_DIR,
            allow_patterns=[f"{ADAPTER_HUB_PATH}/*"],
        )
        model = PeftModel.from_pretrained(model, local_adapter, is_trainable=True)
        print(f"✅ Adapter rechargé depuis {local_adapter}")
    else:
        peft_config = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05, bias="none",
        )
        model = get_peft_model(model, peft_config)
        print("✅ Nouvel adapter LoRA initialisé (r=64, alpha=128).")

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="arabic", task="transcribe"
    )
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False
    model.print_trainable_parameters()

    data_collator   = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    # =========================================================================
    # ── SECTION 6 : Boucle d'entraînement Séquentielle Streaming ─────────────
    # =========================================================================
    wandb.init(
        project=WANDB_PROJECT,
        name=f"whisper-{MODEL_SIZE}-darja--streaming",
        resume="allow",
    )

    for phase in PHASES:
        phase_name = phase["name"]

        if phase_name in completed_phases:
            print(f"\n⏭️ Phase '{phase_name}' déjà terminée — skip.")
            continue

        phase_idx = PHASES.index(phase) + 1
        print(f"\n{'='*70}")
        print(f"🚀 PHASE {phase_idx}/{len(PHASES)} : {phase_name.upper()}")
        print(f"   Dataset : {phase['dataset_id']}")
        print(f"   Epochs  : {phase['epochs']} | LR : {phase['lr']}")
        print(f"{'='*70}")
        print_disk_usage(f"(début phase {phase_name})")

        print("🔍 Récupération taille dataset...")
        n_examples = get_dataset_size(phase["dataset_id"])
        max_steps  = compute_max_steps(
            n_examples,
            batch_size=cfg["batch_size"],
            grad_accum=cfg["grad_accum"],
            n_epochs=phase["epochs"],
        )

        print(f"\n📥 Création du stream : {phase['dataset_id']}")
        _pfx = phase["prefix"]

        def _make_stream(dataset_id=phase["dataset_id"], pfx=_pfx):
            ds = load_dataset(dataset_id, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            ds = ds.map(lambda x: {
                **x,
                "original_video_id": f"{pfx}_{x.get('original_video_id', '')}",
            })
            ds = ds.map(preprocess_transcript)
            ds = ds.filter(is_valid)
            return ds

        eval_ds  = _make_stream().take(EVAL_SUBSET_SIZE)
        train_ds = _make_stream().shuffle(seed=42, buffer_size=2000)

        print(f"✅ Streams prêts — {EVAL_SUBSET_SIZE} exemples eval | train ({max_steps} steps)")

        phase_out = os.path.join(OUTPUT_DIR, f"phase_{phase_name}")
        os.makedirs(phase_out, exist_ok=True)

        resume_ckpt = get_last_checkpoint(phase_out)
        if resume_ckpt:
            print(f"🔄 Reprise locale depuis : {resume_ckpt}")
        else:
            resume_ckpt = download_hub_checkpoint(phase_out, phase_name)
            if not resume_ckpt:
                print("🆕 Démarrage fresh pour la phase.")

        best_saver = BestWERSaveCallback(model=model, save_dir=phase_out)

        training_args = Seq2SeqTrainingArguments(
            output_dir=phase_out,
            remove_unused_columns=False,
            label_names=["labels"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            gradient_checkpointing=True,
            learning_rate=phase["lr"],
            lr_scheduler_type="cosine",
            warmup_steps=phase["warmup"],
            max_steps=max_steps,
            fp16=True,
            eval_strategy="steps",
            eval_steps=300,
            save_strategy="steps",
            save_steps=300,
            logging_steps=20,
            save_total_limit=1,
            predict_with_generate=True,
            generation_max_length=225,
            load_best_model_at_end=False,
            report_to=["wandb"],
            run_name=f"whisper-{MODEL_SIZE}-{phase_name}",
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor,
            callbacks=[
                HubUploadCallback(api=api, repo_id=HF_REPO_ID, token=HF_TOKEN, phase_name=phase_name),
                DiskMonitorCallback(output_dir=phase_out),
                TempCleanupCallback(tmp_dir=SCRATCH_TMP_DIR),
                best_saver,
            ],
        )

        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n🏃 Entraînement streaming — phase {phase_name} ({max_steps} steps)...")
        trainer.train(resume_from_checkpoint=resume_ckpt)

        best_saver.reload_best()

        print(f"\n💾 Sauvegarde adapter LoRA — phase {phase_name}...")
        adapter_local = os.path.join(OUTPUT_DIR, ADAPTER_HUB_PATH)
        os.makedirs(adapter_local, exist_ok=True)
        model.save_pretrained(adapter_local)

        try:
            api.upload_folder(
                folder_path=adapter_local,
                repo_id=HF_REPO_ID,
                path_in_repo=ADAPTER_HUB_PATH,
                token=HF_TOKEN,
                commit_message=f"Adapter LoRA après phase {phase_name}",
            )
            print(f"✅ Adapter poussé → {HF_REPO_ID}/{ADAPTER_HUB_PATH}")
        except Exception as e:
            print(f"⚠️ Upload adapter échoué : {e}")

        global_step_offset += trainer.state.global_step
        completed_phases.add(phase_name)
        phase_state["completed"]          = list(completed_phases)
        phase_state["global_step_offset"] = global_step_offset
        save_phase_state(phase_state)

        del trainer, train_ds, eval_ds
        gc.collect()
        torch.cuda.empty_cache()
        print_disk_usage(f"(fin phase {phase_name})")

        print(f"\n✅ Phase '{phase_name}' terminée.\n")

    # =========================================================================
    # ── SECTION 7 : Push modèle final ────────────────────────────────────────
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏁 Toutes les phases terminées — push du modèle final.")
    print("=" * 70)

    final_dir = f"{OUTPUT_DIR}_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    api.upload_folder(
        folder_path=final_dir,
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        commit_message=f"Modèle final whisper-{MODEL_SIZE} darja algérienne",
    )
    wandb.finish()
    print(f"\n🎉 Entraînement terminé ! https://huggingface.co/{HF_REPO_ID}")

# 4. Point d'entrée local
@app.local_entrypoint()
def main():
    print("🚀 Lancement de l'entraînement Whisper Medium sur Modal (GPU A10G)...")
    run_training.remote()
