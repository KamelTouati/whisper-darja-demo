# =============================================================================
# Fine-tuning Whisper — Darja Algérienne — Entraînement SÉQUENTIEL + STREAMING
# =============================================================================

import os
import subprocess
import gc
import re
import shutil
import json
import string
import time as _time

import torch
import wandb
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── Désinstalle hf_xet (bug ImportError XetProgressReporter) ─────────────────
subprocess.run(["pip", "uninstall", "-y", "-q", "hf_xet"], check=False)

# ── 1 seul GPU (bitsandbytes 4-bit incompatible DataParallel) ─────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_XET"]   = "1"

# ── Cache HF vers /kaggle/tmp (monitoring uniquement — rien de gros en v9) ────
# Le modèle de base (~500 MB) est mis en cache dans hf_home.
# Les datasets en streaming ne créent AUCUN cache disque.
for _d in ["/kaggle/tmp/hf_home", "/kaggle/tmp/scratch"]:
    os.makedirs(_d, exist_ok=True)
os.environ["HF_HOME"] = "/kaggle/tmp/hf_home"
os.environ["TMPDIR"]  = "/kaggle/tmp/scratch"

import tempfile
tempfile.tempdir = "/kaggle/tmp/scratch"

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


# =============================================================================
# ── SECTION 1 : Secrets & Login ───────────────────────────────────────────────
# =============================================================================
from kaggle_secrets import UserSecretsClient
user_secrets  = UserSecretsClient()
HF_TOKEN      = user_secrets.get_secret("HF_TOKEN")
WANDB_API_KEY = user_secrets.get_secret("wandb-api-key")

hf_login(token=HF_TOKEN)
wandb.login(key=WANDB_API_KEY)


# =============================================================================
# ── SECTION 2 : Configuration ─────────────────────────────────────────────────
# =============================================================================
MODEL_SIZE    = "large-v3"
HF_USERNAME   = "touati-kamel"
HF_REPO_ID    = f"{HF_USERNAME}/whisper-algerian-darja-{MODEL_SIZE}"
OUTPUT_DIR    = f"./whisper_algerian_darja_{MODEL_SIZE}"
WANDB_PROJECT = "whisper-algerian-darja"

MODEL_CONFIGS = {
    "small":    {"base_model_id": "openai/whisper-small",    "batch_size": 8, "grad_accum": 4},
    "medium":   {"base_model_id": "openai/whisper-medium",   "batch_size": 4, "grad_accum": 8},
    "large-v3": {"base_model_id": "openai/whisper-large-v3", "batch_size": 2, "grad_accum": 16},
    "large-v2": {"base_model_id": "openai/whisper-large-v2", "batch_size": 2, "grad_accum": 16},
}
cfg = MODEL_CONFIGS[MODEL_SIZE]

# max_steps est calculé dynamiquement depuis la taille réelle du dataset
# (via load_dataset_builder, léger). Ce fallback est utilisé si la taille
# est indisponible (dataset sans métadonnées de split).
DEFAULT_STEPS_FALLBACK = 4000

PHASES = [
    {
        "name":       "kahwa",
        "dataset_id": "oddadmix/arabic-audio-collection-algerian-kahwa-postcast",
        "prefix":     "kahwa",
        "epochs":     2,       # utilisé pour calculer max_steps
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

EVAL_SUBSET_SIZE     = 300    # exemples pour l'eval rapide (streaming .take())
PHASE_STATE_FILENAME = "phase_state.json"
ADAPTER_HUB_PATH     = "last-adapter"

api = HfApi()
print(f"🎯 whisper-{MODEL_SIZE}  |  {HF_REPO_ID}")


# =============================================================================
# ── SECTION 3 : Utilitaires ────────────────────────────────────────────────────
# =============================================================================

def print_disk_usage(label: str = "") -> None:
    out = subprocess.run(
        ["df", "-h", "/", "/kaggle/tmp"],
        capture_output=True, text=True
    ).stdout
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
    """
    Récupère le nombre d'exemples du split train via load_dataset_builder.
    N'a PAS besoin de télécharger les données — lit uniquement le fichier
    de métadonnées du dataset (dataset_infos.json, ~quelques KB).
    Retourne None si les métadonnées sont absentes.
    """
    try:
        builder = load_dataset_builder(dataset_id)
        n = builder.info.splits["train"].num_examples
        print(f"   📊 Taille connue : {n:,} exemples")
        return n
    except Exception as e:
        print(f"   ⚠️  Taille inconnue ({e}), fallback : {DEFAULT_STEPS_FALLBACK} steps")
        return None


def compute_max_steps(n_examples: Optional[int], batch_size: int,
                      grad_accum: int, n_epochs: int) -> int:
    """
    Calcule max_steps depuis la taille du dataset.
    On applique un facteur 0.85 pour tenir compte du filtrage
    (environ 15% des exemples sont rejetés par is_valid).
    """
    if n_examples is None:
        return DEFAULT_STEPS_FALLBACK
    effective_n = int(n_examples * 0.85)   # ~15% filtrés par is_valid
    steps_per_epoch = max(1, effective_n // batch_size)
    total = steps_per_epoch * n_epochs
    print(f"   🔢 max_steps calculé : {effective_n} × {n_epochs} epochs "
          f"/ batch {batch_size} = {total} steps")
    return total


def download_hub_checkpoint(phase_out: str, phase_name: str) -> Optional[str]:
    """
    Télécharge last-checkpoint-{phase_name} depuis HF Hub dans un sous-dossier local.
    Chaque phase a son propre slot de checkpoint sur le Hub :
      last-checkpoint-kahwa, last-checkpoint-loubna, last-checkpoint-rawi
    Cela empêche une phase de récupérer le checkpoint d'une phase précédente.

    Retourne le chemin local du checkpoint ou None si absent/vide.
    """
    hub_folder   = f"last-checkpoint-{phase_name}"
    local_ckpt_dir = os.path.join(phase_out, "hub_checkpoint")
    actual_ckpt    = os.path.join(local_ckpt_dir, hub_folder)
    try:
        print(f"🔄 Pas de checkpoint local — tentative depuis HF Hub ({hub_folder})...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            local_dir=local_ckpt_dir,
            allow_patterns=[f"{hub_folder}/**"],
        )
        # Vérifier qu'il y a bien des poids dans le checkpoint
        has_weights = (
            os.path.isdir(actual_ckpt)
            and any(
                f.endswith((".bin", ".safetensors"))
                for f in os.listdir(actual_ckpt)
            )
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
        else:
            print(f"ℹ️  {hub_folder} sur Hub est absent ou vide — démarrage fresh.")
            return None
    except Exception as e:
        print(f"ℹ️  Impossible de récupérer le checkpoint depuis Hub : {e}")
        return None


# ── Normalisation texte arabe ──────────────────────────────────────────────────
_FR_TAG_RE  = re.compile(r"\[\s*(?:French|FR)\s*:\s*(.*?)\]", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[.*?\]|<.*?>")
_ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670"
_TATWEEL           = "\u0640"
_PUNCT_MAP = {ord(c): None for c in string.punctuation + "\u060C\u061B\u061F\u00AB\u00BB"}


def normalize_arabic(text: str) -> str:
    text = text.translate({ord(c): None for c in _ARABIC_DIACRITICS})
    text = text.replace(_TATWEEL, "")
    text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064A")
    text = text.translate(_PUNCT_MAP)
    return " ".join(text.split()).strip()


def preprocess_transcript(batch: dict) -> dict:
    text = batch["transcript_text"] or ""
    text = _FR_TAG_RE.sub(r"\1", text)
    text = _BRACKET_RE.sub("", text)
    text = normalize_arabic(text)
    batch["transcript_text"] = text
    return batch


def is_valid(example: dict) -> bool:
    duration = example.get("duration", 15)
    if not (0.5 <= duration <= 30.0):
        return False
    text = (example["transcript_text"] or "").strip()
    if not text:
        return False
    return 1.0 <= len(text) / duration <= 25.0


# ── DataCollator ───────────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    En streaming, l'audio n'est PAS en cache disque. Il est décodé ici,
    batch par batch, utilisé pour extraire les Mel spectrogrammes, puis
    libéré par le GC. Aucun fichier audio n'est jamais écrit sur disque.
    """
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


# ── Callbacks ──────────────────────────────────────────────────────────────────
class HubUploadCallback(TrainerCallback):
    """Upload REST checkpoint → HF Hub (pas de clone git).
    Utilise un chemin phase-specific (last-checkpoint-{phase_name})
    pour éviter qu'une phase suivante ne récupère le checkpoint
    d'une phase précédente.
    """
    def __init__(self, api: HfApi, repo_id: str, token: str,
                 phase_name: str) -> None:
        self.api, self.repo_id, self.token = api, repo_id, token
        self.phase_name = phase_name

    def on_save(self, args, state, control, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            return control
        path_in_repo = f"last-checkpoint-{self.phase_name}"
        print(f"☁️  Upload checkpoint step {state.global_step} → {path_in_repo}...")
        try:
            self.api.upload_folder(
                folder_path=ckpt,
                repo_id=self.repo_id,
                path_in_repo=path_in_repo,
                token=self.token,
                commit_message=f"checkpoint step {state.global_step} (phase {self.phase_name})",
            )
            print("✅ Upload OK.")
        except Exception as e:
            print(f"⚠️  Upload échoué (non fatal) : {e}")
        return control


class DiskMonitorCallback(TrainerCallback):
    """
    Log usage disque vers W&B à chaque logging_steps.
    En streaming, hf_datasets devrait rester ~0 GB (rien caché sur disque).
    Ce callback confirme que le disque reste plat — si ça grossit, c'est un bug.
    """
    WATCHED = {
        "hf_home":    "/kaggle/tmp/hf_home",
        "scratch":    "/kaggle/tmp/scratch",
        "system_tmp": "/tmp",
    }

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
    """
    load_best_model_at_end=True est incompatible avec IterableDataset (streaming)
    car le Trainer ne peut pas détecter les frontières d'époques.
    Ce callback le remplace : il sauvegarde l'adapter LoRA localement chaque fois
    que le WER eval s'améliore, et le recharge à la fin de la phase.
    """
    def __init__(self, model, save_dir: str) -> None:
        self.model    = model
        self.save_dir = save_dir
        self.best_wer = float("inf")

    def on_evaluate(self, args, state, control,
                    metrics: Dict[str, float] = None, **kwargs):
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
                print(f"⚠️  Rechargement adapter ignoré ({e}) — modèle garde les poids finaux.")


class TempCleanupCallback(TrainerCallback):
    """Purge les fichiers temporaires anciens (décodages ffmpeg résiduels)."""
    def __init__(self, tmp_dir: str, min_age: int = 300) -> None:
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


# ── Métrique WER ───────────────────────────────────────────────────────────────
def make_compute_metrics(proc: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = proc.tokenizer.pad_token_id
        pred_str  = proc.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = proc.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str  = [normalize_arabic(p) for p in pred_str]
        pairs     = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"wer": 0.0}
        preds, refs = zip(*pairs)
        return {"wer": 100 * jiwer.wer(list(refs), list(preds))}
    return compute_metrics


# =============================================================================
# ── SECTION 4 : Phase State (reprise entre sessions) ──────────────────────────
# =============================================================================

def load_phase_state() -> dict:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO_ID, PHASE_STATE_FILENAME,
                        local_dir=".", token=HF_TOKEN)
        with open(PHASE_STATE_FILENAME) as f:
            state = json.load(f)
        print(f"🔄 Phase state récupéré : {state}")
        return state
    except Exception:
        print("ℹ️  Pas de phase_state.json — première session.")
        return {"completed": [], "global_step_offset": 0}


def save_phase_state(state: dict) -> None:
    with open(PHASE_STATE_FILENAME, "w") as f:
        json.dump(state, f, indent=2)
    try:
        api.upload_file(
            path_or_fileobj=PHASE_STATE_FILENAME,
            path_in_repo=PHASE_STATE_FILENAME,
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            commit_message="update phase_state.json",
        )
    except Exception as e:
        print(f"⚠️  Impossible de pousser phase_state : {e}")


# =============================================================================
# ── SECTION 5 : Init GPU + modèle ─────────────────────────────────────────────
# =============================================================================
print_disk_usage("(démarrage)")
print(f"\n🔍 GPU : {torch.cuda.device_count()} device(s)")
for _i in range(torch.cuda.device_count()):
    _f, _t = torch.cuda.mem_get_info(_i)
    print(f"   GPU {_i} — {torch.cuda.get_device_name(_i)} "
          f"— {_f/1e9:.1f} GB libres / {_t/1e9:.1f} GB")

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
        # PAS de task_type="SEQ_2_SEQ_LM" (bug PEFT #1988 avec Whisper)
    )
    model = get_peft_model(model, peft_config)
    print("✅ Nouvel adapter LoRA initialisé.")

model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="arabic", task="transcribe"
)
model.generation_config.suppress_tokens = []
model.config.use_cache = False
model.print_trainable_parameters()

data_collator   = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
compute_metrics = make_compute_metrics(processor)


# =============================================================================
# ── SECTION 6 : Boucle d'entraînement SÉQUENTIELLE + STREAMING ────────────────
# =============================================================================
wandb.init(
    project=WANDB_PROJECT,
    name=f"whisper-{MODEL_SIZE}-darja--streaming",
    resume="allow",
)

for phase in PHASES:
    phase_name = phase["name"]

    if phase_name in completed_phases:
        print(f"\n⏭️  Phase '{phase_name}' déjà terminée — skip.")
        continue

    phase_idx = PHASES.index(phase) + 1
    print(f"\n{'='*70}")
    print(f"🚀 PHASE {phase_idx}/{len(PHASES)} : {phase_name.upper()}")
    print(f"   Dataset : {phase['dataset_id']}")
    print(f"   Epochs  : {phase['epochs']}  |  LR : {phase['lr']}")
    print(f"{'='*70}")
    print_disk_usage(f"(début phase {phase_name})")

    # ── Récupère la taille du dataset pour calculer max_steps ─────────────────
    print("🔍 Récupération taille dataset...")
    n_examples = get_dataset_size(phase["dataset_id"])
    max_steps  = compute_max_steps(
        n_examples,
        batch_size=cfg["batch_size"],
        grad_accum=cfg["grad_accum"],
        n_epochs=phase["epochs"],
    )

    # ── Streaming dataset ─────────────────────────────────────────────────────
    # L'audio est décodé par le DataCollator batch par batch en RAM puis GC'd.
    # RIEN n'est écrit sur disque — fix définitif du crash disque.
    print(f"\n📥 Création du stream : {phase['dataset_id']}")
    _pfx = phase["prefix"]

    def _make_stream(dataset_id=phase["dataset_id"], pfx=_pfx):
        ds = load_dataset(dataset_id, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = ds.map(lambda x: {
            **x,
            "original_video_id": f"{pfx}_{x['original_video_id']}",
        })
        ds = ds.map(preprocess_transcript)
        ds = ds.filter(is_valid)
        return ds

    # IMPORTANT : deux appels _make_stream() indépendants (eval et train
    # ne partagent PAS le même itérateur).
    #
    # train_ds utilise .repeat() — le stream est INFINI. La longueur
    # d'entraînement est controlée uniquement par max_steps.
    # POURQUOI .repeat() et pas .skip() :
    #   .skip(N) crée un SkipExamplesIterable avec
    #   block_sources_order_when_shuffling=True. Quand ce stream
    #   s'épuise après ~1 passe, le Trainer appelle set_epoch(1) pour
    #   l'époque 2 → shuffle_data_sources() → DataSourcesShufflingDisallowed.
    #   .repeat() crée un RepeatExamplesIterable qui ne se termine jamais
    #   → aucune frontière d'époque → aucun reshuffling → pas de crash.
    eval_ds  = _make_stream().take(EVAL_SUBSET_SIZE)
    # .repeat(100) = effectively infinite for our use case :
    # 100 × ~19 774 exemples = ~1.97M exemples >> 4 942 steps × 32 batch = 158K nécessaires.
    # Note : repeat() sans argument n'est PAS supporté dans cette version de datasets
    # (TypeError: missing 1 required positional argument: 'num_times').
    train_ds = _make_stream().shuffle(seed=42, buffer_size=2000).repeat(100)

    print(f"✅ Streams prêts — {EVAL_SUBSET_SIZE} exemples eval | "
          f"train infini ({max_steps} steps contrôlés par max_steps)")

    # ── Dossier de sortie par phase ────────────────────────────────────────────
    phase_out = os.path.join(OUTPUT_DIR, f"phase_{phase_name}")
    os.makedirs(phase_out, exist_ok=True)

    # ── Reprise intra-phase ────────────────────────────────────────────────────
    # Priorité 1 : checkpoint local (même session Kaggle, rare)
    # Priorité 2 : checkpoint depuis HF Hub (session Kaggle précédente)
    # Priorité 3 : démarrage fresh
    #
    # Le Trainer lit trainer_state.json → global_step=4200, puis
    # reprend à partir de là jusqu'à max_steps=4942 (→ 742 steps restants).
    # Les poids LoRA + état optimiseur sont restaurés depuis le checkpoint.
    # Les données streaming recommencent depuis le début (position non sauvegardée),
    # mais le LR cosine est restauré au bon point → entraînement correct.
    resume_ckpt = get_last_checkpoint(phase_out)
    if resume_ckpt:
        print(f"🔄 Reprise locale depuis : {resume_ckpt}")
    else:
        resume_ckpt = download_hub_checkpoint(phase_out, phase_name)
        if not resume_ckpt:
            print("🆕 Démarrage fresh pour la phase.")  

    # ── Callback BestWERSave (remplace load_best_model_at_end) ────────────────
    best_saver = BestWERSaveCallback(model=model, save_dir=phase_out)

    # ── Training Arguments ─────────────────────────────────────────────────────
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
        # STREAMING : utilise max_steps au lieu de num_train_epochs
        # (l'IterableDataset n'a pas de len() connu par le Trainer)
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
        # load_best_model_at_end=False obligatoire avec IterableDataset
        # (remplacé par BestWERSaveCallback ci-dessus)
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
            HubUploadCallback(api=api, repo_id=HF_REPO_ID, token=HF_TOKEN,
                              phase_name=phase_name),
            DiskMonitorCallback(output_dir=phase_out),
            TempCleanupCallback(tmp_dir="/kaggle/tmp/scratch"),
            best_saver,
        ],
    )

    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n🏃 Entraînement streaming — phase {phase_name} "
          f"({max_steps} steps)...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Recharge le meilleur adapter sauvegardé pendant la phase
    best_saver.reload_best()

    # ── Post-phase : sauvegarde adapter LoRA → HF Hub ─────────────────────────
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
        print(f"⚠️  Upload adapter échoué : {e}")

    # ── Phase state ────────────────────────────────────────────────────────────
    global_step_offset += trainer.state.global_step
    completed_phases.add(phase_name)
    phase_state["completed"]          = list(completed_phases)
    phase_state["global_step_offset"] = global_step_offset
    save_phase_state(phase_state)

    # ── Libération mémoire (pas de cache disque à purger en streaming) ─────────
    del trainer, train_ds, eval_ds
    gc.collect()
    torch.cuda.empty_cache()
    print_disk_usage(f"(fin phase {phase_name})")

    print(f"\n✅ Phase '{phase_name}' terminée.\n")


# =============================================================================
# ── SECTION 7 : Push modèle final ─────────────────────────────────────────────
# =============================================================================
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
    commit_message=(
        f"Modèle final whisper-{MODEL_SIZE} darja algérienne "
        f"(streaming séquentiel  : kahwa→loubna→rawi)"
    ),
)

wandb.finish()
print(f"\n🎉 Entraînement terminé !")
print(f"📦 Modèle : https://huggingface.co/{HF_REPO_ID}")
print(f"\n➡️  Prochaine étape : MODEL_SIZE = \"large-v3\" et relancer.")
