"""Entraîne les deux adaptateurs LoRA du notebook (biaisé et corrigé) hors du notebook.

Utile sur Mac : la progression est visible et la mémoire n'est pas partagée avec le LLM de base
chargé par le notebook. Le notebook réutilise ensuite les adaptateurs de outputs/.

    python scripts/train_lora.py [DATA_DIR] [LORA_ROWS]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep import add_nationality_groups, build_player_seasons, load_raw, temporal_split
from src.fairness import make_decision_labels, reweighing
from src.llm_methods import finetune_lora

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw")
LORA_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Même préparation que le notebook (sections 1 à 3)
df = add_nationality_groups(build_player_seasons(load_raw(DATA_DIR)))
lab = make_decision_labels(df, 10, fit_on=df.season < df.season.max())
train, _ = temporal_split(lab)
train["weight"] = reweighing(train, "confederation", "y_fair")

for tag, label, hide, w in [("biased", "y_hist", (), None), ("fair", "y_fair", ("nationality",), "weight")]:
    out = ROOT / "outputs" / f"lora_{tag}"
    if (out / "adapter_config.json").exists():
        print(f"lora_{tag} déjà entraîné, ignoré", flush=True)
        continue
    finetune_lora(train, MODEL_NAME, str(out), task="decision", label_col=label,
                  threshold_m=10, hide=hide, sample_weight=w, max_rows=LORA_ROWS)
