"""Test des trois méthodes d'adaptation (fine-tuning complet, LoRA, distillation) sur un mini-modèle
Qwen2 aléatoire et un tokenizer local : vérifie que l'entraînement, la sauvegarde, le rechargement et
la lecture de P(YES) fonctionnent, sans GPU ni téléchargement.
Usage : python -m tests.test_adaptation
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.llm_methods import (LLMPredictor, distill, finetune_full, finetune_lora, load_llm, load_meta,
                             make_student)
from src.prompts import decision_messages, decision_answer

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)


def fake_train(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 34, n), "nationality": rng.choice(["Brazil", "Morocco", "France"], n),
        "club_name": rng.choice(["LOSC Lille", "Juventus"], n), "league": rng.choice(["Ligue 1", "Serie A"], n),
        "games": rng.integers(5, 38, n), "minutes": rng.integers(450, 3000, n),
        "goals": rng.integers(0, 25, n), "assists": rng.integers(0, 12, n),
        "y_hist": rng.integers(0, 2, n), "y_fair": rng.integers(0, 2, n), "weight": rng.uniform(0.5, 1.5, n),
        "market_value": rng.uniform(1e6, 5e7, n),
    })


def build_tiny_model(path: Path, train: pd.DataFrame):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = set()
    for _, r in train.iterrows():
        for m in decision_messages(r, 10) + [{"content": decision_answer(1) + " " + decision_answer(0)}]:
            words.update(pre_tokenizers.Whitespace().pre_tokenize_str(m["content"]))
    special = ["[UNK]", "<|im_start|>", "<|im_end|>", "<pad>", "system", "user", "assistant"]
    tokens = special + sorted({w for w, _ in words} - set(special))
    vocab = {w: i for i, w in enumerate(tokens)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                  pad_token="<pad>", bos_token="<|im_start|>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    return str(path)


def main():
    # AutoTokenizer remplacerait ce tokenizer de test par Qwen2Tokenizer (model_type qwen2) :
    # on force le tokenizer générique, utilisé tel quel par le code testé.
    import transformers
    transformers.AutoTokenizer.from_pretrained = transformers.PreTrainedTokenizerFast.from_pretrained
    train = fake_train()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = build_tiny_model(d / "base", train)
        common = dict(task="decision", threshold_m=10, max_rows=16, epochs=1, batch_size=4)

        full_dir = finetune_full(train, base, str(d / "full"), label_col="y_hist", **common)
        lora_dir = finetune_lora(train, base, str(d / "lora"), label_col="y_fair", hide=("nationality",),
                                 sample_weight="weight", r=4, **common)
        dist_dir = distill(train, full_dir, base, str(d / "distill"), n_layers=2, label_col="y_hist", **common)

        tok_s, student = make_student(base, n_layers=2)
        assert student.config.num_hidden_layers == 2

        for name, (tok, model) in {"full": load_llm(full_dir), "lora": load_llm(base, adapter_path=lora_dir),
                                   "distill": load_llm(dist_dir)}.items():
            p = LLMPredictor(tok, model, "finetuned", task="decision", threshold_m=10)
            proba = p.predict_proba(train.head(5))
            assert proba.shape == (5,) and np.all((proba >= 0) & (proba <= 1)), name
            print(name, np.round(proba, 3))

        m_full, m_lora, m_dist = load_meta(full_dir), load_meta(lora_dir), load_meta(dist_dir)
        print(m_full, m_lora, m_dist, sep="\n")
        assert m_full["params_trainable"] == m_full["params_total"]
        assert m_lora["params_trainable"] < m_lora["params_total"]
        assert m_dist["params_total"] < m_full["params_total"]
        assert load_llm(dist_dir)[1].config.num_hidden_layers == 2
    print("OK : fine-tuning complet, LoRA et distillation")


if __name__ == "__main__":
    main()
