"""Adaptation d'un petit LLM par trois méthodes, puis benchmark.

1. Prompting (zero-shot / few-shot fixe)
2. RAG : on récupère les k joueurs les plus proches du train et on les injecte dans le prompt
3. Fine-tuning LoRA (PEFT) sur les paires (profil -> valeur)

Modèle par défaut : Qwen2.5-0.5B-Instruct (tourne sur Colab T4, sur Mac M1/M2 via MPS, ou CPU lentement).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .prompts import (answer_text, chat_messages, decision_answer, decision_messages, describe_player,
                      parse_decision, parse_value)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_llm(model_name: str = DEFAULT_MODEL, adapter_path: str | None = None):
    """Charge le tokenizer et le modèle (optionnellement avec un adaptateur LoRA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    return tok, model


@dataclass
class LLMPredictor:
    """Enveloppe commune aux 3 méthodes : row(s) -> valeur en euros."""
    tok: object
    model: object
    method: str = "zero_shot"          # zero_shot | few_shot | rag | finetuned
    retriever: "Retriever | None" = None
    few_shot_examples: list | None = None
    max_new_tokens: int = 24
    batch_size: int = 16
    task: str = "value"                # value (sujet 1) | decision (sujet 2)
    threshold_m: float = 10
    hide: tuple = ()                   # variables masquées dans le prompt, ex. ("nationality",)

    def _messages(self, row):
        if self.task == "decision":
            ex = self.retriever.examples_for(row) if self.method == "rag" else (
                self.few_shot_examples if self.method == "few_shot" else None)
            return decision_messages(row, self.threshold_m, ex, hide=self.hide)
        if self.method == "few_shot":
            return chat_messages(row, self.few_shot_examples)
        if self.method == "rag":
            return chat_messages(row, self.retriever.examples_for(row))
        return chat_messages(row)  # zero_shot et finetuned : prompt nu

    def generate(self, rows: list[dict]) -> list[str]:
        import torch
        outs = []
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template(self._messages(r), tokenize=False, add_generation_prompt=True)
                     for r in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            outs += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    def predict_proba(self, df: pd.DataFrame | list[dict]) -> np.ndarray:
        """Mode décision : P(YES) lue dans les logits du token qui suit 'Decision:'.
        Plus stable que la génération, et donne un score continu (utile pour LIME / SHAP et les seuils)."""
        import torch
        rows = df.to_dict("records") if isinstance(df, pd.DataFrame) else df
        yes_ids = self.tok.encode(" YES", add_special_tokens=False)
        no_ids = self.tok.encode(" NO", add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1, "' YES' / ' NO' doivent être des tokens uniques"
        yes, no = yes_ids[0], no_ids[0]
        probs = []
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template(self._messages(r), tokenize=False, add_generation_prompt=True)
                     + "Decision:" for r in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]  # padding à gauche -> dernier token réel
            two = torch.stack([logits[:, no], logits[:, yes]], dim=1).float()
            probs += torch.softmax(two, dim=1)[:, 1].cpu().tolist()
        return np.array(probs)

    def predict(self, df: pd.DataFrame | list[dict]) -> np.ndarray:
        rows = df.to_dict("records") if isinstance(df, pd.DataFrame) else df
        parse = parse_decision if self.task == "decision" else parse_value
        return np.array([parse(t) for t in self.generate(rows)])


class Retriever:
    """RAG minimaliste : plus proches voisins sur les variables sportives (+ même championnat)."""

    NUM = ["age", "goals_p90", "assists_p90", "minutes"]

    def __init__(self, train: pd.DataFrame, k: int = 5, label_col: str | None = None, hide: tuple = ()):
        """label_col : colonne 0/1 pour le mode décision (sinon exemples de valeur marchande)."""
        self.train = train.reset_index(drop=True)
        self.k, self.label_col, self.hide = k, label_col, hide
        self.scaler = StandardScaler().fit(self._num(self.train))
        self.nn = NearestNeighbors(n_neighbors=k * 4).fit(self.scaler.transform(self._num(self.train)))

    @staticmethod
    def _num(d):
        d = pd.DataFrame(d if isinstance(d, pd.DataFrame) else [d]).copy()
        d["goals_p90"] = d.get("goals_p90", d["goals"] / d["minutes"] * 90)
        d["assists_p90"] = d.get("assists_p90", d["assists"] / d["minutes"] * 90)
        return d[Retriever.NUM].astype(float).values

    def examples_for(self, row) -> list[tuple[str, str]]:
        _, idx = self.nn.kneighbors(self.scaler.transform(self._num(row)))
        cand = self.train.iloc[idx[0]]
        same = cand[cand["league"] == row.get("league")]
        cand = pd.concat([same, cand.drop(same.index)]).head(self.k)
        if self.label_col:
            return [(describe_player({k: v for k, v in r.items() if k not in self.hide}),
                     decision_answer(r[self.label_col])) for _, r in cand.iterrows()]
        return [(describe_player(r), answer_text(r)) for _, r in cand.iterrows()]


def pick_few_shot(train: pd.DataFrame, n: int = 6, seed: int = 0, label_col: str | None = None,
                  hide: tuple = ()) -> list[tuple[str, str]]:
    """Exemples few-shot fixes, répartis sur les quantiles de valeur."""
    q = pd.qcut(train["market_value"].rank(method="first"), n, labels=False)
    ex = train.groupby(q).sample(1, random_state=seed)
    if label_col:
        return [(describe_player({k: v for k, v in r.items() if k not in hide}), decision_answer(r[label_col]))
                for _, r in ex.iterrows()]
    return [(describe_player(r), answer_text(r)) for _, r in ex.iterrows()]


# ----------------------------------------------------------------------------- fine-tuning LoRA

def finetune_lora(train: pd.DataFrame, model_name: str = DEFAULT_MODEL, out_dir: str = "outputs/lora",
                  epochs: int = 2, lr: float = 2e-4, r: int = 16, max_rows: int | None = 3000, seed: int = 42,
                  task: str = "value", label_col: str | None = None, threshold_m: float = 10,
                  hide: tuple = (), sample_weight: str | None = None, batch_size: int = 4, grad_accum: int = 2):
    """Fine-tuning LoRA supervisé : la perte n'est calculée que sur la réponse de l'assistant.

    task='decision' : entraîne sur label_col (0/1). sample_weight : colonne de poids utilisée pour
    rééchantillonner le train (ex. poids de reweighing pour réduire le biais).
    batch_size x grad_accum = lot effectif (8) : petits lots pour tenir dans la mémoire d'un Mac 16 Go."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    device = get_device()
    if device == "mps":   # libère le cache laissé par les inférences précédentes du notebook
        import gc
        gc.collect()
        torch.mps.empty_cache()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    n = min(len(train), max_rows or len(train))
    weights = train[sample_weight] if sample_weight else None
    data = train.sample(n, random_state=seed, weights=weights, replace=weights is not None)
    samples = []
    for _, row in data.iterrows():
        if task == "decision":
            msgs, ans = decision_messages(row, threshold_m, hide=hide), decision_answer(row[label_col])
        else:
            msgs, ans = chat_messages(row), answer_text(row)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + ans + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        samples.append((f_ids, labels))

    def collate(batch):
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id)
        lab = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            att[i, :len(x)] = 1
        return ids, att, lab

    dl = DataLoader(samples, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_updates = -(-len(dl) // grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * n_updates), n_updates)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for step, (ids, att, lab) in enumerate(dl):
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0 or step + 1 == len(dl):
                opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{len(dl)} loss {loss.item():.4f}")
        print(f"epoch {ep} mean loss {tot / len(dl):.4f}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


# ----------------------------------------------------------------------------- benchmark

def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_pred) & (y_pred > 0)
    lt, lp = np.log10(y_true[ok]), np.log10(y_pred[ok])
    return {
        "valid_answers_pct": round(100 * ok.mean(), 1),
        "MAE_M_EUR": round(np.mean(np.abs(y_true[ok] - y_pred[ok])) / 1e6, 2),
        "MedAPE_pct": round(100 * np.median(np.abs(y_true[ok] - y_pred[ok]) / y_true[ok]), 1),
        "MAE_log10": round(np.mean(np.abs(lt - lp)), 3),
        "within_x2_pct": round(100 * np.mean(np.abs(lt - lp) <= np.log10(2)), 1),
        "spearman": round(pd.Series(lt).corr(pd.Series(lp), method="spearman"), 3),
    }


def benchmark(predictors: dict[str, object], test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """predictors : {nom: objet avec .predict(df)}. Renvoie (métriques, prédictions)."""
    preds = pd.DataFrame(index=test.index)
    rows = []
    for name, p in predictors.items():
        preds[name] = p.predict(test)
        rows.append({"method": name, **regression_metrics(test["market_value"], preds[name])})
    return pd.DataFrame(rows).set_index("method"), preds
