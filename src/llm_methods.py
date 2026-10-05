"""Adaptation d'un petit LLM, puis benchmark.

Les trois méthodes d'adaptation du cours (fine-tuning) :
1. Fine-tuning complet : tous les paramètres sont ajustés
2. LoRA (PEFT) : seules de petites matrices de rang faible sont entraînées
3. Distillation : un modèle élève plus petit imite le modèle professeur fine-tuné

Références sans entraînement : prompting zero-shot / few-shot, et RAG (les k joueurs les plus
proches du train sont injectés dans le prompt).

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


def _hide_old_torchao():
    """Colab préinstalle torchao 0.10 ; peft >= 0.18 refuse alors de charger un adaptateur.
    On ne s'en sert pas : on le masque pour que peft le considère absent."""
    import importlib.metadata
    import sys
    from packaging.version import Version
    try:
        if Version(importlib.metadata.version("torchao")) < Version("0.16.0"):
            sys.modules["torchao"] = None
    except importlib.metadata.PackageNotFoundError:
        pass


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
        _hide_old_torchao()
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
    meta: dict | None = None           # infos d'entraînement (paramètres entraînés, durée), cf. load_meta

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


# ----------------------------------------------------------------------------- outils d'entraînement

def _sft_samples(train: pd.DataFrame, tok, task: str, label_col: str | None, threshold_m: float, hide: tuple,
                 max_rows: int | None, sample_weight: str | None, seed: int):
    """Paires (input_ids, labels) : la perte n'est calculée que sur la réponse de l'assistant.
    sample_weight : colonne de poids utilisée pour rééchantillonner le train (reweighing)."""
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
    return samples


def _loader(samples, tok, batch_size: int, shuffle: bool = True):
    import torch
    from torch.utils.data import DataLoader

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

    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def _train_loop(model, dl, epochs: int, lr: float, grad_accum: int, device: str, loss_fn=None):
    """Boucle commune. loss_fn(model, ids, att, lab) -> perte ; par défaut, perte de langage sur la réponse.
    Sur GPU CUDA : précision mixte fp16 (poids et optimiseur en fp32), plus rapide sur un T4."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    if loss_fn is None:
        def loss_fn(m, ids, att, lab):
            return m(input_ids=ids, attention_mask=att, labels=lab).loss
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n_updates = -(-len(dl) // grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * n_updates), n_updates)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for step, (ids, att, lab) in enumerate(dl):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model, ids, att, lab)
            scaler.scale(loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0 or step + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{len(dl)} loss {loss.item():.4f}")
        print(f"epoch {ep} mean loss {tot / len(dl):.4f}")
    model.eval()


def _free_memory(device: str):
    import gc
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _count(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _save_meta(out_dir: str, method: str, total: int, trainable: int, seconds: float, **extra):
    import json
    from pathlib import Path
    meta = {"method": method, "params_total": int(total), "params_trainable": int(trainable),
            "train_seconds": round(seconds, 1), **extra}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_meta(out_dir: str) -> dict:
    """Infos d'entraînement sauvegardées à côté du modèle (vide si absentes, ex. anciens adaptateurs)."""
    import json
    from pathlib import Path
    f = Path(out_dir) / "meta.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _load_for_training(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    return tok, model


# ----------------------------------------------------------------------------- méthode 1 : fine-tuning complet

def finetune_full(train: pd.DataFrame, model_name: str = DEFAULT_MODEL, out_dir: str = "outputs/full",
                  epochs: int = 2, lr: float = 2e-5, max_rows: int | None = 3000, seed: int = 42,
                  task: str = "decision", label_col: str | None = "y_hist", threshold_m: float = 10,
                  hide: tuple = (), sample_weight: str | None = None, batch_size: int = 4, grad_accum: int = 2):
    """Fine-tuning complet : tous les poids du modèle sont ajustés (taux d'apprentissage faible, 2e-5).
    Gradient checkpointing pour tenir sur un GPU T4 (16 Go). Le modèle est sauvegardé en fp16 (~1 Go)."""
    import time
    import torch
    torch.manual_seed(seed)
    device = get_device()
    _free_memory(device)
    tok, model = _load_for_training(model_name, device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    total, trainable = _count(model)
    print(f"fine-tuning complet : {trainable / 1e6:.0f} M paramètres entraînés sur {total / 1e6:.0f} M")
    samples = _sft_samples(train, tok, task, label_col, threshold_m, hide, max_rows, sample_weight, seed)
    t0 = time.time()
    _train_loop(model, _loader(samples, tok, batch_size), epochs, lr, grad_accum, device)
    secs = time.time() - t0
    model.config.use_cache = True
    model.half().save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model
    _free_memory(device)
    _save_meta(out_dir, "full_finetuning", total, trainable, secs, rows=len(samples), epochs=epochs)
    return out_dir


# ----------------------------------------------------------------------------- méthode 2 : LoRA

def finetune_lora(train: pd.DataFrame, model_name: str = DEFAULT_MODEL, out_dir: str = "outputs/lora",
                  epochs: int = 2, lr: float = 2e-4, r: int = 16, max_rows: int | None = 3000, seed: int = 42,
                  task: str = "value", label_col: str | None = None, threshold_m: float = 10,
                  hide: tuple = (), sample_weight: str | None = None, batch_size: int = 4, grad_accum: int = 2):
    """Fine-tuning LoRA supervisé : seules des matrices de rang r sont entraînées (q, k, v, o).

    task='decision' : entraîne sur label_col (0/1). sample_weight : colonne de poids utilisée pour
    rééchantillonner le train (ex. poids de reweighing pour réduire le biais).
    batch_size x grad_accum = lot effectif (8) : petits lots pour tenir dans la mémoire d'un Mac 16 Go."""
    import time
    import torch
    _hide_old_torchao()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    device = get_device()
    _free_memory(device)
    tok, model = _load_for_training(model_name, device)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    total, trainable = _count(model)
    samples = _sft_samples(train, tok, task, label_col, threshold_m, hide, max_rows, sample_weight, seed)
    t0 = time.time()
    _train_loop(model, _loader(samples, tok, batch_size), epochs, lr, grad_accum, device)
    secs = time.time() - t0
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model
    _free_memory(device)
    _save_meta(out_dir, "lora", total, trainable, secs, rows=len(samples), epochs=epochs, r=r)
    return out_dir


# ----------------------------------------------------------------------------- méthode 3 : distillation

def make_student(model_name: str = DEFAULT_MODEL, n_layers: int = 8, device: str = "cpu"):
    """Élève : même architecture et même tokenizer que le professeur, mais seulement n_layers couches
    (réparties uniformément), initialisées depuis le modèle pré-entraîné (comme DistilBERT)."""
    import numpy as np
    import torch
    tok, model = _load_for_training(model_name, device)
    L = model.config.num_hidden_layers
    keep = sorted(set(np.linspace(0, L - 1, n_layers).round().astype(int).tolist()))
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = i
    model.config.num_hidden_layers = len(keep)
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    if getattr(model.config, "max_window_layers", None):
        model.config.max_window_layers = min(model.config.max_window_layers, len(keep))
    return tok, model


def distill(train: pd.DataFrame, teacher_dir: str, model_name: str = DEFAULT_MODEL, out_dir: str = "outputs/distill",
            n_layers: int = 8, temperature: float = 2.0, alpha: float = 0.5, epochs: int = 2, lr: float = 1e-4,
            max_rows: int | None = 3000, seed: int = 42, task: str = "decision", label_col: str | None = "y_hist",
            threshold_m: float = 10, hide: tuple = (), sample_weight: str | None = None,
            batch_size: int = 4, grad_accum: int = 2):
    """Distillation : l'élève apprend à reproduire la distribution du professeur (fine-tuné) sur les
    tokens de la réponse. Perte = alpha x KL(professeur || élève) x T² + (1 - alpha) x perte sur le vrai label.
    Même tokenizer pour les deux modèles, donc la KL se calcule sur tout le vocabulaire."""
    import time
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    device = get_device()
    _free_memory(device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(teacher_dir, dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    tok, student = make_student(model_name, n_layers, device)
    t_total, _ = _count(teacher)
    total, trainable = _count(student)
    print(f"distillation : professeur {t_total / 1e6:.0f} M paramètres, élève {total / 1e6:.0f} M "
          f"({student.config.num_hidden_layers} couches)")
    samples = _sft_samples(train, tok, task, label_col, threshold_m, hide, max_rows, sample_weight, seed)
    T = temperature

    def kd_loss(m, ids, att, lab):
        out = m(input_ids=ids, attention_mask=att, labels=lab)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids, attention_mask=att).logits
        mask = lab[:, 1:] != -100                       # positions qui prédisent un token de la réponse
        s = out.logits[:, :-1][mask].float() / T
        t = t_logits[:, :-1][mask].float() / T
        kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean")
        return alpha * kl * T * T + (1 - alpha) * out.loss

    t0 = time.time()
    _train_loop(student, _loader(samples, tok, batch_size), epochs, lr, grad_accum, device, loss_fn=kd_loss)
    secs = time.time() - t0
    student.half().save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del student, teacher
    _free_memory(device)
    _save_meta(out_dir, "distillation", total, trainable, secs, rows=len(samples), epochs=epochs,
               teacher=str(teacher_dir), n_layers=n_layers, temperature=T, alpha=alpha)
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
