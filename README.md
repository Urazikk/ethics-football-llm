# LLM-powered forward recruitment shortlist, ethical by design

Projet **Ethics of AI #4** (ECE Paris, ING5 Data & IA).

Un petit LLM décide si un attaquant doit entrer dans la **short-list de cibles prioritaires** d'un club (valeur attendue d'au moins 10 M€). On montre que les données historiques poussent le modèle à favoriser certaines nationalités à performance égale. On réduit ce biais, puis on explique les décisions avec **LIME** et **SHAP**.

## Résultats clés sur les données (attaquants du Big 5, 2012/13 à 2025/26, 6 872 saisons)

| Constat | Chiffre |
|---|---|
| Prime de valeur à performance égale, Amérique du Sud (CONMEBOL) | **+25 %** (IC 95 % : +19 % à +31 %) |
| Décote à performance égale, Afrique (CAF) | **-8 %** |
| Prime CONMEBOL en contrôlant aussi le club | +10 % (une partie du biais passe par l'accès aux grands clubs) |
| Taux de short-list historique CONMEBOL / CAF | 48 % / 35 % |
| Disparate impact minimal du label historique | 0,65 (sous le seuil des 80 %) |

Exemple de démo : **Gervinho** (Côte d'Ivoire, Parma) et **Higuaín** (Argentine, Juventus), Serie A 2019/20, **6 buts et 4 passes chacun**. Valeurs : 5,5 M€ contre 25,5 M€.

## Structure

```
ethics-football-llm/
├── notebooks/projet_recrutement_llm.ipynb   # notebook principal (à rendre sur Boostcamp)
├── src/
│   ├── data_prep.py     # table joueur-saison à partir des CSV Transfermarkt
│   ├── bias.py          # prime inexpliquée par la performance, paires de joueurs comparables
│   ├── fairness.py      # labels historique / corrigé, reweighing, métriques d'équité, seuils par groupe
│   ├── prompts.py       # profils en texte, prompts valeur et décision, parsing des réponses
│   ├── llm_methods.py   # 3 méthodes : few-shot, RAG, fine-tuning LoRA + benchmark
│   ├── xai.py           # LIME et SHAP appliqués au LLM, contrefactuels, substitut global
│   └── plots.py
├── scripts/build_notebook.py   # régénère le notebook
├── tests/                      # test de bout en bout avec un LLM factice
├── SUJETS.md                   # autres sujets possibles avec ce dataset
├── data/raw/                   # mettre les CSV Kaggle ici (non versionnés)
└── outputs/                    # adaptateurs LoRA (non versionnés)
```

## Installation

```bash
git clone https://github.com/<ton-compte>/ethics-football-llm.git
cd ethics-football-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Données : télécharger [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores). Copier `players.csv`, `appearances.csv`, `player_valuations.csv` et `clubs.csv` dans `data/raw/`, ou définir `DATA_DIR` :

```bash
export DATA_DIR=~/Downloads/archive
jupyter notebook notebooks/projet_recrutement_llm.ipynb
```

**VS Code** : installer les extensions Python et Jupyter (proposées à l'ouverture du dossier), ouvrir le notebook et choisir le noyau `.venv`. Le dossier `.vscode/` pointe déjà vers ce venv. Sur Mac Apple Silicon, le LLM tourne sur le GPU (MPS).

**Matériel** : le LLM (Qwen2.5-0.5B-Instruct) tourne sur Colab (GPU T4 gratuit) ou sur Mac Apple Silicon (MPS). Le fine-tuning LoRA prend environ 10 min par modèle sur T4. Réduire `N_EVAL` pour aller plus vite.

**Test rapide sans GPU** : `USE_MOCK_LLM=1` remplace le LLM par un GBM, pour vérifier que tout le pipeline tourne.

```bash
python -m tests.test_pipeline ~/Downloads/archive
```

## Méthodologie

1. **Ethics by design** : cartographie des parties prenantes avant le modèle.
2. **Biais** : la valeur (log10) est expliquée par la performance seule (buts/90, passes/90, minutes, matchs, âge, championnat, saison). Le résidu moyen par nationalité mesure la prime inexpliquée.
3. **Labels** : `y_hist` (valeur réelle ≥ 10 M€, biaisé) et `y_fair` (valeur attendue par la performance ≥ 10 M€).
4. **Réduction** : relabeling + reweighing (données), masquage de la nationalité (prompt), seuils par groupe (post-traitement).
5. **LLM** : few-shot, RAG, LoRA, chacun en version biaisée et corrigée, plus deux ablations (masquage seul, relabeling seul). Choix du modèle par `F1(y_fair) x min(1, DI / 0,8)`.
6. **XAI** : LIME et KernelSHAP sur P(YES), lu dans les logits du LLM. Contrefactuel « même joueur, autre nationalité ». Substitut GBM + TreeSHAP pour la vue globale.
7. **Red teaming** : injection dans le prompt, proxy (club).

## Limites

- La valeur Transfermarkt est une estimation communautaire, pas un prix de transfert.
- `y_fair` dépend d'un modèle linéaire de la performance, sans xG ni données tactiques.
- Une explication n'est pas une justification : l'absence de poids SHAP sur la nationalité n'exclut pas des proxys.
