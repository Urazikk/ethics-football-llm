# Sujets possibles avec le dataset Transfermarkt

Chiffres calculés sur les CSV Kaggle (attaquants, 5 grands championnats, 2012/13 à 2025/26).

## 1. Prédiction de la valeur marchande des attaquants (prédiction réaliste)
- **Titre** : LLM-powered forward market value prediction
- **Biais** : à performance égale, prime de +25 % pour la CONMEBOL et de -8 % pour la CAF. Par pays : Portugal +50 %, Argentine +35 %, Brésil +34 %, Nigeria -18 %.
- **Réduction** : aucune (prédiction réaliste), mais le biais est documenté et expliqué.
- **Démo** : Walid Cheddira (Maroc, Serie A 2025/26, 4 buts) vaut 2 M€, contre 23 M€ pour David Neres (Brésil, 3 buts).
- Déjà codé : `task="value"` dans `LLMPredictor`, `finetune_lora(task="value")`, `regression_metrics`.

## 2. Short-list de recrutement (décision) : sujet retenu dans le notebook
- **Titre** : LLM-powered forward recruitment shortlist
- **Biais** : taux de short-list historique de 48 % pour la CONMEBOL contre 35 % pour la CAF. Disparate impact de 0,65.
- **Réduction** : relabeling, reweighing, masquage de la nationalité, seuils par groupe.
- **Démo** : Gervinho contre Higuaín (6 buts et 4 passes chacun), avant et après correction.

## 3. Prédiction du montant de transfert (`transfers.csv`)
- **Biais** : championnat d'origine. À valeur égale, un joueur qui quitte un « petit » championnat se vend moins cher.

## 4. Temps de jeu des jeunes (`appearances.csv`)
- Le LLM prédit si un joueur de moins de 21 ans sera titulaire la saison suivante.
- **Biais** : joueurs formés au club contre étrangers, à stats égales.

## 5. Sanctions disciplinaires (cartons, `appearances.csv`)
- **Biais** : certaines origines sont-elles plus sanctionnées à profil égal ? Sujet sensible, avec de gros risques de facteurs confondants (poste, style de jeu, championnat).

## Note sur « bonne affaire / sous-évalué »
Un label « bonne affaire » défini par la hausse de valeur la saison suivante (+20 %) a été testé. Il **ne montre pas de biais clair** par nationalité (CAF 27 %, CONMEBOL 21 %, UEFA 23 %). Le label « cible prioritaire » (valeur ≥ 10 M€) a donc été retenu, car il porte un biais mesurable.
