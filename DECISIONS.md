# Decision log

One entry per scientific choice, at least one per milestone. Keep entries short and factual: the oral
exam asks you to defend them, including the alternatives you rejected.

## D1 — Modèle acoustique GMM-HMM pour les mots isolés : 5 états × 4 mixtures   (milestone: d1)
- Question: quelle configuration (nombre d'états × nombre de gaussiennes) choisir pour reconnaître un chiffre isolé, sous un budget de latence interactif ?
- Alternatives: 12 configurations, états ∈ {2, 3, 5, 8} × mixtures ∈ {1, 2, 4}, sous 5 conditions (propre, bruit blanc 10 dB, babble 5 dB, réverbération RT60 0,6 s, téléphone).
- Protocol: FSDD, split officiel (fichiers d'index 0-4 en test = 300 fichiers, le reste en train), MFCC 13 + Δ + ΔΔ avec normalisation globale ; latence = temps moyen de reconnaissance par mot, sur le CPU de la cloudtop (24 cœurs) ; 1 run par configuration.
- Evidence: `results/figures/isolated_digits.png`, `results/speech_classical/isolated_digits.csv`, lignes `task=gmm_isolated, cfg_milestone=d1`. Front de Pareto (précision propre ↑, latence ↓) = S2M1 (0,957 ; 5,6 ms), S2M2 (0,973 ; 6,1 ms), S2M4 (0,990 ; 6,8 ms), S5M4 (0,997 ; 11,6 ms). Les 8 états sont tous dominés (plus lents, pas plus précis).
- Decision: S5M4. Toutes les configurations restent très loin du budget (300 ms), donc on prend la meilleure précision : 0,997 en propre, et aussi la meilleure en téléphone (0,893 contre 0,793 pour S2M4).
- Would change our mind if: les écarts en propre ne sont pas significatifs (300 fichiers → ±1,1 point à 95 %, donc 0,990 et 0,997 sont indiscernables). Si la mémoire ou la latence comptaient vraiment (embarqué), S2M4 offre presque la même précision avec 2,4× moins de paramètres. Pour le bruit blanc, S5M2 est meilleur (0,790 contre 0,757).

## D2 — Modèle de langage n-gramme : trigramme, lissage selon la quantité de texte   (milestone: d1)
- Question: quel ordre N et quel lissage pour un LM de chiffres où chaque chiffre dépend des 2 précédents ?
- Alternatives: N = 1…5 × {add-k (k = 0,1), interpolation Jelinek-Mercer} × {50, 500, 5000} phrases d'entraînement.
- Protocol: grammaire synthétique d'ordre 2 (`second_order_chain`), perplexité sur 1000 phrases de test, 1 run.
- Evidence: `results/speech_classical/ngram_perplexity.csv`, lignes `task=ngram_lm, cfg_milestone=d1`. Avec 5000 phrases, add-k : 10,69 / 8,89 / **5,05** / 5,16 / 5,67 pour N = 1 à 5. Avec 50 phrases, add-k : 10,75 / 11,15 / 8,08 / 8,61 / 8,63, et l'interpolation fait mieux (7,76 en N = 3 ; 7,73 en N = 4).
- Decision: trigramme (N = 3), qui correspond à l'ordre réel de la source. Add-k quand on a beaucoup de texte (≥ 500 phrases) ; interpolation quand on en a peu, car elle recule vers les ordres bas au lieu de diluer.
- Would change our mind if: la vraie dépendance était plus longue, ou si le coût de décodage comptait : un trigramme multiplie les états du graphe par |V| (§ cours L2), alors que le bigramme est « gratuit ».

## D3 — Poids du LM et pénalité d'insertion pour les chiffres enchaînés   (milestone: d1)
- Question: le bigramme améliore-t-il la reconnaissance de chiffres enchaînés, et quel poids λ lui donner ?
- Alternatives: acoustique seule vs bigramme avec λ ∈ {0,5, 1, 2, 4, 8, 16, 32, 64} ; pénalité β ∈ {−600 … −30} réglée sur 30 énoncés de dev, WER rapporté sur 120 énoncés de test ; LM du bon domaine (matched) vs hors domaine (chiffres uniformes, mismatched) ; propre, 20, 15 et 10 dB.
- Protocol: GMM-HMM S5M2 (configuration de référence du cours, pour comparer avec ses chiffres), bigramme add-k entraîné sur 5000 séquences de texte seul, décodeur Viterbi word-loop, 1 run.
- Evidence: `results/figures/connected_digits_lm.png`, `results/speech_classical/connected_digits_lm.csv`, lignes `task=gmm_connected, cfg_milestone=d1`. Matched : propre 4,4 % → 1,5 % (λ = 32) ; 20 dB 25,7 % → 9,4 % (λ = 32) ; 15 dB 45,0 % → 18,6 % (λ = 64) ; 10 dB 58,3 % → 28,8 % (λ = 64). Mismatched avec λ = 64 : propre 1,7 % → 8,1 % ; 20 dB 21,1 % → 38,9 %.
- Decision: utiliser le bigramme avec λ ≈ 16-32 et β réglé sur le dev. Le LM aide surtout quand l'acoustique est faible (bruit). Un λ trop grand fait du mal dès que le LM n'est pas du bon domaine.
- Would change our mind if: on élargissait la grille. À 15 et 10 dB l'optimum est à λ = 64, **au bord de la grille**, donc le vrai optimum peut être plus haut. En mismatched à 20 dB, λ = 16 fait encore baisser le WER (21 % → 15 %) avec β = −30, aussi au bord : c'est l'effet « λ agit aussi comme pénalité d'insertion » vu en cours. Il faudrait comparer les LM à λ fixe pour isoler l'effet du contenu.

## D4 — Correction d'un bug : composante GMM morte dans le modèle du silence   (milestone: d1)
- Question: pourquoi le décodeur de chiffres enchaînés donnait-il 100 % de WER sur les vraies données, alors que les tests unitaires passaient ?
- Alternatives: (a) laisser tel quel ; (b) réduire à 1 gaussienne pour le silence ; (c) réparer les composantes de poids nul après l'apprentissage.
- Protocol: inspection des émissions `B` du décodeur → NaN sur toutes les colonnes de `sil` ; dans le modèle `sil`, un état avait une gaussienne de poids 0 et des variances NaN (0/0 dans EM). Le contrôle « training diverged » fourni ne vérifie pas les variances.
- Evidence: avant correction, `gmm_connected` WER = 1,00 partout (uniquement des suppressions) ; après correction, les résultats de D3, proches de la table du cours (L2).
- Decision: (c), fonction `_repair_dead_components` dans `speech/gmm_hmm.py`. Une composante de poids 0 ne contribue pas au mélange : on lui donne des valeurs finies en gardant son poids à 0. Le modèle est inchangé et les NaN disparaissent.
- Would change our mind if: le problème apparaissait aussi sur des modèles de mots (il faudrait alors plus de données, ou moins de mixtures, par état).
