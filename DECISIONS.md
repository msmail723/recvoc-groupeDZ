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

## D5 — Modèle neuronal pour le push-to-talk sur laptop : wav2vec2-base-960h   (milestone: d2)
- Question: parmi les modèles pré-entraînés, lequel déployer pour un usage interactif (une phrase à la fois) sur CPU ?
- Alternatives: 6 modèles de 4 familles : Whisper (tiny.en 38 M, base.en 73 M, small.en 242 M), distil-Whisper (distil-small.en 166 M), CTC (wav2vec2-base-960h 94 M), Moonshine (moonshine-tiny 27 M).
- Protocol: les mêmes 200 énoncés LibriSpeech test-clean (22,5 min d'audio, ≤ 20 s), batch 1, CPU de la cloudtop (24 cœurs, fp32), 1 échauffement non chronométré ; WER après normalisation minimale (minuscules, sans ponctuation) ; latence = temps par énoncé, p90.
- Evidence: `results/figures/asr_bench.png`, lignes `task=asr, cfg_milestone=d2, cfg_batch=1`. WER / p90 / RTF : wav2vec2 3,3 % / 715 ms / 0,06 ; moonshine-tiny 3,6 % / 1155 ms / 0,09 ; whisper-tiny 5,0 % / 1370 ms / 0,13 ; whisper-base 3,7 % / 1742 ms / 0,17 ; distil-small 3,9 % / 2877 ms / 0,32 ; whisper-small 3,1 % / 7413 ms / 0,71. Front de Pareto (WER, p90) = {wav2vec2-base-960h, whisper-small.en}.
- Decision: wav2vec2-base-960h. C'est le seul modèle sous notre budget (p90 ≤ 1000 ms sur des phrases de ~7 s), avec un WER au niveau des meilleurs. Il est aussi le seul auquel on peut ajouter un LM externe (D7).
- Would change our mind if: les écarts de WER entre 3,1 et 3,7 % ne sont **pas significatifs** sur 200 phrases (il faudrait un bootstrap apparié). Notre normaliseur pénalise Whisper (il écrit « 20 », « Mr. »…), donc son WER est surestimé. Si la mémoire comptait (27 M contre 94 M), moonshine-tiny serait le choix. Whisper-small ne serait choisi que pour du hors-ligne : sa RTF de 0,71 donne ~3,5 s d'attente pour une commande de 5 s.

## D6 — Pas de batch 16 sur CPU (et jamais avec wav2vec2-base)   (milestone: d2)
- Question: traiter 16 phrases à la fois (débit) est-il intéressant sur notre machine ?
- Alternatives: batch 1 vs batch 16, pour les 6 modèles.
- Protocol: comme D5 ; RTF = temps total de calcul / durée totale de l'audio.
- Evidence: lignes `task=asr, cfg_milestone=d2`. Sur CPU, le batch 16 est **plus lent** pour tous les modèles (RTF tiny 0,13 → 0,24 ; small 0,71 → 1,22 ; moonshine 0,09 → 0,18 ; wav2vec2 0,06 → 0,11). Pour wav2vec2-base, le WER explose : 3,3 % → 24,5 %. Diagnostic : son extracteur n'utilise pas de masque d'attention (`return_attention_mask=False`), donc le silence ajouté pour égaliser les longueurs du paquet perturbe la reconnaissance. Sur 16 phrases, la plus courte (2,7 s) est parfaitement reconnue seule et donne une transcription **vide** en batch.
- Decision: batch 1 partout sur cette machine, y compris pour du hors-ligne. Pas de batch avec les modèles sans masque d'attention.
- Would change our mind if: on disposait d'un GPU, où le batch remplit les unités de calcul. **Limite** : Ma cloudtop que j'utilisee n'a pas de GPU, la comparaison CPU/GPU et fp16/fp32 n'a pas été mesurée.

## D7 — LM explicite pour CTC : beam 10 + 3-gramme, α = 0,5, β = 1   (milestone: d2)
- Question: qu'apporte un LM n-gramme externe à un modèle CTC, à quel coût, et avec quels réglages ?
- Alternatives: glouton ; beam sans LM (10, 50, 100) ; beam + 3-gramme LibriSpeech élagué × α ∈ {0,3 ; 0,5 ; 1 ; 2} × β ∈ {0 ; 1 ; 2} × beam ∈ {10, 50, 100}.
- Protocol: wav2vec2-base-100h (modèle acoustique volontairement plus faible, 100 h de données étiquetées), 200 énoncés test-clean, log-probabilités calculées une fois puis décodées par pyctcdecode + KenLM sur CPU ; temps = décodage seul, par énoncé.
- Evidence: `results/figures/ctc_lm_wav2vec2-base-100h.png`, `results/speech_ctc_lm/ctc_lm_wav2vec2-base-100h.csv`, lignes `task=ctc_decoding, cfg_milestone=d2`. Glouton 6,0 % (0,08 ms) ; beam sans LM 6,2 % (10-19 ms), donc **aucun gain** de la recherche seule ; beam + 3-gramme **3,7 %** (α = 0,5, β = 1, beam = 10, 8 ms), soit −39 % d'erreurs relatives. α = 2 dégrade (6,5-7,7 %) : le LM « écrase » l'acoustique, comme λ = 64 au jour 1. Beam 50 ou 100 n'apportent rien de plus.
- Decision: beam = 10, α = 0,5, β = 1. Le coût (~8 ms par phrase) est négligeable devant le modèle acoustique. Comparaison explicite / implicite : CTC + 3-gramme atteint un WER proche de whisper-small (3,1 %) pour une fraction du calcul, et le LM se change en ré-entraînant un n-gramme sur du texte du domaine.
- Would change our mind if: α et β ont été choisis **sur les mêmes 200 phrases** que celles où on rapporte le WER (pas de séparation dev/test), donc 3,7 % est optimiste. À valider sur d'autres phrases (test-other ou un autre tirage). Les écarts entre les meilleurs réglages (3,7-3,8 %) sont dans le bruit. Le 4-gramme n'a pas été testé (1,3 Go).
