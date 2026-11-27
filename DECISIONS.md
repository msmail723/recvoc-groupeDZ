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

## D8 — Budget push-to-talk : attente médiane après relâchement ≤ 0,25 s   (milestone: d3)
- Question: combien de temps l'utilisateur peut-il attendre entre le relâchement du bouton et le texte final, et quelle colonne mesure ce coût ?
- Alternatives: budget sur `finish_p50` (attente médiane après relâchement), sur `lag_p50` (retard médian des mots validés pendant qu'on parle) ou sur `rtf` ; limites de 0,2 s (l'exemple du cours), 0,25 s et 0,5 s.
- Protocol: cas d'usage = commandes vocales courtes (2-5 s, « mets un minuteur de dix minutes »). C'est le texte final qui est exécuté ; les partiels affichés pendant qu'on parle sont un confort. Dans une conversation humaine, les silences entre deux tours de parole durent environ 0,2-0,3 s : au-delà, l'assistant semble « réfléchir ».
- Evidence: `results/streaming.csv`, `results/figures/streaming_pareto.png`, lignes `task=asr_streaming, cfg_milestone=d3` (15 lignes, 20 énoncés LibriSpeech test-clean, CPU de la cloudtop : Intel Xeon 2,2 GHz, 24 vCPU). `finish_p50` va de 0,029 s (transducteurs zipformer) à 1,080 s (Whisper tiny re-décodé toutes les 0,5 s).
- Decision: coût = `finish_p50`, budget = 0,25 s ; qualité = `wer_stream` (WER du texte validé). `lag_p50` sert de critère secondaire : un mot validé ne devrait pas apparaître plus d'environ 1 s après avoir été prononcé.
- Would change our mind if: les requêtes étaient de longues dictées (20 s), où le retard pendant la parole compte plus que l'attente finale ; ou si la machine de démo était plus rapide (sur le Mac du prof, l'attente de la référence tombe à 43 ms ; sur notre cloudtop elle est de 205 ms, si bien qu'un budget de 0,2 s l'exclurait à 5 ms près).

## D9 — Système déployé : le transducteur streaming fastconformer (reference/fastconformer-114M-int8)   (milestone: d3)
- Question: quel système déployer pour la démo push-to-talk sous le budget D8, et que prédit-on avant de l'essayer ?
- Alternatives: 15 configurations streaming :
  - Whisper tiny, wav2vec2-base-960h, notre `student/ctc+3gram` (D7) et notre `student/w2v2-960h+3gram`, chacun re-décodé avec notre LocalAgreement-2 à un pas de 0,5, 1 ou 2 s ;
  - les transducteurs nativement streaming : fastconformer-114M, zipformer-20M et notre `student/zipformer-20M+lead1s` (1 s de silence au début).
- Protocol:
  - `python -m speech.streaming_sim` (configurations par défaut + `--models` pour nos systèmes), 20 énoncés (137 s d'audio, 383 mots), même horloge simulée et mêmes fins de mots (alignement forcé) pour tous les systèmes ;
  - puis `python -m asr_demo.replay --librispeech data/LibriSpeech --n 8 --realtime` sur 6 candidats : l'audio est envoyé à vitesse réelle par morceaux de 100 ms, comme le fait le navigateur ;
  - tout sur le CPU de la cloudtop, sans autre programme lourd.
- Evidence:
  - Front (`wer_stream` ↓, `finish_p50` ↓), calculé avec notre `pareto_mask` sur les 15 lignes d3 : fastconformer (WER 0,021, attente 0,205 s, retard p50 0,68 s), `student/w2v2-960h+3gram` au pas de 2 s (0,018 ; 0,267 s ; retard p50 2,90 s), `student/zipformer-20M+lead1s` (0,091 ; 0,029 s). Toutes les autres configurations sont dominées.
  - Sous le budget de 0,25 s, il ne reste que le fastconformer et le zipformer+lead1s. Le fastconformer fait 4× moins d'erreurs et a le plus petit retard de validation de tous les systèmes (p50 0,68 s, p90 0,87 s).
  - Replay (lignes `task=demo_replay, cfg_milestone=d3`, moyennes sur 8 enregistrements de 2,7 à 12,8 s) :
    - fastconformer : premier partiel 1,05 s, attente 205 ms, RTF 0,38 ;
    - notre `student/local-agreement(w2v2-960h+3gram)` : premier partiel 0,69 s, attente 458 ms ;
    - wav2vec2-960h + 3-gramme hors ligne : aucun partiel, attente 375 ms ;
    - Whisper tiny hors ligne : attente 819 ms.
  - Autres constats :
    - la seconde de silence au début fait passer le WER du petit zipformer de 0,253 à 0,091 (sans elle, il perd les premiers mots, cours L6) ;
    - le n-gramme qui valide trop tôt perd le contexte de droite : `student/ctc+3gram` fait 0,037 en streaming au pas de 0,5 s, contre 0,023 hors ligne ;
    - le modèle 960h garde 0,023 à tous les pas.
- Decision: déployer reference/fastconformer-114M-int8. **Prédiction pour la démo**, écrite avant de l'essayer :
  - premier partiel ≈ 1,0 s après le début de la parole ;
  - attente après relâchement ≈ 0,2 s, **constante quelle que soit la longueur de la requête** (en streaming, il ne reste que le dernier morceau et un flush à traiter) ;
  - WER plus élevé que sur LibriSpeech (0,021), autour de 5-15 % sur nos propres voix (accent non natif, micro d'ordinateur, pas de LM externe).

  À comparer avec notre `student/local-agreement(w2v2-960h+3gram)` re-décodé : premier partiel ≈ 0,7 s, mais une attente qui **augmente avec la longueur de la requête** (≈ 0,3 s pour 3 s de parole, ≈ 0,5 s pour 10 s), car la passe finale décode tout le buffer.
- Would change our mind if:
  - l'écart de WER n'est pas significatif : 2,1 % contre 1,8 %, soit 8 contre 7 erreurs sur 383 mots, très en dessous du bruit de 20 énoncés (il faudrait un bootstrap apparié, cours L5) ;
  - une machine de démo plus rapide faisait passer notre `w2v2-960h+3gram` sous 0,25 s au pas de 0,5 s (0,138 s sur le Mac du prof) : le choix se ferait alors sur le WER et sur les partiels ;
  - la démo montrait des premiers mots perdus ou une forte pénalité d'accent pour le transducteur.
- Vérification de la prédiction (replay temps réel, 8 enregistrements LibriSpeech, lignes `task=demo_replay, cfg_milestone=d3`, droite des moindres carrés « attente en fonction de la durée » sur les 8 enregistrements de chaque système) :
  - **premier partiel** : prédit ≈ 1,0 s, mesuré **1,05 s** en moyenne → **vérifié** ;
  - **attente après relâchement** : prédit ≈ 0,2 s et constante, mesuré **206 ms + 0,0 ms par seconde d'audio** (de 182 à 224 ms, pour des enregistrements de 2,7 à 12,8 s) → **vérifié** : l'attente du transducteur ne dépend pas de la longueur ;
  - **système re-décodé de comparaison** : mesuré 68 ms + 49 ms par seconde d'audio, soit ≈ 0,22 s pour 3 s et ≈ 0,56 s pour 10 s → la tendance prédite (une attente qui grandit avec la longueur) est **vérifiée**. Notre prédiction était trop pessimiste pour les requêtes courtes (0,22 s au lieu de 0,3 s) et un peu optimiste pour les longues (0,56 s au lieu de 0,5 s) ;
  - **WER sur nos voix** : le replay utilise des voix LibriSpeech (WER moyen par enregistrement 0,053 sur ces 8 enregistrements) ; nos voix sont mesurées par la démo live ci-dessous.
- Démo live (micro d'ordinateur → navigateur → serveur sur la cloudtop via la redirection de port de VS Code ; lignes `task=demo, cfg_milestone=d3`, enregistrements dans `results/demo_recordings/`) : 5 phrases en anglais (commandes de 4,5 à 9,1 s, bouton compris), plus 1 essai en français.
  - **WER** : 4 erreurs sur 52 mots = **7,7 %** → dans la fourchette prédite (5-15 %) → **vérifié**. Les erreurs sont surtout en **début de phrase** (« turn on » entendu « trunn » ou « then on », 2 fois sur 2) et sur un mot (« speech » → « speaker »). C'est une version atténuée du phénomène « premiers mots perdus » du cours L6.
  - **Premier partiel** : la page affiche 1,87 à 2,46 s, ce qui semble contredire la prédiction (1,0 s). Mais il est mesuré **depuis l'appui sur le bouton**, et nos enregistrements commencent par 0,74 à 1,46 s de silence avant la parole (mesuré sur les fichiers `.wav`, seuil d'énergie à −30 dB du maximum). **Mesuré depuis le début de la parole** : 0,77 à 1,40 s, **1,03 s en moyenne** → la prédiction est **vérifiée**. Ce métrique dépend donc aussi de l'utilisateur, pas seulement du modèle.
  - **Attente après relâchement** : 208 à 289 ms (médiane 267 ms), **sans lien avec la longueur** (260 ms pour la phrase de 9,1 s) → la forme prédite (constante) est **vérifiée**, mais le niveau est **plus haut d'environ 60 ms** que le replay (205 ms) et **dépasse notre budget de 0,25 s** sur 4 phrases sur 5. Explication probable (non mesurée séparément) : en live, l'audio et le texte final passent par le navigateur, une WebSocket et le tunnel réseau entre l'ordinateur et la cloudtop, alors que le replay tourne dans le même processus que le modèle. Le RTF est aussi un peu plus haut en live (0,45 contre 0,38).
  - **Essai en français** (« Bonjour, j'adore le module de reconnaissance vocale ») : WER = 1,00. Le modèle n'a appris que l'anglais : il « entend » des mots anglais. C'est une limite à montrer à l'oral (une langue hors du domaine d'entraînement).
  - **Conséquence sur la décision** : on garde le fastconformer. Le dépassement du budget vient du chemin réseau de notre démo, pas du modèle (205 ms dans le même processus), et disparaîtrait avec le serveur sur l'ordinateur lui-même, comme le prévoit le cours. Le seul système toujours sous le budget (zipformer + 1 s, 29 ms) fait environ 4× plus d'erreurs (9,1 % contre 2,1 % sur les 20 énoncés). 5 phrases, c'est peu : ces chiffres de démo illustrent, ils ne prouvent pas.
