# Decision log

One entry per scientific choice, at least one per milestone. Keep entries short and factual: the oral
exam asks you to defend them, including the alternatives you rejected.

## D1 — Jour 1 : modèle acoustique GMM-HMM à 5 états × 4 gaussiennes   (milestone: d1)
- Question: combien d'états (cases du HMM) et de gaussiennes (cloches du GMM) par chiffre ?
- Alternatives: 12 configurations, 2, 3, 5 ou 8 états × 1, 2 ou 4 gaussiennes, testées en propre et sous 4 dégradations (bruit blanc, brouhaha, écho, téléphone).
- Protocol: FSDD, 300 fichiers de test, MFCC (39 nombres par trame), latence = temps moyen par mot sur le CPU de la cloudtop, 1 run.
- Evidence: `results/figures/isolated_digits.png`, lignes `task=gmm_isolated, cfg_milestone=d1`. Front de Pareto (précision ↑, latence ↓) : S2M1 (95,7 % ; 5,6 ms), S2M2 (97,3 % ; 6,1 ms), S2M4 (99,0 % ; 6,8 ms), S5M4 (99,7 % ; 11,6 ms). Les modèles à 8 états sont tous dominés : plus lents, et pas plus précis.
- Decision: S5M4. C'est la meilleure précision (aussi au téléphone : 89,3 % contre 79,3 % pour S2M4), et tous les modèles restent très en dessous du budget de 300 ms.
- Would change our mind if: 99,0 % et 99,7 % ne sont pas vraiment différents sur 300 fichiers (±1,1 point). Si la mémoire comptait, S2M4 fait presque aussi bien avec 2,4× moins de paramètres.

## D2 — Jour 1 : le modèle de langage aide surtout dans le bruit ; poids λ ≈ 16-32   (milestone: d1)
- Question: ajouter un bigramme (le « cerveau ») améliore-t-il la reconnaissance de chiffres enchaînés, et quel poids λ lui donner ?
- Alternatives: acoustique seule vs bigramme avec λ de 0,5 à 64 ; pénalité β réglée sur 30 énoncés de dev ; LM du bon domaine vs hors domaine ; propre, 20, 15 et 10 dB.
- Protocol:
  - GMM-HMM S5M2 (configuration de référence du cours), bigramme appris sur 5000 séquences de texte, WER sur 120 énoncés de test.
  - Avant cela, choix de l'ordre du LM par la perplexité (lignes `task=ngram_lm`) : le trigramme gagne (5,05 avec 5000 phrases, la source dépend des 2 chiffres précédents), mais le décodeur utilise un bigramme, qui ne coûte aucun état en plus.
  - Bug corrigé dans le code fourni : une gaussienne vide du modèle du silence donnait des NaN, donc un WER de 100 % (fonction `_repair_dead_components`, à signaler au prof).
- Evidence: `results/figures/connected_digits_lm.png`, lignes `task=gmm_connected, cfg_milestone=d1`.
  - Bon domaine : 25,7 % → 9,4 % à 20 dB ; 45,0 % → 18,6 % à 15 dB.
  - Hors domaine avec λ = 64 : 1,7 % → 8,1 % en propre.
- Decision: bigramme avec λ ≈ 16-32 et β réglé sur le dev. Le cerveau aide quand l'oreille entend mal, et fait du mal quand il se trompe et qu'on l'écoute trop.
- Would change our mind if: à 15 et 10 dB, le meilleur λ (64) est au bord de la grille testée, donc le vrai optimum peut être plus haut : il faudrait élargir la grille.

## D3 — Jour 2 : wav2vec2-base-960h pour un usage interactif sur CPU   (milestone: d2)
- Question: quel modèle pré-entraîné choisir pour traiter une phrase à la fois sur le CPU ?
- Alternatives: 6 modèles de 4 familles (Whisper tiny, base et small ; distil-Whisper small ; wav2vec2-base CTC ; Moonshine tiny), en batch 1 et en batch 16.
- Protocol: les mêmes 200 phrases LibriSpeech test-clean, CPU de la cloudtop (24 cœurs, sans GPU), WER après normalisation, latence p90 par phrase.
- Evidence: `results/figures/asr_bench.png`, lignes `task=asr, cfg_milestone=d2`.
  - Batch 1 : wav2vec2 3,3 % / 715 ms ; Moonshine tiny 3,6 % / 1155 ms ; Whisper tiny 5,0 % / 1370 ms ; Whisper small 3,1 % / 7413 ms. Front de Pareto = {wav2vec2, Whisper small}.
  - Batch 16 : plus lent sur CPU pour tous les modèles ; wav2vec2 passe même à 24,5 % d'erreurs, car il ne sait pas ignorer le silence ajouté pour égaliser les longueurs des phrases.
- Decision: wav2vec2-base-960h, en batch 1 : c'est le seul modèle sous notre budget de 1000 ms, avec un WER au niveau des meilleurs.
- Would change our mind if: les écarts entre 3,1 et 3,7 % ne sont pas significatifs sur 200 phrases. Nous n'avons pas pu tester de GPU.

## D4 — Jour 2 : wav2vec2 + 3-gramme, α = 0,5, β = 1, beam = 10   (milestone: d2)
- Question: qu'apporte un modèle de langage externe (3-gramme) à un modèle CTC, et à quel coût ?
- Alternatives: décodage glouton ; beam search sans LM (10, 50, 100) ; beam + 3-gramme avec α ∈ {0,3 ; 0,5 ; 1 ; 2}, β ∈ {0 ; 1 ; 2} et beam ∈ {10, 50, 100}.
- Protocol: wav2vec2-base-100h (modèle volontairement plus faible), 200 phrases, décodage pyctcdecode + KenLM sur CPU, temps de décodage par phrase.
- Evidence: `results/figures/ctc_lm_wav2vec2-base-100h.png`, lignes `task=ctc_decoding, cfg_milestone=d2`. Glouton 6,0 % ; beam sans LM 6,2 % (la recherche seule n'apporte rien) ; beam + 3-gramme **3,7 %** pour ~8 ms par phrase. Avec α = 2, le LM écrase l'acoustique : 6,5 à 7,7 %.
- Decision: beam = 10, α = 0,5, β = 1 (le même réglage que celui que le prof a mis dans la démo).
- Would change our mind if: α et β ont été choisis sur les mêmes phrases que celles du score, donc 3,7 % est optimiste : il faudrait le valider sur d'autres phrases (le nouveau script du prof sépare dev et test).

## D5 — Jour 3 : déployer le transducteur fastconformer, budget d'attente de 0,25 s   (milestone: d3)
- Question: quel système déployer pour l'assistant push-to-talk, et quelle attente après relâchement accepter ?
- Alternatives: 15 configurations streaming :
  - re-décodage avec notre LocalAgreement-2 (pas de 0,5, 1 ou 2 s) de Whisper tiny, wav2vec2, et de nos deux wav2vec2 + 3-gramme ;
  - transducteurs nativement streaming : fastconformer (fourni par le prof, pré-entraîné par NVIDIA), zipformer, et zipformer avec 1 s de silence au début.
- Protocol:
  - **Budget** : attente médiane après relâchement (`finish_p50`) ≤ 0,25 s. Cas d'usage = commandes courtes (« set a timer for ten minutes ») ; une pause de 0,2-0,3 s est naturelle dans une conversation.
  - Mesures : `python -m speech.streaming_sim` sur 20 phrases, replay temps réel sur 8 phrases, démo live au micro sur 5 phrases ; tout sur le CPU de la cloudtop.
- Evidence: `results/figures/streaming_pareto.png`, `results/streaming.csv`, lignes `task=asr_streaming`, `task=demo_replay` et `task=demo` (`cfg_milestone=d3`).
  - Front (erreurs ↓, attente ↓) : fastconformer (2,1 % ; 0,205 s), notre wav2vec2-960h + 3-gramme au pas de 2 s (1,8 % ; 0,267 s, mais les mots apparaissent 2,9 s après avoir été dits), zipformer + 1 s (9,1 % ; 0,029 s).
  - Sous le budget, il reste le fastconformer et le zipformer + 1 s : le fastconformer fait 4× moins d'erreurs et valide les mots le plus vite (0,68 s).
  - **Prédiction écrite avant la démo** : premier mot affiché ≈ 1,0 s après le début de la parole ; attente ≈ 0,2 s, constante quelle que soit la longueur ; 5-15 % d'erreurs sur nos voix.
  - **Vérification**. Replay : premier mot 1,05 s, attente 206 ms, constante (182-224 ms pour des phrases de 2,7 à 12,8 s) ✓. Démo live : 7,7 % d'erreurs (4 sur 52 mots) ✓. Premier mot 1,03 s après le début de la parole ✓ : la page affiche ~2 s parce qu'elle compte depuis l'appui sur le bouton, et on attendait ~1 s avant de parler. Attente 267 ms, toujours constante, mais ~60 ms au-dessus du replay, à cause du réseau entre l'ordinateur et la cloudtop. Essai en français : 100 % d'erreurs (le modèle ne connaît que l'anglais).
- Decision: fastconformer. On le garde malgré le dépassement du budget en live : le surcoût vient du réseau, pas du modèle, et le seul système sous le budget fait 4× plus d'erreurs.
- Would change our mind if: 2,1 % contre 1,8 % = 8 contre 7 erreurs sur 383 mots, ce n'est pas significatif. Sur une machine plus rapide (le Mac du prof), notre wav2vec2-960h + 3-gramme passerait sous le budget, et le choix se ferait sur les erreurs et sur la réactivité.
