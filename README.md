# Groupe DZ — Projet S (reconnaissance vocale)

## Membres et contributions
| Membre | Contributions |
|---|---|
| Massinissa SMAIL | À COMPLÉTER |
| Sofia ZABOUR | À COMPLÉTER |
| Reda HADDADI | À COMPLÉTER |

## Lancer le projet
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "uvicorn[standard]" hypothesis pygtrie
pip install --no-deps pyctcdecode && pip install https://github.com/kpu/kenlm/archive/master.zip
python scripts/fetch_data.py                      # FSDD, LibriSpeech test-clean, 3-gram, modèles de la démo
pytest -m "d1 or d2 or d3 or stretch"             # 31 tests
bash run_smoke.sh                                 # mini-pipeline de bout en bout, < 10 min (sorties dans results/smoke/)
```
- **Jour 1** : `python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part {isolated,lm,connected,plot}`
- **Jour 2** : `python -m speech.asr_benchmark --librispeech data/LibriSpeech --device cpu --batch 1 16 --models ...` ; `python -m speech.ctc_lm --librispeech data/LibriSpeech --model facebook/wav2vec2-base-100h --arpa data/lm/3-gram.pruned.1e-7.arpa --device cpu`
- **Jour 3** : `python -m speech.streaming_sim` (7 lignes), puis `--models student/ctc+3gram student/w2v2-960h+3gram student/zipformer-20M+lead1s reference/zipformer-20M-int8` ; `python -m asr_demo.replay --librispeech data/LibriSpeech --n 8 --realtime --backends ...` ; démo live : `python -m asr_demo.server` puis http://localhost:8000
- Nos systèmes de la démo : `my_backends.py`. Décisions : `DECISIONS.md` (D1 à D9). Résultats : `results/results.jsonl`, `results/figures/`, `results/pareto.yaml`.
- Machine des mesures : cloudtop Linux, Intel Xeon 2,2 GHz, 24 vCPU, sans GPU.

---

# M2 ISD 26-27 — Data science — Projet S

Ce dépôt est le matériel étudiant du M2 ISD 26-27, Data science, Projet S (reconnaissance de la parole).

- Page du cours : https://ecampus.paris-saclay.fr/course/section.php?id=785660 (section Data science de https://ecampus.paris-saclay.fr/course/view.php?id=256459).
- Cours : ouvrez `lectures/L*.html` dans un navigateur. Les flèches font défiler, F passe en plein écran, N affiche les notes de l'orateur.
- Projet : `SUBJECT.md`, `SUBMISSION_SPEC.md`, `DECISIONS.md`, puis la mise en place ci-dessous.
- Les jeux de données sont téléchargés par `scripts/fetch_data.sh` et ne sont pas dans le dépôt.

# M2 ISD 26-27 — Data science — Project S

This repository is the student material for M2 ISD 26-27, Data science, Project S (speech recognition).

- Course page: https://ecampus.paris-saclay.fr/course/section.php?id=785660 (section Data science of https://ecampus.paris-saclay.fr/course/view.php?id=256459).
- Lectures: open `lectures/L*.html` in a browser. Arrows move, F fullscreen, N speaker notes.
- Project: `SUBJECT.md`, `SUBMISSION_SPEC.md`, `DECISIONS.md`, then the setup below.
- Datasets are downloaded by `scripts/fetch_data.sh` and are not in the repository.

# Project S — Speech recognition under real-time constraints (student bundle)

Start here: `SUBJECT.md` (what to do), `SUBMISSION_SPEC.md` (what to hand in), `DECISIONS.md` (template).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_data.sh data            # FSDD digits, LibriSpeech test-clean, 3-gram LM (speech data only)
pytest -m d1                               # failing tests = what you must implement on day 1
export DSP_MILESTONE=d1
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part isolated --states 2 3 5 --mixes 1 2
python -m speech.run_classical_baseline --fsdd data/fsdd/recordings --part plot
```
Live demo (day 3, on your laptop): `python -m asr_demo.server` then open http://localhost:8000.
On Snowflake: see `snowflake/notebook_bootstrap.py` (first cell of your notebook).

| Folder | Content |
|---|---|
| `common/` | measurement harness (`bench.py`) and Pareto tools (`pareto.py`, one TODO) |
| `speech/` | pipeline stages; `TODO(mX)` stubs are yours, everything else is provided |
| `asr_demo/` | live demo: server, browser page, backend registry, replay tool |
| `tests/public/` | daily contracts (`pytest -m d2`; demo: `pytest -m stretch`) |
| `scripts/` | data download |
| `results/` | your measurements, figures, `pareto.yaml` |
