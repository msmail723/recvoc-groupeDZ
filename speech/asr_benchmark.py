"""Offline ASR Pareto front on LibriSpeech: WER vs real-time factor, latency, energy, size.

    python -m speech.asr_benchmark --librispeech /data/LibriSpeech --subset test-clean --n 200 \
        --models openai/whisper-tiny openai/whisper-small facebook/wav2vec2-base-960h --batch 1 16
    python -m speech.asr_benchmark --plot results/asr_bench.jsonl

batch=1 measures LATENCY (interactive use); batch=16 measures THROUGHPUT (offline transcription).
The two give different fronts -- one of the key messages of the project.
UNTESTED in the authoring sandbox (no model downloads possible there). Verify model ids on the Hub;
some families (e.g. Moonshine, Parakeet) may need a recent transformers or their own library.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common.bench import (BenchRecord, cuda_sync, energy_scope, log_record, peak_memory_mb, reset_peak_memory,
                          summarize_latencies)
from speech.data import load_librispeech, wer

DEFAULT_MODELS = [
    "openai/whisper-tiny.en", "openai/whisper-base.en", "openai/whisper-small.en", "openai/whisper-medium.en",
    "openai/whisper-large-v3-turbo", "distil-whisper/distil-large-v3",
    "facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h-lv60-self",
    "UsefulSensors/moonshine-tiny", "UsefulSensors/moonshine-base",
]


def bench_model(model_id, data, batch, device, fp16):
    """Benchmark one HF ASR model on `data` (list of dicts audio/sr/text).
    Returns (metrics, hypotheses). Latency = one call of the pipeline on one batch (batch=1 -> per utterance)."""
    import torch
    from transformers import pipeline

    # 1) Choix du matériel et de la précision : fp16 seulement sur GPU (sur CPU c'est lent ou non supporté)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if (fp16 and use_cuda) else torch.float32

    # 2) Chargement du modèle (chronométré à part : on ne le paie qu'une fois, pas à chaque phrase)
    t0 = time.perf_counter()
    asr = pipeline("automatic-speech-recognition", model=model_id, device="cuda:0" if use_cuda else "cpu", dtype=dtype)
    cuda_sync()
    load_s = time.perf_counter() - t0

    # Certains tokenizers (ex. Moonshine) n'ont pas de symbole de remplissage (pad) : impossible de traiter un
    # paquet de phrases de longueurs différentes. Correctif suggéré par transformers : utiliser le symbole de fin (eos).
    tok = getattr(asr, "tokenizer", None)
    if batch > 1 and tok is not None and getattr(tok, "pad_token_id", 0) is None:
        eos = getattr(asr.model.config, "eos_token_id", None) or tok.eos_token_id
        tok.pad_token_id = eos
        if getattr(asr.model, "generation_config", None) is not None:
            asr.model.generation_config.pad_token_id = eos

    # Whisper multilingue : on force l'anglais (sinon il peut détecter une autre langue ou traduire)
    kw = {}
    if "whisper" in model_id and not model_id.endswith(".en"):
        kw["generate_kwargs"] = {"language": "en", "task": "transcribe"}

    # 3) Entrées au format du pipeline, découpées en paquets (batches) de taille `batch`
    inputs = [{"raw": np.asarray(d["audio"], dtype=np.float32), "sampling_rate": d["sr"]} for d in data]
    batches = [inputs[i:i + batch] for i in range(0, len(inputs), batch)]
    audio_s = sum(len(d["audio"]) / d["sr"] for d in data)

    hyps, lats = [], []
    res: dict = {}
    # ATTENTION : le pipeline HF modifie les dictionnaires qu'on lui donne (il retire la clé "raw").
    # On lui passe donc une copie à chaque appel, sinon le 2e passage (après l'échauffement) plante.
    def fresh(b):
        return [dict(x) for x in b]

    with torch.inference_mode():
        # 4) Échauffement : le 1er appel paie l'initialisation (allocation, compilation...) -> non chronométré
        asr(fresh(batches[0]), batch_size=batch, **kw)
        cuda_sync()
        reset_peak_memory()
        # 5) Mesure : on chronomètre chaque paquet ; cuda_sync() attend la vraie fin du calcul sur GPU
        with energy_scope(res):
            for b in batches:
                b = fresh(b)
                t1 = time.perf_counter()
                out = asr(b, batch_size=batch, **kw)
                cuda_sync()
                lats.append(time.perf_counter() - t1)
                hyps.extend(o["text"] for o in out)

    # 6) Les métriques
    compute_s = float(sum(lats))
    n_params = sum(p.numel() for p in asr.model.parameters())
    m = dict(
        wer=float(wer([d["text"] for d in data], hyps)),       # références et hypothèses normalisées (data.normalise)
        rtf=compute_s / audio_s,                                # temps de calcul / durée de l'audio
        throughput_audio_s_per_s=audio_s / compute_s,
        **summarize_latencies(lats),                            # lat_mean/p50/p90/p99_ms (par paquet)
        lat_per_utt_mean_ms=compute_s / len(data) * 1000,
        n_params_M=n_params / 1e6,
        load_s=load_s,
        audio_s=audio_s,
        compute_s=compute_s,
        device="cuda" if use_cuda else "cpu",
        dtype="fp16" if dtype == torch.float16 else "fp32",
        energy_j=res.get("energy_j"),
        energy_meter=res.get("energy_meter"),
        **peak_memory_mb(),
    )
    return m, hyps


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.bench import load_records
    from common.pareto import plot_pareto
    df = load_records(path)
    df["label"] = df.model.str.split("/").str[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, b in zip(axes, sorted(df.cfg_batch.unique())[:2]):
        plot_pareto(df[df.cfg_batch == b].reset_index(drop=True), "rtf", "wer", "min", "min", label="label",
                    ax=ax, logx=True, title=f"WER vs real-time factor, batch={b}")
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", type=Path)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/asr_bench.jsonl"))
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)
    data = load_librispeech(args.librispeech, args.subset, args.n)
    print(f"{len(data)} utterances, {sum(len(d['audio']) / d['sr'] for d in data) / 60:.1f} min of audio")
    for mid in args.models:
        for b in args.batch:
            try:
                m, _ = bench_model(mid, data, b, args.device, args.fp16)
            except Exception as e:
                print(f"[skip] {mid} batch={b}: {e}")
                continue
            log_record(BenchRecord("asr", mid, dict(subset=args.subset, batch=b, fp16=args.fp16, n=len(data)), m), args.out)
            print(mid, b, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
