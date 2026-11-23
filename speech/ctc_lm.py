"""Stage 2 of the speech track: the same acoustic model, with and without language information.

    acoustic only : CTC model (wav2vec2-style) + greedy argmax decoding
    + search      : CTC prefix beam search, no LM (isolates the effect of search)
    + n-gram LM   : beam search with KenLM, sweeping alpha (LM weight) and beta (word bonus)
                    -- the same two knobs as the GMM-HMM word-loop decoder of stage 1.

Install:  pip install --no-deps pyctcdecode   (its metadata pins numpy<2 and would DOWNGRADE numpy,
          breaking librosa/scipy; it works fine with numpy 2)
          pip install https://github.com/kpu/kenlm/archive/master.zip
LM:       LibriSpeech n-gram ARPA files from https://www.openslr.org/11 (the pruned 3-gram is small),
          or write_arpa_bigram() below for a toy LM trained on any text.

    python -m speech.ctc_lm --selftest                      # synthetic posteriors, runs on CPU in seconds
    python -m speech.ctc_lm --librispeech /data/LibriSpeech --model facebook/wav2vec2-base-100h \
        --arpa /data/lm/3-gram.pruned.1e-7.arpa                                       # UNTESTED here
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
def greedy_ctc(logprobs: np.ndarray, labels: list[str], blank: int = 0) -> str:
    ids = np.argmax(logprobs, axis=1)                       # 1) le symbole le plus probable à chaque trame
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:                        # 2) fusionner les répétitions, 3) enlever les blanks
            out.append(labels[i])
        prev = i                                            #    (un blank entre deux 'l' permet de garder "ll")
    text = "".join(" " if c == "|" else c for c in out)     # 4) '|' = séparateur de mots
    return " ".join(text.split())                           # espaces multiples / en bord -> un seul espace


def labels_from_processor(processor) -> tuple[list[str], int]:
    """pyctcdecode wants labels ordered by id, blank as '' and word delimiter as ' '."""
    vocab = processor.tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, i in vocab.items():
        labels[i] = tok
    blank = processor.tokenizer.pad_token_id
    labels = ["" if i == blank else (" " if t == "|" else t.lower()) for i, t in enumerate(labels)]
    return labels, blank


def write_arpa_bigram(sentences: list[list[str]], path: Path, discount: float = 0.75):
    """Interpolated absolute-discounting bigram in ARPA format (exact backoff semantics)."""
    uni, bi = Counter(), Counter()
    for s in sentences:
        toks = ["<s>"] + s + ["</s>"]
        uni.update(toks)
        bi.update(zip(toks[:-1], toks[1:]))
    total = sum(v for k, v in uni.items() if k != "<s>")
    p_uni = {w: c / total for w, c in uni.items() if w != "<s>"}
    p_uni["<unk>"] = 1e-6
    follow, n_types = Counter(), Counter()  # total successor tokens / distinct successor types
    for (u, _), c in bi.items():
        follow[u] += c
        n_types[u] += 1
    lines_bi, bo = [], {}
    for u in uni:
        if u == "</s>":
            continue
        lam = discount * n_types[u] / follow[u] if follow[u] else 1.0
        bo[u] = lam
    for (u, w), c in bi.items():
        p = max(c - discount, 0) / follow[u] + bo[u] * p_uni[w]
        lines_bi.append(f"{math.log10(p):.6f}\t{u} {w}")
    with open(path, "w") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={len(p_uni) + 1}\nngram 2={len(lines_bi)}\n\n\\1-grams:\n")
        f.write(f"-99\t<s>\t{math.log10(bo['<s>']):.6f}\n")
        for w, p in p_uni.items():
            b = f"\t{math.log10(bo[w]):.6f}" if w in bo else ""
            f.write(f"{math.log10(p):.6f}\t{w}{b}\n")
        f.write("\n\\2-grams:\n" + "\n".join(lines_bi) + "\n\n\\end\\\n")


def make_decoder(labels, arpa: str | None, alpha: float, beta: float):
    from pyctcdecode import build_ctcdecoder
    return build_ctcdecoder(labels, kenlm_model_path=arpa, alpha=alpha, beta=beta)


def sweep(logprobs_list, refs, labels, blank, arpa, alphas, betas, beams, unigrams=None):
    """Returns rows: decoder, alpha, beta, beam, WER, ms per utterance."""
    from speech.data import wer
    n = len(logprobs_list)
    rows = []

    def run(name, decode_fn, alpha, beta, beam):
        t0 = time.perf_counter()
        hyps = [decode_fn(lp) for lp in logprobs_list]
        ms = (time.perf_counter() - t0) / n * 1000
        rows.append(dict(decoder=name, alpha=alpha, beta=beta, beam=beam, wer=float(wer(refs, hyps)), ms_per_utt=ms))

    # 1) Glouton : le modèle acoustique seul (aucune recherche, aucune connaissance des mots)
    run("greedy", lambda lp: greedy_ctc(lp, labels, blank), None, None, 1)

    # 2) Beam search SANS LM : isole l'effet de la recherche (garder plusieurs hypothèses)
    dec = make_decoder(labels, None, 0.0, 0.0)
    for beam in beams:
        run("beam, no LM", lambda lp, b=beam: dec.decode(lp, beam_width=b), None, None, beam)

    # 3) Beam search + LM n-gramme (KenLM) : isole l'effet de la connaissance de la langue
    #    score = log P_ctc + alpha * log P_LM + beta * nb_mots   (alpha ~ lambda, beta ~ pénalité du jour 1)
    if arpa:
        from pyctcdecode import build_ctcdecoder
        name = f"beam + LM ({Path(arpa).stem})"
        dec_lm = build_ctcdecoder(labels, kenlm_model_path=str(arpa), unigrams=unigrams, alpha=alphas[0], beta=betas[0])
        for alpha in alphas:
            for beta in betas:
                dec_lm.reset_params(alpha=alpha, beta=beta)  # on change les poids sans recharger le LM (lent)
                for beam in beams:
                    run(name, lambda lp, b=beam: dec_lm.decode(lp, beam_width=b), alpha, beta, beam)
    return rows


# --------------------------------------------------------------------------- #
def synthetic_posteriors(text: str, labels: list[str], rng, p_ok: float = 0.6, sharp: float = 3.0):
    """Fake CTC output: each char held 2-4 frames, blanks in between; with prob 1-p_ok the char's
    frames favour a random wrong char. Lets students test decoders without any acoustic model."""
    idx = {c: i for i, c in enumerate(labels)}
    seq = []
    for c in text:
        k = idx[c]
        n = int(rng.integers(2, 5))
        wrong = rng.random() > p_ok
        alt = int(rng.integers(2, len(labels))) if wrong else k
        for _ in range(n):
            seq.append((k, alt))
        seq.append((0, 0))
    V = len(labels)
    lp = []
    for k, alt in seq:
        z = rng.normal(0, 1, V)
        z[alt] += sharp
        z[k] += sharp * (0.85 if alt != k else 1.0)  # the right char is still close -> LM can recover it
        lp.append(z - np.logaddexp.reduce(z))
    return np.array(lp, dtype=np.float32)


def selftest(out: Path):
    import pandas as pd
    rng = np.random.default_rng(0)
    vocab = ("the cat dog bird sat ran flew on under over a red big small mat tree house roof "
             "garden near in").split()
    nouns = ["cat", "dog", "bird", "mat", "tree", "house", "roof", "garden"]
    def sent():  # noqa: E306
        return ["the", rng.choice(["red", "big", "small"]), rng.choice(nouns[:3]),
                rng.choice(["sat", "ran", "flew"]), rng.choice(["on", "under", "over", "near", "in"]),
                "the", rng.choice(nouns[3:])]
    corpus = [[str(w) for w in sent()] for _ in range(3000)]
    out.mkdir(parents=True, exist_ok=True)
    arpa = out / "toy_bigram.arpa"
    write_arpa_bigram(corpus, arpa)
    labels = ["", " "] + sorted(set("".join(vocab)))
    tests = [" ".join(str(w) for w in sent()) for _ in range(60)]
    rows = []
    for p_ok in (0.9, 0.75, 0.6):
        lps = [synthetic_posteriors(t, labels, rng, p_ok) for t in tests]
        for r in sweep(lps, tests, labels, 0, str(arpa), alphas=[0.5, 1.0, 2.0], betas=[0.0, 1.5], beams=[1, 10, 50]):
            rows.append(dict(char_acc_sim=p_ok, **r))
    df = pd.DataFrame(rows)
    df.to_csv(out / "ctc_lm_selftest.csv", index=False)
    best = df.loc[df.groupby(["char_acc_sim", "decoder"]).wer.idxmin()]
    print(best[["char_acc_sim", "decoder", "alpha", "beta", "beam", "wer", "ms_per_utt"]].round(3).to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
def librispeech_logits(root, model_id, n, device):
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from speech.data import load_librispeech
    proc = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    data = load_librispeech(root, n=n)
    lps = []
    with torch.inference_mode():
        for d in data:
            x = proc(d["audio"], sampling_rate=d["sr"], return_tensors="pt").input_values.to(device)
            lps.append(torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy())
    labels, blank = labels_from_processor(proc)
    return lps, [d["text"].lower() for d in data], labels, blank


def plot_ctc(csv_path: Path, out_dir: Path = Path("results/figures")):
    """Front de Pareto WER vs temps de décodage pour un CSV produit par sweep (ajouté par le groupe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from common.pareto import plot_pareto
    df = pd.read_csv(csv_path)
    df["label"] = [d if d == "greedy" else (f"beam={b}" if "no LM" in d else f"a={a},b={bb},beam={b}")
                   for d, a, bb, b in zip(df.decoder, df.alpha, df.beta, df.beam)]
    df["famille"] = df.decoder.str.replace(r" \(.*\)", "", regex=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pareto(df, "ms_per_utt", "wer", "min", "min", label="label", hue="famille", ax=ax, logx=True,
                title=f"CTC decoding: WER vs decoding time ({Path(csv_path).stem})", annotate=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(csv_path).stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("figure ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plot", type=Path, help="CSV de résultats à tracer (ex. results/speech_ctc_lm/ctc_lm_wav2vec2-base-100h.csv)")
    ap.add_argument("--librispeech", type=Path)
    ap.add_argument("--model", default="facebook/wav2vec2-base-100h")
    ap.add_argument("--arpa")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/speech_ctc_lm"))
    ap.add_argument("--jsonl", type=Path, default=Path("results/results.jsonl"))
    args = ap.parse_args()
    if args.plot:
        return plot_ctc(args.plot)
    if args.selftest:
        return selftest(args.out)
    import pandas as pd
    lps, refs, labels, blank = librispeech_logits(args.librispeech, args.model, args.n, args.device)
    rows = sweep(lps, refs, labels, blank, args.arpa, alphas=[0.3, 0.5, 1.0, 2.0], betas=[0.0, 1.0, 2.0],
                 beams=[10, 50, 100])
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / f"ctc_lm_{args.model.split('/')[-1]}.csv", index=False)
    from common.bench import BenchRecord, log_record
    for r in rows:
        log_record(BenchRecord("ctc_decoding", f"{args.model}|{r['decoder']}", dict(alpha=r["alpha"], beta=r["beta"], beam=r["beam"]),
                               dict(wer=r["wer"], ms_per_utt=r["ms_per_utt"])), args.jsonl)


if __name__ == "__main__":
    main()
