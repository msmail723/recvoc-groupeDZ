"""Classical acoustic-only baseline: MFCC features + one left-to-right GMM-HMM per word,
and a word-loop Viterbi decoder into which a bigram language model can be plugged.

Why per-WORD models rather than per-PHONE models?  A phone-level recogniser needs
(i) frame-level phone alignments or embedded training, (ii) a pronunciation lexicon and
(iii) a decoding graph (phone loop -> words). That is the Kaldi/HTK stack and does not fit
in a 25 h project.  Word models on a small vocabulary keep every concept (MFCC, GMM
emissions, HMM topology, Viterbi, LM integration, LM weight, insertion penalty) visible
in ~200 lines of numpy, which is the pedagogical point of the baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import librosa
import numpy as np
from hmmlearn.hmm import GMMHMM


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13, win_ms: float = 25, hop_ms: float = 10,
                  deltas: bool = True, cmvn: bool = False) -> np.ndarray:
    """(T, D) MFCC (+delta, +delta-delta). cmvn=True -> per-utterance normalisation.
    For connected-word decoding prefer a GLOBAL normaliser (see Normaliser) so that isolated
    training words and long test utterances share the same feature statistics."""
    n_fft = int(round(sr * win_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop,
                             win_length=n_fft, n_mels=40, fmax=sr / 2)
    feats = [m]
    if deltas and m.shape[1] >= 9:
        feats += [librosa.feature.delta(m, order=1), librosa.feature.delta(m, order=2)]
    elif deltas:  # very short segment: pad deltas with zeros
        feats += [np.zeros_like(m), np.zeros_like(m)]
    f = np.vstack(feats).T
    if cmvn:
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    return f.astype(np.float64)


class Normaliser:
    """Global mean/variance normalisation estimated on the training set."""

    def fit(self, feats: list[np.ndarray]):
        X = np.concatenate(feats)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return (f - self.mu) / self.sd


def add_noise(y: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None:
        return y
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return y + rng.normal(0, np.sqrt(p_noise), size=y.shape)


# --------------------------------------------------------------------------- #
# Word models
# --------------------------------------------------------------------------- #
def left_to_right_hmm(n_states: int, n_mix: int, n_iter: int = 15, seed: int = 0) -> GMMHMM:
    """Bakis topology: each state loops or moves to the next one. EM keeps zeros at zero,
    so the topology is preserved during training."""
    m = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=n_iter,
               random_state=seed, init_params="mcw", params="stmcw", min_covar=1e-3)
    m.startprob_ = np.eye(n_states)[0]
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            A[i, i] = 1.0
        else:
            A[i, i], A[i, i + 1] = 0.6, 0.4
    m.transmat_ = A
    return m


@dataclass
class WordModels:
    models: dict  # word -> GMMHMM
    n_states: int
    n_mix: int
    train_time_s: float

    @property
    def n_params(self) -> int:
        n = 0
        for m in self.models.values():
            S, M, D = m.n_components, m.n_mix, m.n_features
            n += S * M * (2 * D + 1) + S * S  # means, diag covars, weights, transitions
        return n


def uniform_segmentation_init(m: GMMHMM, seqs: list[np.ndarray], seed: int = 0):
    """HTK-style flat start: cut every training sequence into n_states equal chunks, fit one
    GMM per state on its chunk frames. Much more robust than k-means on pooled frames
    (which often leaves a state with an empty mixture component -> NaNs)."""
    from sklearn.mixture import GaussianMixture
    S, M = m.n_components, m.n_mix
    D = seqs[0].shape[1]
    per_state = [[] for _ in range(S)]
    for q in seqs:
        for s, chunk in enumerate(np.array_split(q, S)):
            if len(chunk):
                per_state[s].append(chunk)
    means = np.zeros((S, M, D))
    covars = np.ones((S, M, D))
    weights = np.full((S, M), 1.0 / M)
    for s in range(S):
        X = np.concatenate(per_state[s])
        k = min(M, max(1, len(X) // 10))
        g = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=seed, max_iter=50).fit(X)
        means[s, :k], covars[s, :k], weights[s, :k] = g.means_, g.covariances_, g.weights_
        if k < M:  # pad unused components with perturbed copies
            for j in range(k, M):
                means[s, j] = g.means_[0] + 0.1 * np.random.default_rng(seed + j).standard_normal(D)
                covars[s, j] = g.covariances_[0]
            weights[s] = 1.0 / M
    m.means_, m.covars_, m.weights_ = means, covars, weights / weights.sum(1, keepdims=True)
    m.n_features = D


def train_word_models(train: dict[str, list[np.ndarray]], n_states: int = 5, n_mix: int = 2,
                      n_iter: int = 8, seed: int = 0) -> WordModels:
    t0 = time.perf_counter()
    models = {}
    for w, seqs in train.items():
        # silence is acoustically homogeneous: many mixtures collapse onto it (degenerate covariances)
        s, k = (n_states, n_mix) if w != "sil" else (3, min(n_mix, 2))
        m = left_to_right_hmm(s, k, n_iter, seed)
        m.init_params = ""
        uniform_segmentation_init(m, seqs, seed)
        X = np.concatenate(seqs)
        m.fit(X, lengths=[len(q) for q in seqs])
        _repair_dead_components(m)
        if not np.all(np.isfinite(m.means_)) or not np.isfinite(m.transmat_).all():
            raise RuntimeError(f"training diverged for word {w}")
        models[w] = m
    return WordModels(models, n_states, n_mix, time.perf_counter() - t0)


def _repair_dead_components(m: GMMHMM):
    """Correctif (ajouté par le groupe) : pendant EM, une gaussienne peut ne plus recevoir aucune trame
    (poids 0). Ses variances deviennent alors NaN (0/0), et ce NaN contamine TOUTES les log-vraisemblances
    de l'état -> Viterbi ne peut plus décider (observé sur le modèle 'sil' : WER 100 %).
    Une composante de poids 0 ne contribue pas au mélange : on lui donne des valeurs inoffensives
    (poids exactement 0, moyenne/variance d'une composante saine), ce qui ne change pas le modèle."""
    bad = ~np.isfinite(m.covars_).all(axis=-1) | ~np.isfinite(m.means_).all(axis=-1) | (m.weights_ <= 0)
    if not bad.any():
        return
    for s, k in zip(*np.nonzero(bad)):
        good = np.nonzero(~bad[s])[0]
        ref = good[0] if len(good) else None
        m.means_[s, k] = m.means_[s, ref] if ref is not None else 0.0
        m.covars_[s, k] = m.covars_[s, ref] if ref is not None else 1.0
        m.weights_[s, k] = 0.0
    dead_states = m.weights_.sum(axis=1) == 0
    m.weights_[dead_states] = 1.0 / m.n_mix  # cas extrême : aucune composante saine dans l'état
    m.weights_ = m.weights_ / m.weights_.sum(axis=1, keepdims=True)


def recognise_isolated(wm: WordModels, feats: np.ndarray, vocab: list[str]) -> str:
    # On demande sa note (log-vraisemblance) à chaque modèle de mot, et on garde le mot qui a la meilleure.
    # .score(feats) = log P(feats | modèle du mot), calculé par l'algorithme forward de hmmlearn.
    scores = {w: wm.models[w].score(feats) for w in vocab}
    return max(scores, key=scores.get)


# --------------------------------------------------------------------------- #
# Connected-word decoding: one big HMM = loop over all word HMMs.
# Transition (end of word u) -> (start of word v) carries  lm_weight * log P(v|u) + insertion penalty.
# Because the source state identifies u, a bigram needs no state expansion.
# --------------------------------------------------------------------------- #
class WordLoopDecoder:
    def __init__(self, wm: WordModels, vocab: list[str], bigram: dict | None = None,
                 lm_weight: float = 1.0, word_penalty: float = 0.0, exit_logp: float = np.log(0.5)):
        self.vocab = vocab
        self.wm = wm
        self.offsets, self.sizes = {}, {}
        o = 0
        for w in vocab:
            self.offsets[w], self.sizes[w] = o, wm.models[w].n_components
            o += self.sizes[w]
        self.S = o
        self.word_of_state = np.empty(self.S, dtype=object)
        for w in vocab:
            self.word_of_state[self.offsets[w]:self.offsets[w] + self.sizes[w]] = w
        V = len(vocab)
        if bigram is None:  # acoustic-only: uniform "LM"
            bigram = {u: {v: 1.0 / V for v in vocab} for u in ["<s>"] + vocab}
        self.bigram = bigram
        self.lm_weight, self.word_penalty, self.exit_logp = lm_weight, word_penalty, exit_logp
        self._build()

    def _lm(self, u, v):
        return self.lm_weight * np.log(max(self.bigram[u].get(v, 1e-12), 1e-12)) + self.word_penalty

    def _build(self):
        S = self.S
        NEG = -np.inf
        # logA[i, j] = log-score pour passer de l'état i à l'état j (-inf = transition interdite)
        self.logA = np.full((S, S), NEG)
        # is_entry[i, j] = True si la transition i -> j fait ENTRER dans un nouveau mot
        self.is_entry = np.zeros((S, S), dtype=bool)
        # log_start[j] = log-score pour commencer l'énoncé dans l'état j
        self.log_start = np.full(S, NEG)
        # is_final[j] = True si j est le dernier état d'un mot (on a le droit de finir l'énoncé là)
        self.is_final = np.zeros(S, dtype=bool)

        first = {w: self.offsets[w] for w in self.vocab}                        # 1er état de chaque mot
        last = {w: self.offsets[w] + self.sizes[w] - 1 for w in self.vocab}     # dernier état de chaque mot
        log_stay_last = np.log1p(-np.exp(self.exit_logp)) if self.exit_logp < 0 else NEG  # log(1 - P(sortie))

        with np.errstate(divide="ignore"):  # log(0) = -inf est voulu (transitions interdites)
            # 1) Transitions À L'INTÉRIEUR de chaque mot : on recopie la matrice du HMM du mot
            for w in self.vocab:
                o, n = self.offsets[w], self.sizes[w]
                block = np.log(self.wm.models[w].transmat_)
                # dans le dernier état, on ne reste qu'avec la proba 1 - P(sortie) : le reste part vers le mot suivant
                block[n - 1, n - 1] = log_stay_last
                self.logA[o:o + n, o:o + n] = block
                self.is_final[last[w]] = True

            # 2) Transitions ENTRE mots : fin du mot u -> début du mot v
            #    score = log P(sortir de u) + lm_weight * log P(v|u) + word_penalty   (_lm fait les 2 derniers termes)
            for u in self.vocab:
                for v in self.vocab:
                    i, j = last[u], first[v]
                    s = self.exit_logp + self._lm(u, v)
                    if s > self.logA[i, j]:  # (seulement utile si un mot avait 1 seul état)
                        self.logA[i, j] = s
                        self.is_entry[i, j] = True

            # 3) Départ : on commence forcément au début d'un mot v, avec le score du LM P(v | <s>)
            for v in self.vocab:
                self.log_start[first[v]] = self._lm("<s>", v)

    def emissions(self, feats: np.ndarray) -> np.ndarray:
        """(T, S) log b_s(o_t) concatenated over word models."""
        cols = [self.wm.models[w]._compute_log_likelihood(feats) for w in self.vocab]
        return np.hstack(cols)

    def decode(self, feats: np.ndarray, drop: tuple = ("sil",)) -> list[str]:
        return self.decode_emissions(self.emissions(feats), drop)

    def decode_emissions(self, B: np.ndarray, drop: tuple = ("sil",)) -> list[str]:
        """Viterbi on precomputed emissions (lets you sweep LM weights without recomputing GMMs)."""
        T, S = B.shape
        # delta[j] = meilleur score d'un chemin qui arrive dans l'état j à la trame courante
        # psi[t, j] = d'où venait ce meilleur chemin (la "flèche" pour remonter à la fin)
        psi = np.zeros((T, S), dtype=np.int64)

        # Trame 0 : score de départ + ressemblance de la trame 0 avec chaque état
        delta = self.log_start + B[0]

        # Trames suivantes : le tableau du cours, colonne par colonne
        for t in range(1, T):
            # cand[i, j] = score si on était en i à t-1 et qu'on passe en j
            cand = delta[:, None] + self.logA
            psi[t] = np.argmax(cand, axis=0)                     # meilleur état précédent pour chaque j
            delta = cand[psi[t], np.arange(S)] + B[t]            # meilleur score + ressemblance de la trame t

        # Fin : on doit finir dans le dernier état d'un mot
        end_scores = np.where(self.is_final, delta, -np.inf)
        state = int(np.argmax(end_scores))

        # Remonter les flèches de la dernière trame à la première
        path = np.empty(T, dtype=np.int64)
        path[-1] = state
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]

        # Lire les mots : un nouveau mot commence à t=0 et à chaque transition d'entrée
        words = [self.word_of_state[path[0]]]
        for t in range(1, T):
            if self.is_entry[path[t - 1], path[t]]:
                words.append(self.word_of_state[path[t]])
        return [w for w in words if w not in drop]


# --------------------------------------------------------------------------- #
# Bigram LM with add-k smoothing
# --------------------------------------------------------------------------- #
def train_bigram(sentences: list[list[str]], vocab: list[str], k: float = 0.1,
                 sil_token: str | None = "sil", p_sil: float = 0.15) -> dict:
    """Bigram add-k : P(v|u) = (count(u v) + k) / (count(u) + k|V|), avec <s> comme mot précédent du 1er mot.
    Silence : il peut apparaître après n'importe quel mot avec la proba fixe p_sil (les mots se partagent
    le reste, 1 - p_sil). Le silence ne fait pas partie du texte : après un silence on ne sait plus quel
    était le mot précédent, donc la ligne 'sil' utilise la distribution unigramme (fréquence de chaque mot)."""
    from collections import Counter
    V = len(vocab)
    pair = Counter()      # pair[(u, v)] = nombre de fois où v suit u
    hist = Counter()      # hist[u] = nombre de fois où u est suivi d'un mot
    uni = Counter()       # uni[v] = nombre de fois où v apparaît
    for s in sentences:
        prev = "<s>"
        for w in s:
            pair[(prev, w)] += 1
            hist[prev] += 1
            uni[w] += 1
            prev = w

    def addk_row(u):  # distribution add-k P(.|u) sur les mots du vocabulaire (somme = 1)
        return {v: (pair[(u, v)] + k) / (hist[u] + k * V) for v in vocab}

    rows = {u: addk_row(u) for u in ["<s>"] + list(vocab)}
    if sil_token is None:
        return rows

    n_uni = sum(uni.values())
    rows[sil_token] = {v: (uni[v] + k) / (n_uni + k * V) for v in vocab}  # après un silence : unigramme
    for u, row in rows.items():
        for v in vocab:
            row[v] *= 1 - p_sil          # les mots se partagent 1 - p_sil
        row[sil_token] = p_sil           # le silence prend p_sil  -> la ligne somme toujours à 1
    return rows
