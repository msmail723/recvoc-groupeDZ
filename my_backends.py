"""Register YOUR systems here: they appear in the demo's backend list (python -m asr_demo.server), in
`python -m asr_demo.replay --backends ...` and in `python -m speech.streaming_sim --models ...`.
Same contract as the provided ones (asr_demo/contract.py). Lecture L6, "Hijack the demo".

Group DZ systems (see DECISIONS.md):
  * student/ctc+3gram                 our day-2 decoder (D4): wav2vec2-base-100h + pruned 3-gram,
                                      alpha=0.5, beta=1, N_beam=10
  * student/w2v2-960h+3gram           our day-2 acoustic model choice (D3, wav2vec2-base-960h) with the same
                                      3-gram and the same (alpha, beta, N_beam), tuned on base-100h
  * student/local-agreement(...)      our day-3 streaming policy (LocalAgreement-2, 0.5 s step) around each
  * student/zipformer-20M+lead1s      the small streaming transducer with 1 s of silence first (L6, slide 8)
"""
from asr_demo.backends import CTCWithLM, SherpaTransducer, ZIPFORMER_20M, get, register
from speech.streaming_live import LocalAgreementBackend

ARPA = "lm/3-gram.pruned.1e-7.arpa"

# 1. Our day-2 decoder, with the values our sweep chose (DECISIONS.md, D4)
register("student/ctc+3gram", "wav2vec2-base-100h + 3-gram, our d2 choice (alpha=0.5, beta=1, N_beam=10)")(
    lambda: CTCWithLM("facebook/wav2vec2-base-100h", ARPA, alpha=0.5, beta=1.0, beam=10))

# 2. Our day-2 acoustic model (D3) with the same LM settings
register("student/w2v2-960h+3gram", "wav2vec2-base-960h + 3-gram (alpha=0.5, beta=1, N_beam=10)")(
    lambda: CTCWithLM("facebook/wav2vec2-base-960h", ARPA, alpha=0.5, beta=1.0, beam=10))

# 3. Our day-3 streaming policy around them (speech/streaming_live.py)
register("student/local-agreement(ctc+3gram)", "LocalAgreement-2, 0.5 s step, around student/ctc+3gram")(
    lambda: LocalAgreementBackend(get("student/ctc+3gram"), step_s=0.5))
register("student/local-agreement(w2v2-960h+3gram)", "LocalAgreement-2, 0.5 s step, around student/w2v2-960h+3gram")(
    lambda: LocalAgreementBackend(get("student/w2v2-960h+3gram"), step_s=0.5))

# 4. The small streaming transducer with silence before the speech (it loses first words without it)
register("student/zipformer-20M+lead1s", "20M zipformer, 1 s of silence first")(
    lambda: SherpaTransducer(ZIPFORMER_20M, int8=True, lead_s=1.0))
