from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from src.stt.ngram_lm import NGramLM


@dataclass
class BeamHypothesis:
    score: float
    tokens: List[int]
    text: str


def _collapse_ctc_tokens(tokens: List[int], blank_id: int) -> List[int]:
    out: List[int] = []
    prev = None
    for t in tokens:
        if t == blank_id:
            prev = None
            continue
        if t != prev:
            out.append(t)
            prev = t
    return out


def constrained_ctc_beam_search(
    log_probs: torch.Tensor,
    vocab: List[str],
    blank_id: int,
    lm: NGramLM | None,
    beam_size: int = 20,
    lm_weight: float = 1.0,
    bias_terms: List[str] | None = None,
    bias_weight: float = 1.0,
) -> str:
    bias_terms = [x.lower() for x in (bias_terms or [])]
    beams = [BeamHypothesis(score=0.0, tokens=[], text="")]

    for t in range(log_probs.size(0)):
        frame = log_probs[t]
        topk_scores, topk_ids = torch.topk(frame, k=min(beam_size, frame.numel()))

        candidates: List[BeamHypothesis] = []
        for hyp in beams:
            for score, token_id in zip(topk_scores.tolist(), topk_ids.tolist()):
                new_tokens = hyp.tokens + [token_id]
                collapsed = _collapse_ctc_tokens(new_tokens, blank_id)
                chars = [vocab[idx] for idx in collapsed if idx < len(vocab)]
                text = "".join(chars).replace("|", " ").strip()

                lm_bonus = 0.0
                if lm is not None and text:
                    words = text.split()
                    if words:
                        history = words[:-1]
                        lm_bonus = lm_weight * lm.score_next(history, words[-1])

                bias_bonus = 0.0
                text_l = text.lower()
                for term in bias_terms:
                    if term in text_l:
                        bias_bonus += bias_weight

                candidates.append(
                    BeamHypothesis(
                        score=hyp.score + score + lm_bonus + bias_bonus,
                        tokens=new_tokens,
                        text=text,
                    )
                )

        candidates.sort(key=lambda x: x.score, reverse=True)
        beams = candidates[:beam_size]

    return beams[0].text if beams else ""
