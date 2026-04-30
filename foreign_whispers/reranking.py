"""Deterministic failure analysis and translation re-ranking.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function uses a hybrid
strategy: rule-based Spanish contractions (tier 1), MarianMT
re-translation (tier 2), and sentence-boundary truncation (tier 3).
"""

import dataclasses
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Duration heuristic
# ---------------------------------------------------------------------------
_CHARS_PER_SECOND = 15.0  # Spanish TTS, empirically ~15 chars/second

# ---------------------------------------------------------------------------
# Rule-based Spanish phrase contractions  (verbose phrase → concise form)
# Each entry is (compiled_regex, replacement_string).
# Replacements that are "" remove the phrase entirely (discourse markers).
# ---------------------------------------------------------------------------
_PHRASE_CONTRACTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\ben este momento\b", re.I | re.U), "ahora"),
    (re.compile(r"\ben la actualidad\b", re.I | re.U), "hoy"),
    (re.compile(r"\ba continuación\b", re.I | re.U), "luego"),
    (re.compile(r"\bcon respecto a\b", re.I | re.U), "sobre"),
    (re.compile(r"\bdebido a que\b", re.I | re.U), "porque"),
    (re.compile(r"\ba pesar de que\b", re.I | re.U), "aunque"),
    (re.compile(r"\ben el caso de que\b", re.I | re.U), "si"),
    (re.compile(r"\ben el momento en que\b", re.I | re.U), "cuando"),
    (re.compile(r"\ba causa de\b", re.I | re.U), "por"),
    (re.compile(r"\bde tal manera que\b", re.I | re.U), "así que"),
    (re.compile(r"\bcon el fin de\b", re.I | re.U), "para"),
    (re.compile(r"\ben lo que respecta a\b", re.I | re.U), "sobre"),
    (re.compile(r"\bhay que tener en cuenta que\b", re.I | re.U), ""),
    (re.compile(r"\bes importante destacar que\b", re.I | re.U), ""),
    (re.compile(r"\bcabe señalar que\b", re.I | re.U), ""),
    (re.compile(r"\bes necesario mencionar que\b", re.I | re.U), ""),
    (re.compile(r"\blo que es más importante\b", re.I | re.U), "además"),
    (re.compile(r"\bes decir\b", re.I | re.U), "o sea"),
    (re.compile(r"\bsin embargo\b", re.I | re.U), "pero"),
    (re.compile(r"\bno obstante\b", re.I | re.U), "pero"),
    (re.compile(r"\bpor otro lado\b", re.I | re.U), "además"),
    (re.compile(r"\bpor otra parte\b", re.I | re.U), "además"),
    (re.compile(r"\ben primer lugar\b", re.I | re.U), "primero"),
    (re.compile(r"\ben segundo lugar\b", re.I | re.U), "segundo"),
    (re.compile(r"\ben realidad\b", re.I | re.U), "realmente"),
    (re.compile(r"\bde todas formas\b", re.I | re.U), "igual"),
    (re.compile(r"\bpor supuesto\b", re.I | re.U), "claro"),
    (re.compile(r"\bnaturalmente\b", re.I | re.U), "claro"),
    (re.compile(r"\bobviamente\b", re.I | re.U), "claro"),
    (re.compile(r"\bevidente(?:mente)? que\b", re.I | re.U), ""),
    (re.compile(r"\bpor lo tanto\b", re.I | re.U), "así"),
    (re.compile(r"\bpor consiguiente\b", re.I | re.U), "así"),
    (re.compile(r"\ben consecuencia\b", re.I | re.U), "así"),
    (re.compile(r"\ba lo largo de\b", re.I | re.U), "durante"),
    (re.compile(r"\bdurante el transcurso de\b", re.I | re.U), "durante"),
    (re.compile(r"\bno solamente\b", re.I | re.U), "no solo"),
    (re.compile(r"\bnos encontramos ante\b", re.I | re.U), "hay"),
    (re.compile(r"\bse puede observar\b", re.I | re.U), "vemos"),
    (re.compile(r"\bse puede ver\b", re.I | re.U), "vemos"),
    (re.compile(r"\bhemos de\b", re.I | re.U), "debemos"),
    (re.compile(r"\bes posible que\b", re.I | re.U), "puede que"),
    (re.compile(r"\bpuede ser que\b", re.I | re.U), "puede que"),
]

# Discourse-marker filler words that can be stripped from sentence start
_FILLER_PREFIX = re.compile(
    r"^(?:(?:pues|bueno|bien|mira|oye|sabes|claro|eh|o\s+sea)[,\s]+)+",
    re.I | re.U,
)

# Trailing discourse markers that add no content
_FILLER_SUFFIX = re.compile(
    r"[,\s]+(?:¿no|verdad|¿eh|¿sabes)\??\s*$",
    re.I | re.U,
)

_MULTI_SPACE = re.compile(r"  +")

# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _bigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity of character bigrams — lightweight semantic proxy.

    Returns a value in [0, 1] where 1 means identical and 0 means no shared
    bigrams.  Requires no external dependencies.
    """
    def bigrams(s: str) -> set:
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()

    ba, bb = bigrams(a.lower()), bigrams(b.lower())
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def _candidate_score(
    text: str,
    target_duration_s: float,
    baseline_es: str,
    lambda_sem: float = 0.1,
) -> float:
    """Score a candidate translation — lower is better.

    Score = (predicted_duration − target_duration)² + λ × semantic_distance

    Duration is estimated with the alignment module's multi-feature OLS model
    when available, falling back to the crude chars-per-second heuristic.
    The semantic term uses character-bigram Jaccard distance so that candidates
    which drift far from the original meaning are lightly penalised.

    Args:
        text: Candidate translation text.
        target_duration_s: Desired TTS duration in seconds.
        baseline_es: Original baseline translation (used for semantic distance).
        lambda_sem: Weight for the semantic distance term (default 0.1).

    Returns:
        Non-negative float; lower means closer to the target duration while
        remaining semantically similar to the baseline.
    """
    try:
        from foreign_whispers.alignment import _estimate_duration
        predicted = _estimate_duration(text)
    except Exception:
        predicted = len(text) / _CHARS_PER_SECOND

    duration_err = (predicted - target_duration_s) ** 2
    sem_dist = 1.0 - _bigram_similarity(text, baseline_es)
    return duration_err + lambda_sem * sem_dist


# ---------------------------------------------------------------------------
# MarianMT lazy-loaded model cache (module-level, loaded on first use)
# ---------------------------------------------------------------------------
_marian_model: Optional[object] = None
_marian_tokenizer: Optional[object] = None


def _apply_spanish_rules(text: str) -> str:
    """Apply lexical contractions to produce a shorter Spanish text."""
    result = text
    for pattern, replacement in _PHRASE_CONTRACTIONS:
        result = pattern.sub(replacement, result)
    result = _FILLER_PREFIX.sub("", result)
    result = _FILLER_SUFFIX.sub("", result)
    result = _MULTI_SPACE.sub(" ", result).strip()
    if result:
        result = result[0].upper() + result[1:]
    return result


def _marian_translate(source_text: str) -> Optional[str]:
    """Translate *source_text* (EN→ES) via MarianMT; returns None on failure."""
    global _marian_model, _marian_tokenizer
    try:
        if _marian_model is None:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore
            _model_name = "Helsinki-NLP/opus-mt-en-es"
            logger.info("Loading MarianMT %s (first call)", _model_name)
            _marian_tokenizer = MarianTokenizer.from_pretrained(_model_name)
            _marian_model = MarianMTModel.from_pretrained(_model_name)
            _marian_model.eval()  # type: ignore[union-attr]
        tokens = _marian_tokenizer(  # type: ignore[operator]
            [source_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        output_ids = _marian_model.generate(**tokens)  # type: ignore[union-attr]
        return _marian_tokenizer.decode(  # type: ignore[union-attr]
            output_ids[0], skip_special_tokens=True
        )
    except Exception as exc:
        logger.debug("MarianMT unavailable: %s", exc)
        return None


def _truncate_to_budget(text: str, budget_chars: float) -> tuple[Optional[str], str]:
    """Trim *text* to fit *budget_chars* by greedily accumulating fragments.

    Tries, in order: sentence boundary → clause boundary → word boundary.
    Returns (result, rationale) where result is None if the text already fits
    or cannot be shortened further.
    """
    if len(text) <= budget_chars:
        return None, ""

    # Sentence-level: accumulate full sentences from the left
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        accumulated = ""
        for sent in sentences:
            trial = (accumulated + " " + sent).strip() if accumulated else sent
            if len(trial) <= budget_chars:
                accumulated = trial
            else:
                break
        if accumulated and len(accumulated) < len(text):
            return accumulated, "sentence-boundary truncation"

    # Clause-level: accumulate comma/semicolon clauses from the left
    clauses = re.split(r"[,;]\s*", text)
    if len(clauses) > 1:
        accumulated = ""
        for clause in clauses:
            sep = ", " if accumulated else ""
            trial = accumulated + sep + clause
            if len(trial) <= budget_chars:
                accumulated = trial
            else:
                break
        if accumulated and len(accumulated) < len(text):
            return accumulated, "clause-boundary truncation"

    # Word-boundary hard truncation
    words = text.split()
    kept, count = [], 0
    for word in words:
        cost = len(word) + (1 if kept else 0)
        if count + cost > budget_chars:
            break
        kept.append(word)
        count += cost
    if kept and len(" ".join(kept)) < len(text):
        return " ".join(kept), "word-boundary truncation"

    return None, ""


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
        duration_score: Combined score used for ranking — lower is better.
            Computed as ``(predicted_duration − target_duration)² + λ × semantic_distance``.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""
    duration_score: float = 0.0


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        List of ``TranslationCandidate`` objects, sorted shortest first.
        All candidates are strictly shorter than *baseline_es*.
        Returns an empty list only if *baseline_es* already fits within
        the budget or no shorter form can be produced.
    """
    budget_chars = target_duration_s * _CHARS_PER_SECOND
    logger.info(
        "get_shorter_translations: %.1fs budget (%.0f chars), baseline %d chars",
        target_duration_s,
        budget_chars,
        len(baseline_es),
    )

    seen: set[str] = set()
    candidates: list[TranslationCandidate] = []

    def _add(text: str, rationale: str) -> None:
        text = text.strip()
        # Only keep candidates that are strictly shorter than the baseline
        if not text or text in seen or len(text) >= len(baseline_es):
            return
        seen.add(text)
        candidates.append(TranslationCandidate(
            text=text,
            char_count=len(text),
            brevity_rationale=rationale,
            duration_score=_candidate_score(text, target_duration_s, baseline_es),
        ))

    # ------------------------------------------------------------------
    # Tier 1: rule-based phrase contractions on the baseline
    # ------------------------------------------------------------------
    rule_text = _apply_spanish_rules(baseline_es)
    _add(rule_text, "rule-based phrase contractions")

    # ------------------------------------------------------------------
    # Tier 2: MarianMT re-translation of the source text
    # ------------------------------------------------------------------
    marian_text = _marian_translate(source_text)
    if marian_text:
        _add(marian_text, "MarianMT alternative translation")
        # Also apply rules to the MarianMT output for a combined candidate
        marian_rule_text = _apply_spanish_rules(marian_text)
        _add(marian_rule_text, "MarianMT + rule contractions")

    # ------------------------------------------------------------------
    # Tier 3: sentence/clause/word-boundary truncation (guaranteed fallback)
    # Operate on whichever existing candidate is currently shortest, or on
    # the baseline if no candidate was produced yet.
    # ------------------------------------------------------------------
    ref_for_trunc = (
        min(candidates, key=lambda c: c.char_count).text
        if candidates
        else baseline_es
    )
    truncated, rationale = _truncate_to_budget(ref_for_trunc, budget_chars)
    if truncated:
        _add(truncated, rationale)
    elif not candidates:
        # Nothing helped; produce at minimum a word-boundary cut of the baseline
        hard_cut, hard_rationale = _truncate_to_budget(baseline_es, budget_chars)
        if hard_cut:
            _add(hard_cut, hard_rationale)

    return sorted(candidates, key=lambda c: c.duration_score)
