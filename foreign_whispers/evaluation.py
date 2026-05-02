"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    _count_syllables,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein word-edit distance normalised by reference length (stdlib only)."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    row = list(range(m + 1))
    for i in range(1, n + 1):
        prev = row[:]
        row[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j], row[j - 1], prev[j - 1])
    return row[m] / n


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
    *,
    stt_transcripts: list[str] | None = None,
    semantic_scores: list[float] | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Scores four dimensions, each normalised to [0, 1] (1 = best):

    **timing** — duration accuracy, severe-stretch rate, cumulative drift.

    **intelligibility** — measures whether the audience can parse the speech.
    *Proxy mode* (default): models audibility from stretch factor and clipping.
    FAIL segments score 0; REQUEST_SHORTER segments are penalised by the fraction
    of audio that overflows its slot; stretched segments lose 0.3 points per unit
    of excess stretch.  *STT round-trip mode*: if *stt_transcripts* is supplied
    (one ASR hypothesis string per segment), the score is ``1 − mean_WER``.

    **semantic_fidelity** — measures meaning preservation.
    *Proxy mode* (default): EN→ES character-length ratio penalises translations
    that deviate sharply from the expected ×1.2 expansion factor — extreme
    compression signals semantic truncation.  *Embedding mode*: if
    *semantic_scores* is supplied (one pre-computed cosine similarity per
    segment, in [0, 1]), those values are averaged directly.

    **naturalness** — speaking-rate coefficient of variation (CV) across segments.
    Human speech has CV ≈ 0.3–0.5.  Penalty applies above CV = 0.3; CV ≥ 1.5 → 0.

    Overall score is a weighted composite:
        timing × 0.40 + intelligibility × 0.30 + semantic_fidelity × 0.20 + naturalness × 0.10

    Parameters
    ----------
    metrics:
        Output of ``compute_segment_metrics``.
    aligned:
        Output of ``global_align`` or ``global_align_dp``.
    align_report:
        Optional pre-computed ``clip_evaluation_report`` dict.  Computed
        internally if omitted (avoids redundant work when you already have it).
    stt_transcripts:
        Optional list of ASR hypothesis strings, one per segment in order.
        When provided, upgrades intelligibility from proxy to WER-based.
    semantic_scores:
        Optional list of pre-computed cosine similarity scores in [0, 1],
        one per segment.  When provided, upgrades semantic fidelity from
        length-ratio proxy to embedding-based.

    Returns
    -------
    dict with keys ``timing``, ``intelligibility``, ``semantic_fidelity``,
    ``naturalness``, and ``overall``.  Each sub-dict exposes the supporting
    metrics used to derive its ``score``.
    """
    if not metrics:
        _zero = {"score": 0.0}
        return {
            "timing":           {**_zero, "mean_abs_duration_error_s": 0.0,
                                 "pct_severe_stretch": 0.0, "total_cumulative_drift_s": 0.0},
            "intelligibility":  {**_zero, "method": "proxy"},
            "semantic_fidelity":{**_zero, "method": "length_ratio"},
            "naturalness":      {**_zero, "rate_cv": 0.0, "mean_syllables_per_s": 0.0},
            "overall":          0.0,
        }

    if align_report is None:
        align_report = clip_evaluation_report(metrics, aligned)

    n = len(metrics)

    # ── 1. Timing accuracy ────────────────────────────────────────────────────
    mae        = align_report["mean_abs_duration_error_s"]
    pct_severe = align_report["pct_severe_stretch"]
    total_drift = align_report["total_cumulative_drift_s"]

    dur_err_score  = max(0.0, 1.0 - mae / 2.0)               # 2 s MAE → 0
    severe_score   = 1.0 - pct_severe / 100.0
    drift_score    = max(0.0, 1.0 - abs(total_drift) / 60.0)  # 60 s total drift → 0

    timing_score = _stats.mean([dur_err_score, severe_score, drift_score])
    timing_dim = {
        "mean_abs_duration_error_s": mae,
        "duration_error_score":      round(dur_err_score, 3),
        "pct_severe_stretch":        pct_severe,
        "severe_stretch_score":      round(severe_score, 3),
        "total_cumulative_drift_s":  total_drift,
        "drift_score":               round(drift_score, 3),
        "score":                     round(timing_score, 3),
    }

    # ── 2. Intelligibility ────────────────────────────────────────────────────
    if stt_transcripts is not None and len(stt_transcripts) == n:
        wers = [
            _word_error_rate(m.translated_text, hyp)
            for m, hyp in zip(metrics, stt_transcripts)
        ]
        mean_wer    = _stats.mean(wers)
        intel_score = max(0.0, 1.0 - mean_wer)
        intel_dim   = {
            "method":   "wer",
            "mean_wer": round(mean_wer, 3),
            "score":    round(intel_score, 3),
        }
    else:
        per_seg = []
        for m, a in zip(metrics, aligned):
            if a.action == AlignAction.FAIL:
                # FAIL → silence; nothing heard
                per_seg.append(0.0)
            elif a.action == AlignAction.REQUEST_SHORTER:
                # Audio overflows its slot → tail clipped at slot boundary
                heard = min(1.0, m.source_duration_s / max(0.1, m.predicted_tts_s))
                per_seg.append(heard)
            else:
                # Time-stretching degrades quality: −0.3 per unit of excess stretch
                per_seg.append(max(0.0, 1.0 - 0.3 * (a.stretch_factor - 1.0)))
        intel_score = _stats.mean(per_seg)
        intel_dim   = {
            "method":    "proxy",
            "n_fail":    sum(1 for a in aligned if a.action == AlignAction.FAIL),
            "n_clipped": sum(1 for a in aligned if a.action == AlignAction.REQUEST_SHORTER),
            "score":     round(intel_score, 3),
        }

    # ── 3. Semantic fidelity ──────────────────────────────────────────────────
    if semantic_scores is not None and len(semantic_scores) == n:
        sem_score = _stats.mean(semantic_scores)
        sem_dim   = {
            "method":       "embedding",
            "mean_cosine":  round(sem_score, 3),
            "score":        round(sem_score, 3),
        }
    else:
        # EN→ES expansion ratio proxy.  Expected ratio ≈ 1.2 (Spanish is ~20% longer).
        # Deviation from 1.2 — normalised by 1.2 so ±100% deviation → score 0.
        ratios = [
            len(m.translated_text) / max(1, len(m.source_text))
            for m in metrics
        ]
        scores = [max(0.0, 1.0 - abs(r - 1.2) / 1.2) for r in ratios]
        sem_score = _stats.mean(scores)
        sem_dim   = {
            "method":            "length_ratio",
            "mean_length_ratio": round(_stats.mean(ratios), 3),
            "score":             round(sem_score, 3),
        }

    # ── 4. Naturalness ────────────────────────────────────────────────────────
    # Speaking rate = syllables per scheduled second.  Uses the same syllable
    # counter as the duration model for consistency.
    rates = [
        _count_syllables(a.text) / max(0.1, a.scheduled_end - a.scheduled_start)
        for a in aligned
    ]
    mean_rate = _stats.mean(rates) if rates else 0.0
    cv = (_stats.stdev(rates) / mean_rate) if len(rates) >= 2 and mean_rate > 0 else 0.0

    # Human speech has natural CV ≈ 0.3.  Penalise only above that floor.
    # CV = 1.5 → naturalness = 0; scale is 0.3–1.5 (range 1.2).
    naturalness_score = max(0.0, 1.0 - max(0.0, cv - 0.3) / 1.2)
    natural_dim = {
        "mean_syllables_per_s": round(mean_rate, 2),
        "rate_cv":              round(cv, 3),
        "score":                round(naturalness_score, 3),
    }

    # ── Weighted composite ────────────────────────────────────────────────────
    overall = (
        0.40 * timing_score
        + 0.30 * intel_score
        + 0.20 * sem_score
        + 0.10 * naturalness_score
    )

    return {
        "timing":            timing_dim,
        "intelligibility":   intel_dim,
        "semantic_fidelity": sem_dim,
        "naturalness":       natural_dim,
        "overall":           round(overall, 3),
    }


def scorecard_from_align_json(
    align_data: dict,
    source_segments: list[dict] | None = None,
) -> dict:
    """Compute a dubbing scorecard directly from the ``align.json`` dict on disk.

    The align JSON (saved by the TTS engine) does not serialise ``SegmentMetrics``
    or ``AlignedSegment`` objects, so ``dubbing_scorecard`` cannot be called
    directly.  This function reconstructs equivalent scores from the JSON fields.

    Parameters
    ----------
    align_data:
        Dict loaded from ``<title>.align.json``.
    source_segments:
        Optional list of Whisper transcription segments
        ``[{"start", "end", "text", ...}]``.  When provided, enables the
        EN→ES length-ratio semantic-fidelity score.  Without it the semantic
        score is omitted (neutral 0.5 proxy).

    Returns
    -------
    Same shape as ``dubbing_scorecard``.
    """
    segs = align_data.get("segments", [])
    n = len(segs)
    if n == 0:
        return {
            "timing":           {"score": 0.0},
            "intelligibility":  {"score": 0.0, "method": "proxy"},
            "semantic_fidelity":{"score": 0.0, "method": "length_ratio"},
            "naturalness":      {"score": 0.0, "rate_cv": 0.0, "mean_syllables_per_s": 0.0},
            "overall":          0.0,
            "note":             "empty align data",
        }

    # ── 1. Timing ─────────────────────────────────────────────────────────────
    mae        = align_data.get("mean_abs_duration_error_s", 0.0)
    pct_severe = align_data.get("pct_severe_stretch", 0.0)
    drift      = align_data.get("total_cumulative_drift_s", 0.0)

    dur_score    = max(0.0, 1.0 - mae / 2.0)
    severe_score = 1.0 - pct_severe / 100.0
    drift_score  = max(0.0, 1.0 - abs(drift) / 60.0)
    timing_score = _stats.mean([dur_score, severe_score, drift_score])
    timing_dim = {
        "mean_abs_duration_error_s": round(mae, 3),
        "duration_error_score":      round(dur_score, 3),
        "pct_severe_stretch":        pct_severe,
        "severe_stretch_score":      round(severe_score, 3),
        "total_cumulative_drift_s":  round(drift, 3),
        "drift_score":               round(drift_score, 3),
        "score":                     round(timing_score, 3),
    }

    # ── 2. Intelligibility (proxy from stretch_factor / action) ───────────────
    per_intel = []
    n_fail = n_clip = 0
    for seg in segs:
        action = (seg.get("action") or "").lower()
        sf     = seg.get("stretch_factor", 1.0)
        if action == "fail":
            per_intel.append(0.0)
            n_fail += 1
        elif action in ("request_shorter", "clip"):
            # audio overflows slot — estimate heard fraction
            raw = seg.get("raw_duration_s", 1.0)
            tgt = max(0.1, seg.get("target_sec", raw))
            per_intel.append(min(1.0, tgt / raw))
            n_clip += 1
        else:
            per_intel.append(max(0.0, 1.0 - 0.3 * (sf - 1.0)))
    intel_score = _stats.mean(per_intel)
    intel_dim = {
        "method":    "proxy",
        "n_fail":    n_fail,
        "n_clipped": n_clip,
        "score":     round(intel_score, 3),
    }

    # ── 3. Semantic fidelity ──────────────────────────────────────────────────
    if source_segments and len(source_segments) == n:
        ratios = [
            len(t.get("text", "")) / max(1, len(s.get("text", "")))
            for t, s in zip(segs, source_segments)
        ]
        scores   = [max(0.0, 1.0 - abs(r - 1.2) / 1.2) for r in ratios]
        sem_score = _stats.mean(scores)
        sem_dim = {
            "method":            "length_ratio",
            "mean_length_ratio": round(_stats.mean(ratios), 3),
            "score":             round(sem_score, 3),
        }
    else:
        # No source text available — neutral proxy
        sem_score = 0.5
        sem_dim = {"method": "neutral_proxy", "score": sem_score}

    # ── 4. Naturalness (syllables per target slot second) ─────────────────────
    rates = [
        _count_syllables(seg.get("text", "")) / max(0.1, seg.get("target_sec", 1.0))
        for seg in segs
    ]
    mean_rate = _stats.mean(rates)
    cv = (_stats.stdev(rates) / mean_rate) if len(rates) >= 2 and mean_rate > 0 else 0.0
    nat_score = max(0.0, 1.0 - max(0.0, cv - 0.3) / 1.2)
    natural_dim = {
        "mean_syllables_per_s": round(mean_rate, 2),
        "rate_cv":              round(cv, 3),
        "score":                round(nat_score, 3),
    }

    # ── Weighted composite ────────────────────────────────────────────────────
    overall = (
        0.40 * timing_score
        + 0.30 * intel_score
        + 0.20 * sem_score
        + 0.10 * nat_score
    )
    return {
        "timing":            timing_dim,
        "intelligibility":   intel_dim,
        "semantic_fidelity": sem_dim,
        "naturalness":       natural_dim,
        "overall":           round(overall, 3),
    }
