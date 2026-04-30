"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import collections
import logging
import sys

import torchaudio as _torchaudio

# ── torchaudio 2.10+ compatibility shim ──────────────────────────────────────
# pyannote.audio 3.x was written against torchaudio 2.0-2.2.  In torchaudio
# 2.10+ the entire I/O layer was replaced by torchcodec (not installed here).
# We patch the missing top-level symbols with soundfile-backed equivalents so
# pyannote can import and run without any native torchaudio I/O support.

_AudioMetaData = collections.namedtuple(
    "AudioMetaData",
    ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
)

if not hasattr(_torchaudio, "AudioMetaData"):
    _torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(_torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"]
    _torchaudio.list_audio_backends = _list_audio_backends

if not hasattr(_torchaudio, "info"):
    def _info(path, backend=None):
        import soundfile as _sf
        _i = _sf.info(str(path))
        return _AudioMetaData(
            sample_rate=_i.samplerate,
            num_frames=_i.frames,
            num_channels=_i.channels,
            bits_per_sample=0,
            encoding="PCM_S",
        )
    _torchaudio.info = _info

# torchaudio 2.10 load() requires torchcodec; patch with soundfile fallback.
_orig_load = _torchaudio.load
def _load_shim(uri, frame_offset=0, num_frames=-1, normalize=True,
               channels_first=True, format=None, buffer_size=4096, backend=None):
    try:
        return _orig_load(uri, frame_offset=frame_offset, num_frames=num_frames,
                          normalize=normalize, channels_first=channels_first,
                          format=format, buffer_size=buffer_size, backend=backend)
    except (ImportError, RuntimeError):
        pass
    import soundfile as _sf
    import torch as _torch
    _data, _sr = _sf.read(
        str(uri), start=frame_offset,
        frames=num_frames if num_frames > 0 else -1,
        dtype="float32", always_2d=True,
    )
    _wf = _torch.from_numpy(_data.T if channels_first else _data)
    return _wf, _sr
_torchaudio.load = _load_shim

# PyTorch 2.6+ changed the default of torch.load to weights_only=True.
# Pyannote checkpoints embed several custom types (TorchVersion, Specifications,
# etc.) that are not in the safe-globals allowlist.  Restore the pre-2.6 default
# so that trusted pyannote models load without enumerating every embedded type.
try:
    import torch as _torch
    _orig_torch_load = _torch.load
    def _permissive_torch_load(*_args, **_kwargs):
        _kwargs["weights_only"] = False  # force — pyannote checkpoints need legacy unpickling
        return _orig_torch_load(*_args, **_kwargs)
    _torch.load = _permissive_torch_load
except Exception:
    pass

# Evict any partially-cached pyannote entries so the patched torchaudio is
# visible the next time pyannote.audio is imported from scratch.
for _key in list(sys.modules.keys()):
    if _key.startswith("pyannote"):
        del sys.modules[_key]
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        logger.warning("pyannote.audio not available (%s) — returning empty diarization.", exc)
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label.  If diarization is
    empty or no interval overlaps a segment, that segment defaults to
    ``"SPEAKER_00"``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        The original list and its dicts are not mutated.
    """
    result = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for interval in diarization:
            overlap = max(0.0, min(seg_end, interval["end_s"]) - max(seg_start, interval["start_s"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = interval["speaker"]
        result.append({**seg, "speaker": best_speaker})
    return result
