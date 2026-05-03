"""HuggingFace Spaces entry point — Foreign Whispers dubbing pipeline.

Single-container deployment: yt-dlp → faster-whisper → argostranslate →
Chatterbox TTS (in-process) → ffmpeg stitch → Gradio video output.

Deploy by pushing this repo to an HF Docker Space (GPU hardware recommended).
See spaces/Dockerfile for the container definition.
"""

import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import threading
import traceback

import gradio as gr

# Add project root to path so api.src.* and foreign_whispers.* are importable.
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Serialize GPU TTS calls — in-process model cannot handle concurrent requests.
os.environ.setdefault("FW_TTS_WORKERS", "1")


# ---------------------------------------------------------------------------
# In-process Chatterbox TTS  (replaces the HTTP ChatterboxClient)
# ---------------------------------------------------------------------------

class _InProcessChatterboxClient:
    """Chatterbox TTS loaded directly into the process.

    Matches the ``ChatterboxClient.tts_to_file`` interface so it can be
    passed as ``tts_engine`` to ``tts_engine.text_file_to_speech``.
    Thread-safe via a lock — the GPU model is not re-entrant.
    """

    _instance: "_InProcessChatterboxClient | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "_InProcessChatterboxClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[tts] Loading Chatterbox on {device}…")
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self._sr: int = self._model.sr
        print("[tts] Chatterbox ready.")

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        import torch
        import torchaudio

        speaker_wav: str = kwargs.get("speaker_wav", "") or ""
        chunks = self._split_text(text) if len(text) > 200 else [text]

        with self._lock:
            parts = []
            for chunk in chunks:
                if speaker_wav and pathlib.Path(speaker_wav).exists():
                    wav = self._model.generate(chunk, audio_prompt_path=speaker_wav)
                else:
                    wav = self._model.generate(chunk)
                parts.append(wav)

        combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        torchaudio.save(file_path, combined, self._sr)

    @staticmethod
    def _split_text(text: str, max_len: int = 200) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for s in sentences:
            if current and len(current) + len(s) + 1 > max_len:
                chunks.append(current.strip())
                current = s
            else:
                current = f"{current} {s}".strip() if current else s
        if current:
            chunks.append(current.strip())
        return chunks or [text]


# ---------------------------------------------------------------------------
# In-process faster-whisper STT  (replaces the speaches HTTP service)
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            import torch
            from faster_whisper import WhisperModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            print(f"[stt] Loading faster-whisper medium on {device}…")
            _whisper_model = WhisperModel("medium", device=device, compute_type=compute)
            print("[stt] Whisper ready.")
    return _whisper_model


def _transcribe_video(video_path: pathlib.Path) -> dict:
    model = _get_whisper()
    segs_iter, info = model.transcribe(str(video_path), language="en", beam_size=5)
    segs, texts = [], []
    for i, s in enumerate(segs_iter):
        segs.append({"id": i, "start": s.start, "end": s.end, "text": s.text.strip()})
        texts.append(s.text.strip())
    return {"language": info.language, "text": " ".join(texts), "segments": segs}


# ---------------------------------------------------------------------------
# VTT caption generation
# ---------------------------------------------------------------------------

def _segments_to_vtt(segments: list[dict]) -> str:
    segs = [s for s in segments if s.get("text", "").strip()]
    if not segs:
        return "WEBVTT\n"

    def _fmt(t: float) -> str:
        h, m, s = int(t // 3600), int((t % 3600) // 60), t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    lines = ["WEBVTT", ""]
    prev = None
    for i, seg in enumerate(segs, 1):
        text = seg["text"].strip()
        lines += [str(i), f"{_fmt(seg['start'])} --> {_fmt(seg['end'])}",
                  f"{text}\n{prev}" if prev else text, ""]
        prev = text
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full dubbing pipeline
# ---------------------------------------------------------------------------

def dub_video(youtube_url: str, progress=gr.Progress()) -> tuple[str | None, str]:
    """End-to-end pipeline. Returns (dubbed_video_path, log_text)."""
    if not youtube_url.strip():
        return None, "Please enter a YouTube URL."

    logs: list[str] = []

    def log(msg: str) -> str:
        logs.append(msg)
        return "\n".join(logs)

    try:
        with tempfile.TemporaryDirectory() as _tmp:
            work = pathlib.Path(_tmp)

            # ── 1. Download ──────────────────────────────────────────────
            progress(0.05, desc="Downloading video…")
            from api.src.services.download_engine import (
                download_caption,
                download_video,
                get_video_info,
            )

            video_id, raw_title = get_video_info(youtube_url)
            title = re.sub(r'[:|/\\]', "", raw_title).strip()
            log(f"✓ Video: {title}")

            vid_path = pathlib.Path(download_video(youtube_url, str(work), title))
            log(f"✓ Downloaded ({vid_path.stat().st_size // 1_000_000} MB)")
            progress(0.15)

            # ── 2. Transcribe ────────────────────────────────────────────
            transcript: dict | None = None

            # Try YouTube captions first (accurate timestamps, no GPU needed)
            cap_path = work / f"{title}.txt"
            try:
                download_caption(youtube_url, str(work), title)
                if cap_path.exists():
                    from api.src.routers.transcribe import _youtube_captions_to_segments
                    transcript = _youtube_captions_to_segments(cap_path)
                    log(f"✓ YouTube captions ({len(transcript['segments'])} segs)")
            except Exception:
                pass

            if not transcript:
                progress(0.25, desc="Transcribing with Whisper…")
                transcript = _transcribe_video(vid_path)
                log(f"✓ Whisper STT ({len(transcript['segments'])} segs)")

            progress(0.35)

            # ── 3. Translate ─────────────────────────────────────────────
            progress(0.40, desc="Translating to Spanish…")
            from api.src.services.translation_engine import (
                download_and_install_package,
                translate_sentence,
            )
            import copy

            download_and_install_package("en", "es")
            es_transcript = copy.deepcopy(transcript)
            for seg in es_transcript.get("segments", []):
                seg["text"] = translate_sentence(seg["text"], "en", "es")
            es_transcript["text"] = translate_sentence(es_transcript.get("text", ""), "en", "es")
            es_transcript["language"] = "es"
            log(f"✓ Translated {len(es_transcript['segments'])} segments EN→ES")
            progress(0.55)

            # Lay out the directory structure tts_engine._load_en_transcript expects:
            #   data/translations/argos/{title}.json   ← ES translation
            #   data/transcriptions/whisper/{title}.json ← EN source (for alignment)
            #   data/youtube_captions/{title}.txt       ← for speech offset (optional)
            data_root = work / "data"
            trans_dir = data_root / "translations" / "argos"
            trans_dir.mkdir(parents=True)
            transcr_dir = data_root / "transcriptions" / "whisper"
            transcr_dir.mkdir(parents=True)
            yt_cap_dir = data_root / "youtube_captions"
            yt_cap_dir.mkdir(parents=True)

            trans_path = trans_dir / f"{title}.json"
            transcr_path = transcr_dir / f"{title}.json"
            trans_path.write_text(json.dumps(es_transcript))
            transcr_path.write_text(json.dumps(transcript))
            if cap_path.exists():
                shutil.copy(cap_path, yt_cap_dir / f"{title}.txt")

            # ── 4. TTS + temporal alignment ──────────────────────────────
            progress(0.60, desc="Synthesizing dubbed audio…")
            from api.src.services.tts_engine import text_file_to_speech

            tts_dir = work / "tts"
            tts_dir.mkdir()
            text_file_to_speech(
                str(trans_path),
                str(tts_dir),
                tts_engine=_InProcessChatterboxClient.get(),
                alignment=True,
                lang="es",
            )

            audio_path = tts_dir / f"{title}.wav"
            if not audio_path.exists():
                raise FileNotFoundError("TTS output not found — check logs above")
            log(f"✓ Dubbed audio ({audio_path.stat().st_size // 1_000} KB)")
            progress(0.88)

            # ── 5. Stitch ────────────────────────────────────────────────
            progress(0.92, desc="Stitching video…")
            dubbed_tmp = work / f"{title}_dubbed.mp4"
            from api.src.services.stitch_engine import stitch_audio

            stitch_audio(str(vid_path), str(audio_path), str(dubbed_tmp))
            log("✓ Dubbed video ready")

            # Generate VTT captions for the dubbed video
            vtt_content = _segments_to_vtt(es_transcript.get("segments", []))

            # Copy outputs to /tmp/fw_out (survives TemporaryDirectory cleanup)
            out_dir = pathlib.Path("/tmp/fw_out") / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            final_vid = out_dir / "dubbed.mp4"
            final_vtt = out_dir / "dubbed.vtt"
            shutil.copy(dubbed_tmp, final_vid)
            final_vtt.write_text(vtt_content)

            progress(1.0, desc="Done!")
            return str(final_vid), "\n".join(logs)

    except Exception as exc:
        return None, f"Pipeline failed: {exc}\n\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Foreign Whispers — AI Video Dubbing") as demo:
    gr.Markdown(
        """
# 🎬 Foreign Whispers
**Automatic video dubbing** — paste a YouTube URL to receive a Spanish-dubbed MP4 with aligned subtitles.

Pipeline: `yt-dlp` → `Whisper STT` → `argostranslate` → `Chatterbox TTS` → `ffmpeg`

> First run loads Whisper and Chatterbox models (~2 min). Subsequent runs are faster.
        """
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=4,
        )
        run_btn = gr.Button("Dub", variant="primary", scale=1)

    with gr.Row():
        video_out = gr.Video(label="Dubbed Video", interactive=False, scale=3)
        log_out = gr.Textbox(label="Pipeline log", lines=14, interactive=False, scale=2)

    run_btn.click(
        fn=dub_video,
        inputs=url_input,
        outputs=[video_out, log_out],
    )

    gr.Examples(
        examples=["https://www.youtube.com/watch?v=GYQ5yGV_-Oc"],
        inputs=url_input,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
