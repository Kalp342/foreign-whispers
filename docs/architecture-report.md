# Foreign Whispers — Technical Architecture Report

**Contributors:** Kalp (kpp10037@nyu.edu), Prashanth (ps5761@nyu.edu)

---

## 1. Overview

Foreign Whispers is a video dubbing pipeline that accepts a YouTube URL and
produces a new MP4 in which the original English speech has been replaced with
synthesized Spanish speech, with WebVTT subtitles served alongside. The pipeline
runs five sequential stages: video download (yt-dlp), speech-to-text
(Whisper / YouTube captions), translation (argostranslate), text-to-speech
(Chatterbox), and audio-video stitching (ffmpeg). Temporal alignment — making
the synthesized speech fit inside the original segment windows — is handled by a
separate Python library (`foreign_whispers`) that is callable independently of
the running service stack.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  Host                                                                │
│                                                                      │
│  pipeline_data/    ← bind-mounted into API container                │
│  foreign_whispers/ ← bind-mounted into API container                │
│  api/              ← bind-mounted into API container                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Docker Compose (profile: nvidia)                            │   │
│  │                                                              │   │
│  │  ┌──────────────────┐   HTTP    ┌──────────────────────┐    │   │
│  │  │ foreign-whispers │ :8080/api │ foreign-whispers-stt │    │   │
│  │  │ -api  (CPU)      │──────────▶│ (Whisper GPU) :8000  │    │   │
│  │  │ FastAPI          │           │ speaches image        │    │   │
│  │  │ uvicorn          │◀──────────│ faster-whisper-medium │    │   │
│  │  └────────┬─────────┘   JSON    └──────────────────────┘    │   │
│  │           │                                                  │   │
│  │           │ HTTP                ┌──────────────────────┐    │   │
│  │           └────────────────────▶│ foreign-whispers-tts │    │   │
│  │                                 │ (Chatterbox GPU):8020│    │   │
│  │                                 │ travisvn/chatterbox  │    │   │
│  │                                 │ multilingual model   │    │   │
│  │                                 └──────────────────────┘    │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │ foreign-whispers-frontend  (Next.js) :8501           │   │   │
│  │  │ Proxies /api/* → http://localhost:8080               │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

Data flow:
  YouTube URL → /api/download → /api/transcribe → /api/diarize (optional)
              → /api/translate → /api/tts → /api/stitch → GET /api/video/*
```

---

## 3. Pipeline Stages

### 3.1 Download (`POST /api/download`)

**Input:** YouTube URL  
**Output:** `pipeline_data/api/videos/{title}.mp4`, `pipeline_data/api/youtube_captions/{title}.txt`

`api/src/services/download_engine.py` wraps yt-dlp to fetch the video as MP4
and separately download the auto-generated English captions as line-delimited
JSON (`{"text", "start", "duration"}` per line). A `cookies.txt` file is
bind-mounted into the API container to handle authentication where YouTube
requires it. The video registry (`video_registry.yml`) maps YouTube video IDs to
canonical title strings; those title strings become the filename stem for every
downstream artifact. This avoids filename conflicts and keeps artifact paths
predictable without a database.

---

### 3.2 Transcription (`POST /api/transcribe/{video_id}`)

**Input:** `videos/{title}.mp4` (or `youtube_captions/{title}.txt`)  
**Output:** `transcriptions/whisper/{title}.json`

The default path (`use_youtube_captions=True`) converts the yt-dlp caption JSON
directly into a Whisper-compatible segment dict and caches it as
`transcriptions/whisper/{title}.json`. The YouTube captions are preferred because
their timestamps are accurate relative to the video start; Whisper's internal
timestamps reset to zero at the beginning of each audio file and need an offset
correction.

When YouTube captions are absent or the caller forces `use_youtube_captions=False`,
the transcribe router loads Whisper via a lazy singleton (`get_whisper_model` in
`api/src/main.py`). The remote Whisper path (`inference/whisper_remote.py`)
POSTs the audio file to `http://localhost:8000/v1/audio/transcriptions`
(the speaches service running `Systran/faster-whisper-medium`), which returns
`verbose_json` with per-segment start/end timestamps.

The output JSON structure — `{"language", "text", "segments": [{"id", "start", "end", "text"}]}` —
is the shared wire format consumed by all downstream stages.

---

### 3.3 Diarization (`POST /api/diarize/{video_id}`) — optional

**Input:** `videos/{title}.mp4`  
**Output:** speaker labels merged into `transcriptions/whisper/{title}.json`

`foreign_whispers/diarization.py` wraps `pyannote/speaker-diarization-3.1` via
`pyannote.audio`. Because pyannote was written against torchaudio 2.0–2.2, the
module patches `torchaudio.load`, `torchaudio.info`, and `torchaudio.AudioMetaData`
with soundfile-backed equivalents at import time, to survive torchaudio 2.10+
where the I/O layer was removed. It also forces `torch.load(weights_only=False)`
for pyannote checkpoints, which contain custom types that PyTorch 2.6+'s
safe-unpickling rejects.

`assign_speakers` (also in `diarization.py`) merges the diarization intervals
into the transcription segments: for each Whisper segment it finds the
diarization interval with the greatest temporal overlap and copies its
`SPEAKER_XX` label into the segment dict. The labeled segments are written back
to the transcription JSON so that the TTS stage can map speakers to distinct
reference voices.

Diarization requires a Hugging Face token (`FW_HF_TOKEN`) to download the
pyannote model. When the token is absent the function returns an empty list and
the pipeline continues without speaker labels. The `alignment` dependency group
in `pyproject.toml` installs `pyannote.audio`, `silero-vad`, and related
packages; the base install does not.

---

### 3.4 Translation (`POST /api/translate/{video_id}`)

**Input:** `transcriptions/whisper/{title}.json`  
**Output:** `translations/argos/{title}.json`

`api/src/services/translation_engine.py` uses `argostranslate` (offline
OpenNMT) to translate each segment's text in place. The segment schema is
preserved — start/end timestamps carry through unchanged — and the `language`
field is updated to the target code. Argostranslate packages must be
pre-installed or downloaded at runtime; the `download_and_install_package`
helper handles this.

When a segment's translation is too long for its time window, the
`REQUEST_SHORTER` alignment action invokes `foreign_whispers/reranking.py`.
That module produces shorter candidates via three tiers: (1) a rule-based table
of 40+ Spanish phrase contractions and filler-word strippers, (2) a MarianMT
(`Helsinki-NLP/opus-mt-en-es`) re-translation, and (3) sentence/clause/word-
boundary truncation as a guaranteed fallback. Candidates are scored by
`(predicted_duration − target)² + 0.1 × bigram_distance` and the lowest-score
option is used.

---

### 3.5 TTS Synthesis (`POST /api/tts/{video_id}`)

**Input:** `translations/argos/{title}.json`  
**Output:** `tts_audio/chatterbox/{config}/{title}.wav`, `{title}.align.json`

This is the most complex stage. `api/src/services/tts_engine.py:text_file_to_speech`
runs in two phases.

**Phase 1 — GPU synthesis (concurrent):** A `ThreadPoolExecutor` with
`FW_TTS_WORKERS` threads (default 3 in development, 1 in Docker via the
override file) submits each segment to `ChatterboxClient.tts_to_file`, which
POSTs to `http://localhost:8020/v1/audio/speech`. When a speaker reference WAV
is available (via the voice map built from diarization labels),
`/v1/audio/speech/upload` is called instead with the reference WAV as a
multipart attachment, enabling zero-shot voice cloning. Texts longer than 200
characters are split at sentence boundaries before posting to stay within
Chatterbox's input limits.

The `config` URL parameter is an opaque 7-character hex string derived from the
DJB2 hash of a JSON config object (e.g. `{"d":"baseline"}`). This allows
multiple TTS runs with different settings to coexist in `tts_audio/chatterbox/`
without overwriting each other.

**Phase 2 — CPU post-processing (sequential assembly):** After all GPU calls
complete, the raw WAV bytes for each segment are time-stretched to fit the
original segment window using `pyrubberband.time_stretch`. The stretch factor
comes from the DP alignment precomputed at the start of the function
(`global_align_dp` from `foreign_whispers/alignment.py`). A `pydub.AudioSegment`
timeline is built by cursor: silence fills gaps between segments, each segment
audio is trimmed or padded to match its target window exactly, and the result is
exported as a single WAV at `tts_audio/chatterbox/{config}/{title}.wav`.

A sidecar `{title}.align.json` is written alongside the WAV with per-segment
metrics (raw duration, speed factor, action taken) and a summary from
`clip_evaluation_report`.

**Voice resolution** (`foreign_whispers/voice_resolution.py`): The TTS router
builds a per-speaker voice map using `resolve_speaker_wav`. Resolution order:
`speakers/{lang}/{speaker_id}.wav` → `speakers/{lang}/default.wav` →
`speakers/default.wav`. Reference WAVs are stored in `pipeline_data/speakers/`
and bind-mounted into the Chatterbox container at `/app/voices/`.

**Recent fix (commit 52f1a3e):** TTS workers were set back to sequential
(`FW_TTS_WORKERS=1` in the override) and the request timeout extended from 30s
to 180s after silent segments appeared when concurrent GPU calls were dropped.

---

### 3.6 Stitch (`POST /api/stitch/{video_id}`)

**Input:** `videos/{title}.mp4`, `tts_audio/chatterbox/{config}/{title}.wav`  
**Output:** `dubbed_videos/{config}/{title}.mp4`, `dubbed_captions/{title}.vtt`

`api/src/services/stitch_engine.py:stitch_audio` invokes ffmpeg with
`-c:v copy` (no video re-encoding) to replace the audio track. The
`-map 0:v:0 -map 1:a:0 -shortest` flags select the video stream from the
original file and the audio stream from the TTS WAV, stopping at the shorter
stream's end. This is fast (no GPU needed) and lossless for the video stream.

WebVTT captions are generated in `_write_dubbed_captions` (stitch router):
translated segments have the YouTube caption timing offset applied, then are
formatted in rolling two-line style (current line on top, previous line on
bottom, matching Google's caption UX). The VTT file is served via
`GET /api/captions/{id}` using an HTML `<track>` element — no subtitle burn-in.

---

## 4. Service Architecture

```
Service                 Image                                   Profile   Port   GPU
─────────────────────── ─────────────────────────────────────── ──────── ───── ─────
foreign-whispers-stt    ghcr.io/speaches-ai/speaches:cuda-12.6 nvidia   8000   yes
foreign-whispers-tts    travisvn/chatterbox-tts-api:latest      nvidia   8020   yes
foreign-whispers-api    ./Dockerfile (multi-stage, cpu target)  nvidia   8080   no
foreign-whispers-frontend ./frontend/Dockerfile                 nvidia   8501   no
```

All four services use `network_mode: host` in `docker-compose.yml`, so they
communicate over localhost. The override file (`docker-compose.override.yml`)
switches to `network_mode: bridge` with explicit port mappings and replaces
`CHATTERBOX_API_URL` with `http://host.docker.internal:8020`, which is the
typical developer setup on macOS.

The API container runs as the host UID/GID (set via `UID`/`GID` env vars) so
all files written to `pipeline_data/` are owned by the developer, not root.

The STT and TTS containers expose OpenAI-compatible endpoints:
- `POST /v1/audio/transcriptions` (Whisper)
- `POST /v1/audio/speech` and `POST /v1/audio/speech/upload` (Chatterbox)

The API container contains no GPU code; it delegates all inference via HTTP,
which means GPU containers can be swapped independently. The `inference/` layer
in the API source (`base.py`, `whisper_remote.py`, `tts_remote.py`) defines
abstract base classes and their HTTP implementations, though in practice the TTS
engine used by `tts_engine.py` is the `ChatterboxClient` class directly, not
the `RemoteTTSBackend`.

Both `foreign_whispers/` and `api/` are bind-mounted from the host into the API
container, so Python changes are visible immediately after a container restart
(or without restart if `--reload` is added to the uvicorn command).

---

## 5. Key Technical Decisions

### 5.1 Temporal alignment via DP optimization

The central problem in dubbing is that translated speech takes a different amount
of time than the source speech. Spanish is typically 15–25% longer than English
for the same content.

`foreign_whispers/alignment.py` addresses this with a two-algorithm approach:

**Greedy (`global_align`):** Single left-to-right pass. For each segment it
calls `decide_action(m, available_gap_s)` and schedules the segment on a shared
timeline, accumulating cumulative drift from gap-shifts.

**DP (`global_align_dp`):** Minimises a joint objective of stretch penalty plus
drift cost across all segments. The state is `(segment_index, cumulative_drift)`
discretised at 50 ms steps. For each state it evaluates two choices — gap-shift
or no gap-shift — and propagates the minimum-cost path. The backtrack phase
reconstructs the full decision sequence. DP is selective about where to borrow
silence: it takes a gap-shift only when the stretch penalty saved exceeds the
drift penalty incurred. The greedy always takes every eligible gap.

The `decide_action` function maps predicted stretch to five actions:
- `ACCEPT` (≤ 1.1×): no modification
- `MILD_STRETCH` (1.1–1.4×): time-stretch with pyrubberband, capped at 1.4×
- `GAP_SHIFT` (1.4–1.8×, gap available): expand into adjacent silence
- `REQUEST_SHORTER` (1.8–2.5×): invoke translation reranking
- `FAIL` (> 2.5×): replace with silence

### 5.2 Duration prediction via OLS model

Rather than a raw syllable-rate heuristic (4.5 syl/s), the duration model
(`_estimate_duration`) uses an OLS regression fitted on 156 Chatterbox/Spanish
segments. Coefficients: 0.145 s/syllable, 0.110 s/word (inter-word pauses),
0.154 s/comma, 0.091 s/sentence-terminal, 0.532 s fixed overhead. Mean absolute
error on the training set: 0.30 s vs 0.45 s for the heuristic. Syllables are
counted by stripping accents (NFKD decomposition) then counting contiguous vowel
runs, which works well for Romance languages.

### 5.3 Sequential TTS synthesis

An earlier concurrent implementation (multiple threads hammering the Chatterbox
GPU) produced silent segments. Commit 52f1a3e switched to `FW_TTS_WORKERS=1`
in the Docker override and extended the per-request timeout to 180 s. The
code still supports concurrent workers (controlled by `FW_TTS_WORKERS`); the
default in development is 3, but the Docker override caps it at 1 for
reliability.

### 5.4 Speech offset disabled

Commit 393abb3 zeroed out `_compute_speech_offset` in `tts_engine.py`. The
prior implementation computed the delta between the first YouTube caption
timestamp and the first Whisper segment timestamp. When YouTube captions
included non-speech markers (e.g. `[Music]`) at t=0 while Whisper correctly
placed the first segment at 5–10 s, the offset was large and negative, shifting
all dubbed audio well ahead of the corresponding video speech. The fix: Whisper
timestamps are already relative to the video start, so no offset is needed.

### 5.5 Caching via config ID

Pipeline artifacts are namespaced by a 7-character DJB2 hash of a JSON config
object (e.g. `c-fb1074a` for baseline, `c-86ab861` for aligned). This lets
`tts_audio/chatterbox/` and `dubbed_videos/` hold multiple outputs from different
parameter combinations without overwriting each other. The same hash function is
implemented in both Python (`foreign_whispers/client.py:_djb2`) and TypeScript
(`frontend/src/lib/config-id.ts`) so the frontend and SDK agree on directory names.

### 5.6 Video registry without a database

`video_registry.yml` is a flat YAML list loaded at startup by
`api/src/core/video_registry.py`. The API exposes it at `GET /api/videos`. Title
strings from the registry become the filename stem for all artifacts — there is
no database, no ID generation, and no migration path. Adding a video requires
editing the YAML and restarting the API container.

---

## 6. Data Model

### 6.1 `video_registry.yml`

Each entry has: `id` (YouTube video ID), `title` (human-readable, used as file
stem), `url`, `source_language`, `target_language`. Currently four 60 Minutes
interviews are registered, all `en → es`.

### 6.2 `pipeline_data/api/` directory layout

```
pipeline_data/api/
├── videos/                          # Downloaded source MP4s
│   └── {title}.mp4
├── youtube_captions/                # Line-delimited JSON from yt-dlp
│   └── {title}.txt
├── transcriptions/
│   └── whisper/                     # Whisper/YouTube caption output
│       └── {title}.json             # {language, text, segments:[{id,start,end,text}]}
├── diarizations/                    # pyannote speaker intervals (if run)
│   └── {title}.json
├── translations/
│   └── argos/                       # argostranslate segment output
│       └── {title}.json             # same schema; text fields replaced with ES
├── tts_audio/
│   └── chatterbox/
│       └── {config}/                # e.g. c-fb1074a/
│           ├── {title}.wav          # full dubbed audio track
│           └── {title}.align.json   # per-segment metrics + summary
├── dubbed_captions/
│   └── {title}.vtt                  # WebVTT target-language subtitles
├── dubbed_videos/
│   └── {config}/
│       └── {title}.mp4              # final video with dubbed audio
└── speakers/                        # Reference voice WAVs for cloning
    ├── es/
    │   ├── default.wav
    │   └── SPEAKER_00.wav (optional)
    └── default.wav
```

The segment JSON schema `{id, start, end, text, speaker?}` is shared across
transcription, diarization merge, and translation. The `speaker` field is added
by `assign_speakers` and carried through to the TTS stage for voice map
construction.

### 6.3 Alignment sidecar (`.align.json`)

Written by `_write_align_report` in `tts_engine.py`:

```json
{
  "mean_abs_duration_error_s": 0.302,
  "pct_severe_stretch": 12.5,
  "n_gap_shifts": 3,
  "n_translation_retries": 1,
  "total_cumulative_drift_s": 0.85,
  "alignment_enabled": true,
  "segments": [
    { "index": 0, "text": "...", "target_sec": 3.2,
      "stretch_factor": 1.1, "raw_duration_s": 3.8,
      "speed_factor": 1.19, "action": "mild_stretch" }
  ]
}
```

This sidecar is consumed by `evaluation.scorecard_from_align_json` to produce
the four-dimensional dubbing scorecard (timing, intelligibility, semantic
fidelity, naturalness) without needing the original Python objects.

---

## 7. Limitations and Known Issues

**Translation reranking is partially a stub.** `foreign_whispers/reranking.py:get_shorter_translations`
is documented as a student assignment. The three tiers (phrase contractions,
MarianMT re-translation, truncation) are implemented, but MarianMT is loaded
lazily and adds latency. The function returns an empty list (falling back to the
baseline translation) when MarianMT is unavailable. This means `REQUEST_SHORTER`
segments frequently play the baseline translation clipped at the slot boundary
rather than a genuinely shorter alternative.

**Greedy alignment is the fallback in the alignment library, but the DP
optimizer (`global_align_dp`) is the path actually used in production.** The
difference matters: the greedy accepts every eligible gap-shift regardless of
downstream cost; the DP is selective. Both are O(n) and O(n × N_D) respectively
and complete in milliseconds, but the DP produces measurably lower cumulative
drift on clips where multiple segments compete for the same silence regions.

**No diarization without HuggingFace token.** When `FW_HF_TOKEN` is not set,
`diarize_audio` returns an empty list and all segments get `SPEAKER_00`. This
means voice cloning is effectively single-speaker unless the user configures the
token. The `alignment` dependency group must also be installed
(`uv run --group alignment ...`) for pyannote to be importable in the API
container.

**torchaudio 2.10+ compatibility shim is fragile.** The entire `diarization.py`
module opens with a runtime patch of six torchaudio symbols. This will break if
pyannote.audio 4.x changes its I/O surface or if torch changes its `load` API
again. A proper fix would pin the pyannote / torchaudio versions or replace
pyannote with a checkpoint that does not depend on the legacy I/O layer.

**No retry or partial recovery in the pipeline.** Each stage is idempotent via
file existence checks, but if a stage fails mid-run the artifact is not written
and the stage must be re-run from scratch. Long TTS runs (10–20 minutes for a
15-minute video) that fail near the end waste significant GPU time.

**Whisper is not actually used for STT in the current default configuration.**
YouTube captions are preferred (`use_youtube_captions=True` by default). The
local Whisper model and the speaches container both exist as fallbacks but are
rarely exercised with the registered 60 Minutes videos, which all have captions.

**TTS temporal alignment is marked as an open issue** (`fw-tov`) in the beads
tracker. The current implementation (post-hoc stretch via pyrubberband) is the
baseline; the design doc (`docs/tts-temporal-alignment-research.md`) surveys
isochronous machine translation and duration-controlled TTS as more principled
approaches. The field literature (Lakew et al., Wu et al., Microsoft TDA-TTS)
consistently shows that ±15% post-hoc stretch is the practical quality ceiling
before audible artifacts appear.

---

## 8. Test Coverage

The `tests/` directory contains 28 test files covering: alignment unit tests
(`test_alignment.py`), service-layer tests (`test_services.py`,
`test_tts_service_alignment.py`, `test_translation_service_rerank.py`),
router-level tests with httpx (`test_tts_router.py`, `test_stitch_router.py`,
`test_transcribe_router.py`), and infrastructure tests
(`test_docker_compose.py`, `test_path_portability.py`). Tests that require
silero-vad or pyannote are marked with `@pytest.mark.requires_silero` and
`@pytest.mark.requires_pyannote` and skipped unless the deps are present.
The `test_alignment.py` suite guarantees the syllable counter, `decide_action`
thresholds, and `compute_segment_metrics` pairing logic at the unit level.
