# Foreign Whispers

**A video dubbing pipeline that translates and dubs YouTube videos into a target language.**

**Contributors:** Kalp Patel (kpp10037@nyu.edu), Prashanth Sreenivasan (ps5761@nyu.edu)

---

## What it does

Foreign Whispers accepts a YouTube URL and produces a new MP4 in which the original English speech has been replaced with synthesized Spanish audio. Each segment of speech is transcribed, translated, and re-synthesized by a neural TTS model, then time-stretched to fit inside the original segment's timestamp window. The final video is served with WebVTT subtitles via the browser `<track>` element — no subtitle burn-in.

The pipeline runs five stages in order: **download** (yt-dlp) → **transcribe** (Whisper or YouTube captions) → **translate** (argostranslate, offline) → **synthesize speech** (Chatterbox, GPU) → **stitch** (ffmpeg, no video re-encoding). An optional **diarize** stage (pyannote.audio) identifies speakers and routes each segment to a per-speaker reference voice for cloning.

---

## Architecture at a glance

| Service | Image | Port | GPU |
|---------|-------|------|-----|
| `foreign-whispers-api` | `./Dockerfile` (python:3.11-slim) | 8080 | no |
| `foreign-whispers-frontend` | `./frontend/Dockerfile` (node:22-alpine) | 8501 | no |
| `foreign-whispers-stt` | `ghcr.io/speaches-ai/speaches:latest-cuda-12.6.3` | 8000 | yes |
| `foreign-whispers-tts` | `travisvn/chatterbox-tts-api:latest` | 8020 | yes |

The API container is CPU-only and delegates all inference to the STT and TTS containers over HTTP. See [docs/architecture-report.md](docs/architecture-report.md) for a full technical breakdown.

---

## Prerequisites

- **OS:** Linux or macOS. Windows is untested.
- **Docker + Docker Compose:** v25+ recommended. Install from [docs.docker.com](https://docs.docker.com/get-docker/).
- **NVIDIA GPU + drivers:** Required for the STT and TTS containers. CPU-only mode (`--profile cpu`) skips those containers but you will need to provide Whisper and TTS endpoints externally.
  - Verify your setup: `docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi`
  - CUDA 12.x driver required (the speaches image uses CUDA 12.6).
- **Disk space:** ~15 GB. Docker images (≈5 GB), Whisper model cache (≈150 MB for `base`), Chatterbox model (≈2 GB), and pipeline artifacts per video (≈500 MB each).
- **HuggingFace account:** Required only for speaker diarization (`POST /api/diarize`). Accept the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model terms, then generate a read token at <https://huggingface.co/settings/tokens>.

---

## Quick start

```bash
# 1. Clone the repo
git clone <repo-url> foreign-whispers
cd foreign-whispers

# 2. Set your host UID/GID and credentials
cp .env.example .env
#   Open .env and fill in:
#     UID=<output of: id -u>
#     GID=<output of: id -g>
#     HF_TOKEN=hf_...          # optional, only needed for diarization
```

```bash
# 3. Start all services (NVIDIA GPU profile)
docker compose --profile nvidia up -d

# Shortcut via Makefile:
make up
```

```bash
# 4. Open the UI
open http://localhost:8501
```

The first startup takes several minutes while Docker pulls the speaches and Chatterbox images and they download their model weights. Watch progress with:

```bash
docker compose --profile nvidia logs -f
# or: make logs
```

To stop everything:

```bash
docker compose --profile nvidia down
# or: make down
```

After editing Python source files, restart the API container:

```bash
docker compose --profile nvidia restart api
```

After changing `pyproject.toml` or adding dependencies, rebuild first:

```bash
docker compose --profile nvidia build api
docker compose --profile nvidia up -d api
# or: make rebuild
```

---

## Environment variables

Copy `.env.example` to `.env` before starting. Docker Compose reads `.env` automatically.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `UID` | yes | `1000` | Host user ID. Files written to `pipeline_data/` will be owned by this UID. Run `id -u` to find yours. |
| `GID` | yes | `1000` | Host group ID. Run `id -g`. |
| `HF_TOKEN` | no | _(empty)_ | HuggingFace read token. Required for `POST /api/diarize` (pyannote model). Without it, diarization silently returns no labels and all segments use a single voice. |
| `LOGFIRE_TOKEN` | no | _(empty)_ | Pydantic Logfire write token for request tracing. Tracing is disabled when empty. |
| `CHATTERBOX_API_URL` | no | `http://localhost:8020` | URL of the Chatterbox TTS service. The override file sets this to `http://host.docker.internal:8020` for macOS bridge networking. |
| `FW_WHISPER_MODEL` | no | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). Larger models are slower but more accurate. |
| `FW_TTS_WORKERS` | no | `3` (dev) / `1` (Docker) | Concurrent TTS synthesis threads. The `docker-compose.override.yml` caps this at 1 to prevent silent segments from GPU contention. |
| `FW_ALIGNMENT` | no | `on` | Set to `off` to disable temporal alignment and use unclamped time-stretching (the legacy baseline). |
| `CHATTERBOX_SPEAKER_WAV` | no | _(empty)_ | Path to a default speaker reference WAV, relative to `pipeline_data/speakers/`. Overrides the auto-resolved voice when set. |

> **macOS note:** `docker-compose.override.yml` is already present in the repo and is picked up automatically by `docker compose`. It switches networking from `host` mode to `bridge` and sets `CHATTERBOX_API_URL=http://host.docker.internal:8020`. No action needed.

---

## Running the pipeline

### UI path

1. Open <http://localhost:8501>.
2. Select a video from the sidebar, or paste a YouTube URL into the download panel.
3. Click through the pipeline stages in order: **Download → Transcribe → Translate → Synthesize → Stitch**.
4. The dubbed video plays in the player on the right. Toggle between original and dubbed audio. Enable subtitles via the `CC` button.

Diarization is an optional step between Transcribe and Translate. Enable it in the sidebar if you have `HF_TOKEN` set and want per-speaker voice assignment.

### API path

The video ID `GYQ5yGV_-Oc` is in the registry. Use it to test the full pipeline from curl:

```bash
VIDEO=GYQ5yGV_-Oc
API=http://localhost:8080

# 1. Download video + captions
curl -s -X POST $API/api/download \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v='$VIDEO'"}' | jq .

# 2. Transcribe (uses YouTube captions by default)
curl -s -X POST $API/api/transcribe/$VIDEO | jq .segments[0:3]

# 3. (Optional) Diarize — requires HF_TOKEN
curl -s -X POST $API/api/diarize/$VIDEO | jq .speakers

# 4. Translate (English → Spanish)
curl -s -X POST "$API/api/translate/$VIDEO?target_language=es" | jq .segments[0:3]

# 5. Synthesize speech (aligned mode)
curl -s -X POST "$API/api/tts/$VIDEO?config=c-86ab861&alignment=true" | jq .

# 6. Stitch audio into video
curl -s -X POST "$API/api/stitch/$VIDEO?config=c-86ab861" | jq .

# 7. Stream the dubbed video
curl -o dubbed.mp4 "$API/api/video/$VIDEO?config=c-86ab861"

# 8. Fetch WebVTT captions
curl "$API/api/captions/$VIDEO"
```

The `config` parameter is an opaque cache key. Two pre-computed values:
- `c-fb1074a` — baseline (no alignment)
- `c-86ab861` — aligned (DP optimizer + pyrubberband)

All stages are idempotent: re-running returns the cached result if the output file already exists.

**Full API reference:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/download` | Download video + captions |
| `POST` | `/api/transcribe/{id}` | Whisper STT (or YouTube captions) |
| `POST` | `/api/diarize/{id}` | Speaker diarization (pyannote) |
| `POST` | `/api/translate/{id}` | argostranslate EN→ES |
| `POST` | `/api/tts/{id}` | TTS synthesis + time alignment |
| `POST` | `/api/stitch/{id}` | ffmpeg audio remux |
| `GET` | `/api/video/{id}` | Stream dubbed MP4 (range requests) |
| `GET` | `/api/video/{id}/original` | Stream original MP4 |
| `GET` | `/api/captions/{id}` | Translated WebVTT |
| `GET` | `/api/captions/{id}/original` | Original English WebVTT |
| `GET` | `/api/audio/{id}` | TTS WAV audio |
| `GET` | `/api/videos` | Video catalog from registry |
| `GET` | `/healthz` | Health check |

---

## Adding a new video

Edit `video_registry.yml` at the repo root:

```yaml
videos:
  - id: dQw4w9WgXcQ                          # YouTube video ID (from the URL)
    title: "Rick Astley Never Gonna Give You Up"  # used as the filename stem
    url: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    source_language: en
    target_language: es
```

Then restart the API container to reload the registry:

```bash
docker compose --profile nvidia restart api
```

The `title` string becomes the filename stem for every artifact: `{title}.mp4`, `{title}.json`, `{title}.vtt`. Avoid colons and slashes in titles.

---

## Adding a speaker reference voice

Place WAV files in `pipeline_data/speakers/{lang}/`. The Chatterbox container mounts this directory at `/app/voices/`.

```
pipeline_data/speakers/
├── es/
│   ├── default.wav          # used for all segments when no speaker ID is known
│   └── SPEAKER_00.wav       # optional: assigned to SPEAKER_00 after diarization
└── default.wav              # global fallback if no language-specific default exists
```

Voice resolution order (per segment, after diarization):
1. `speakers/{lang}/{speaker_id}.wav` — speaker-specific file if it exists
2. `speakers/{lang}/default.wav` — language default
3. `speakers/default.wav` — global fallback

Reference audio should be clean speech, mono or stereo, at least 5–10 seconds. WAV or MP3.

---

## Development

### Running the API locally (no Docker)

Python 3.11 is required. Install [uv](https://docs.astral.sh/uv/) first.

```bash
# Install base dependencies
uv sync

# Install the optional alignment group (VAD + diarization)
uv sync --group alignment

# Start the API (connects to STT/TTS containers if they are running)
uv run uvicorn api.src.main:app --host 0.0.0.0 --port 8080 --reload
```

### Using the Python SDK from a notebook

```python
from foreign_whispers.client import FWClient

fw = FWClient()                  # connects to http://localhost:8080
result = fw.run_pipeline(
    "https://www.youtube.com/watch?v=GYQ5yGV_-Oc",
    alignment=True,
)
print(result["video_id"])
```

Register the uv-managed kernel in VS Code / Jupyter:

```bash
uv pip install ipykernel
uv run python -m ipykernel install --user --name foreign-whispers
```

Then select the **foreign-whispers** kernel in the kernel picker.

### Running tests

```bash
# All tests (pure-Python, no GPU required)
uv run pytest

# Skip tests that need silero-vad or pyannote
uv run pytest -m "not requires_silero and not requires_pyannote"

# Run with alignment group installed
uv run --group alignment pytest
```

Tests that need optional deps are marked:
- `@pytest.mark.requires_silero` — needs `silero-vad` + `torch`
- `@pytest.mark.requires_pyannote` — needs `pyannote.audio` and `FW_HF_TOKEN`

---

## Troubleshooting

**GPU not detected by Docker**

```bash
# Confirm nvidia-container-toolkit is installed and the runtime is registered
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

If that fails, install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and restart the Docker daemon.

**Chatterbox requests timing out / silent segments**

The Chatterbox container needs time to load its model weights on first start. Wait ~2 minutes after `docker compose up` before running TTS. If you see timeout errors mid-run, check `FW_TTS_WORKERS=1` is set (it is by default in `docker-compose.override.yml`) and that the per-request timeout is at least 180 s. Tail Chatterbox logs:

```bash
docker compose --profile nvidia logs -f chatterbox-gpu
```

**Diarization returns no speakers**

Speaker diarization requires `FW_HF_TOKEN` (mapped from `HF_TOKEN` in `.env`). Confirm it is set:

```bash
docker compose --profile nvidia exec api printenv FW_HF_TOKEN
```

Also ensure you have accepted the model license on HuggingFace at <https://huggingface.co/pyannote/speaker-diarization-3.1>.

**Port already in use**

The four services bind ports 8000, 8020, 8080, and 8501. If any are taken:

```bash
lsof -i :8000   # find the conflicting process
```

Then either stop the conflicting process or change the host port in `docker-compose.override.yml` (e.g. `"8081:8080"`).

**yt-dlp fails with "Sign in to confirm you're not a bot"**

Some videos require authentication. Place a Netscape-format `cookies.txt` file at the repo root (yt-dlp reads it automatically via the volume mount in `docker-compose.yml`). Export cookies from your browser using a browser extension such as [Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc).

**Files in `pipeline_data/` are owned by root**

An older container run created them as root. Fix:

```bash
sudo chown -R $(id -u):$(id -g) pipeline_data/
```

Then ensure `UID` and `GID` in `.env` match your host user (`id -u && id -g`).

**Translation fails with "no json files found"**

The transcription step must complete before translation. Check that `pipeline_data/api/transcriptions/whisper/{title}.json` exists. If not, re-run `POST /api/transcribe/{id}`.

---

## Project layout

```
foreign-whispers/
├── api/src/                     # FastAPI backend
│   ├── main.py                  # App factory, lazy model loading
│   ├── core/config.py           # Pydantic settings (FW_ env prefix)
│   ├── routers/                 # Thin HTTP handlers (one file per stage)
│   ├── services/                # Business logic (HTTP-agnostic)
│   ├── schemas/                 # Pydantic request/response models
│   └── inference/               # WhisperBackend / TTSBackend ABCs + HTTP impls
├── foreign_whispers/            # Standalone alignment library
│   ├── alignment.py             # global_align, global_align_dp, SegmentMetrics
│   ├── backends.py              # DurationAwareTTSBackend ABC
│   ├── diarization.py           # pyannote.audio wrapper + torchaudio shim
│   ├── vad.py                   # Silero VAD wrapper
│   ├── reranking.py             # Translation re-ranking (rule-based + MarianMT)
│   ├── evaluation.py            # clip_evaluation_report, dubbing_scorecard
│   ├── voice_resolution.py      # Speaker WAV resolution for voice cloning
│   └── client.py                # FWClient SDK (drives pipeline over HTTP)
├── frontend/                    # Next.js + shadcn/ui
│   └── src/
│       ├── components/          # Pipeline cards, video player, transcript views
│       ├── hooks/use-pipeline.ts # Pipeline state machine
│       └── lib/api.ts           # API client
├── notebooks/                   # Jupyter notebooks (one per pipeline stage)
├── tests/                       # pytest suite (28 files)
├── docs/
│   ├── architecture-report.md   # Full technical architecture report
│   └── tts-temporal-alignment-research.md  # Literature survey
├── pipeline_data/               # Runtime artifacts (volume-mounted, git-ignored)
│   └── api/                     # videos/, transcriptions/, translations/, tts_audio/, …
├── video_registry.yml           # Video catalog (single source of truth)
├── docker-compose.yml           # Service definitions
├── docker-compose.override.yml  # macOS/bridge networking overrides
├── Dockerfile                   # API container (python:3.11-slim)
├── Makefile                     # Lifecycle shortcuts (make up / down / build / logs)
└── pyproject.toml               # Python package + uv dependency groups
```

---

## License

Source-available under AGPL-3.0 with a Commons Clause restriction. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms.
