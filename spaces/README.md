---
title: Foreign Whispers
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# Foreign Whispers

Automatic AI video dubbing — paste a YouTube URL and receive a Spanish-dubbed MP4.

**Pipeline:** `yt-dlp` → `faster-whisper` STT → `argostranslate` EN→ES → `Chatterbox` TTS → `ffmpeg` stitch

> GPU hardware tier recommended. First run loads Whisper and Chatterbox models (~2 min).
