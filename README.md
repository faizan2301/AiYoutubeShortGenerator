# AI YouTube Shorts Generator

Turn a long YouTube video (or a local file) into ranked, viral-ready vertical shorts for TikTok, Reels, and YouTube Shorts.

```text
YouTube / local video
        ↓
     yt-dlp  (or reuse cached download)
        ↓
  faster-whisper  (or reuse cached .srt)
        ↓
  Gemini / OpenAI highlight ranking
        ↓
  ffmpeg + OpenCV vertical crop
        ↓
  output/short_01.mp4 …
```

## Features

- **Two modes**
  - `api` — MuAPI handles download, transcription, ranking, and crop
  - `local` — runs on your machine with `yt-dlp`, `faster-whisper`, OpenAI or Gemini, and `ffmpeg`/`opencv`
- **Virality-aware clipping** — hooks, emotional peaks, opinion bombs, revelations, conflict, quotables, story peaks, practical value
- **Smart caching** — downloads and transcripts are reused on reruns
- **Gemini resilience** — retries with backoff, optional fallback models on temporary outages
- **Long-video support** — videos over 30 minutes are chunked with overlap
- **CLI + Python API** — use from the shell or import `generate_shorts(...)`

## Requirements

- Python 3.10+
- **API mode:** `MUAPI_API_KEY`
- **Local mode:** `ffmpeg` on your `PATH`, plus `OPENAI_API_KEY` or `GEMINI_API_KEY`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
# For --mode local:
pip install -r requirements-local.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### Example `.env` (local + Gemini)

```env
LLM_PROVIDER=gemini

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.6-flash,gemini-3.5-flash
GEMINI_MAX_RETRIES=3

LOCAL_WHISPER_MODEL=base
LOCAL_WHISPER_DEVICE=auto
LOCAL_OUTPUT_DIR=output
```

For OpenAI instead of Gemini, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.

For API mode, set `MUAPI_API_KEY`.

## Usage

### Local mode (recommended for self-hosting)

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local
```

Rendered clips land in `./output/short_01.mp4`, `short_02.mp4`, …

### Already-downloaded video

If `output/source_<youtube_id>.mp4` already exists, the download is skipped. You can also pass a local path:

```bash
python main.py "output/source_VIDEO_ID.mp4" --mode local
python main.py "/path/to/video.mp4" --mode local
```

If `output/source_<id>.srt` exists and is newer than the video, Whisper is skipped too. Gemini failures leave the transcript in place so you can retry without retranscribing.

### API mode

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Options

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --mode local \
  --num-clips 5 \
  --aspect-ratio 9:16 \
  --format 720 \
  --language en \
  --output-json result.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `api` | `api` or `local` |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | Output aspect ratio |
| `--format` | `720` | Source download height: `360` / `480` / `720` / `1080` |
| `--language` | auto | Force Whisper language (e.g. `en`) |
| `--output-json` | — | Write full result JSON to a file |

### Python API

```python
from shorts_generator import generate_shorts

result = generate_shorts(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    num_clips=3,
    aspect_ratio="9:16",
    mode="local",
)

for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

## Pipeline modes

| Step | API mode | Local mode |
|------|----------|------------|
| Download | MuAPI | `yt-dlp` (cached as `output/source_<id>.mp4`) |
| Transcription | MuAPI Whisper | `faster-whisper` (cached as `output/source_<id>.srt`) |
| Highlights | MuAPI LLM | OpenAI or Gemini (`LLM_PROVIDER`) |
| Vertical crop | MuAPI autocrop | `ffmpeg` + OpenCV face tracking |
| Output | Hosted URLs | Local `output/short_XX.mp4` paths |

## Gemini retry / fallback

When `LLM_PROVIDER=gemini`:

1. Call `GEMINI_MODEL`
2. On temporary errors (`429`, `500`, `502`, `503`, `504`), retry with exponential backoff (`2s`, `4s`, `8s`…) up to `GEMINI_MAX_RETRIES`
3. If the primary model is exhausted or returns `404`, try each model in `GEMINI_FALLBACK_MODELS`
4. Permanent errors (`401`, `403`, bad request) fail immediately — no pointless retries

Example logs:

```text
[gemini] model=gemini-3.8-flash attempt=1/3
[gemini] temporary error 503 UNAVAILABLE, retrying in 2s...
[gemini] switching to fallback model=gemini-3.6-flash
```

## Project layout

```text
AI-Youtube-Shorts-Generator/
├── main.py
├── requirements.txt
├── requirements-local.txt
├── .env.example
└── shorts_generator/
    ├── config.py
    ├── pipeline.py
    ├── highlights.py
    ├── muapi.py                 # API mode helpers
    ├── downloader.py
    ├── transcriber.py
    ├── clipper.py
    └── local/                   # --mode local backends
        ├── downloader.py
        ├── transcriber.py
        ├── llm.py
        └── clipper.py
```

## Important dependency pins (local mode)

| Package | Pin | Why |
|---------|-----|-----|
| `av` | `>=11,<19` | PyAV 19 breaks `faster-whisper` (`metadata_errors`) |
| `opencv-python` | `>=4.8,<5` | OpenCV 5 removed `CascadeClassifier` used by the local cropper |
| `google-genai` | `>=1.0` | Gemini client |

## Troubleshooting

**`yt-dlp is required` / missing local deps**  
Activate the venv and install local requirements:

```bash
source venv/bin/activate
pip install -r requirements-local.txt
```

**`[gemini] GEMINI_API_KEY is not configured.`**  
Add the key to `.env`. Never commit `.env`.

**Gemini `503 UNAVAILABLE`**  
Expected under load. The app retries and falls over to `GEMINI_FALLBACK_MODELS`. Your transcript stays in `output/`.

**Whisper produced no segments**  
Try `--language en` (or the correct ISO-639-1 code). Music-only / silent videos will also fail here.

**OpenCV / face crop errors**  
Ensure `opencv-python` is below 5:

```bash
pip install "opencv-python>=4.8.0,<5"
```

## License

MIT — see [LICENSE](LICENSE).
