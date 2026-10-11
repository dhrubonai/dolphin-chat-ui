# Unchensed AI by Dhrubo

A mobile-first PWA chat UI powered by **Dolphin 2.9.3 Qwen2 0.5B** (uncensored-style fine-tune of Qwen2 0.5B). Self-contained FastAPI + `llama-cpp-python` app — designed to deploy on **botkeep**.

## What it does

- **Chat UI** — mobile-first, dark mode, safe-area aware (iPhone notch), PWA-installable on phone home screen
- **Streaming responses** — token-by-token SSE
- **Markdown + syntax highlighting** — render code blocks, tables, lists
- **File upload** — `.txt`, `.md`, `.json`, `.csv`, `.py`, `.pdf`, etc. — extracted text injected as context
- **Agentic tool calling** — model can emit `calculator`, `time_now`, `web_fetch`, `read_file`, `list_files` calls. ReAct loop, capped at `MAX_TOOL_ITERS=3`
- **OpenAI-compatible** — `/v1/chat/completions` works with any OpenAI client (streaming + non-streaming)
- **Multi-session history** in browser localStorage

## Deploy on botkeep (5 clicks)

1. Open the botkeep dashboard → **New application** → source: **GitHub** → pick `dhrubonai/dolphin-chat-ui`
2. **Application name**: `Unchensed-AI-by-Dhrubo` (this becomes your subdomain)
3. **Language**: Python 3.11+ · **Startup command**: `bash start.sh`
4. **Plan**: Free — to give it ALL your free-tier RAM/CPU/storage, make sure this is your **only** application slot on the free plan (delete any others so the entire pool goes to this slot)
5. **Persistent volume**: `/data` (≥1 GB) → click **Deploy**

After ~30 s of `pip install`, the slot boots. The first chat request triggers a ~400 MB model download from HuggingFace (cached in `/data/models/` thereafter) — expect ~60 s for the first response.

Once the slot is **Running**, click the **Domains** tab in the panel and add a domain — botkeep gives you a subdomain like:

```
https://unchensed-ai-by-dhrubo.botkeep.cloud
```

Open that URL on your phone, hit ⌠ Add to Home Screen ⌡ for the app icon.

> Note: botkeep does not hand out raw IPs. The platform routes through their own HTTPS gateway. The subdomain *is* your permanent link.

## Endpoints (when live)

| Method | Path | Description |
|--------|------|-------------|
| `GET`   | `/`                          | Chat UI (PWA) |
| `GET`   | `/health`                    | `{status, model_loaded}` |
| `GET`   | `/api/info`                  | service info |
| `GET`   | `/v1/models`                  | OpenAI-style model list |
| `POST`  | `/v1/chat/completions`        | OpenAI-compatible (supports `stream: true` SSE) |
| `POST`  | `/api/files`                  | multipart file upload, returns file IDs |
| `GET`   | `/api/files/{id}`             | file metadata + extracted text |
| `DELETE`| `/api/files/{id}`             | remove file |
| `POST`  | `/api/agent/chat`             | UI chat — SSE with custom events: `delta`, `tool_call`, `tool_result`, `done`, `error` |

## Environment variables

| Var | Default | Notes |
|-----|---------|-------|
| `SERVER_PORT` | `8080` | set by botkeep |
| `PORT` | — | generic PaaS fallback |
| `MODEL_CTX` | `4096` | context size — lower to `2048` if RAM-tight |
| `N_THREADS` | `2` | botkeep free = 1.5 vCore |
| `MAX_TOOL_ITERS` | `3` | cap on tool calls per turn |
| `MAX_FILE_CHARS` | `4000` | truncate per file in prompt |
| `HF_TOKEN` | — | not needed (model is public) |

## Local dev

```bash
pip install -r requirements.txt
DATA_DIR=./data PORT=8080 python -m uvicorn app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080
```

## Honest limitations

- **0.5B model is unreliable at structured tool calling.** The framework is in place so when you swap in a bigger Qwen2.5-7B / Llama-3.1-8B later, tools will work without code changes.
- **No vision / audio** — file picker accepts them, the model can't actually see/hear them.
- **Short context** — `MODEL_CTX=4096`; large files truncated to 4 k chars.
- **CPU-only on free tier** — ~5–15 tok/s; expect 2–8 s per short reply.

## Files

```
.
├── app.py                       # FastAPI server: streaming, files, tools, static mount
├── requirements.txt             # pinned deps
├── start.sh                     # botkeep entrypoint (SERVER_PORT > PORT > 8080)
├── README.md
└── static/
    ├── index.html               # mobile-first PWA shell, branded "Unchensed AI by Dhrubo"
    ├── styles.css               # dark mode, safe-area aware
    ├── app.js                   # streaming chat client, sessions, file upload, agentic SSE
    ├── manifest.json            # PWA manifest
    ├── sw.js                    # service worker for offline shell
    └── favicon.svg              # app icon
```
