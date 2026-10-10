# Dolphin Chat UI

A mobile-first PWA for chatting with **Dolphin 2.9.3 Qwen2 0.5B** — chat UI + FastAPI inference server in one deployable unit. Designed to run on botkeep's free tier.

## Features

- **Chat interface** — mobile-first, dark mode, safe-area aware (notch), PWA-installable on phone home screen
- **Streaming responses** — token-by-token SSE (OpenAI-style on `/v1/chat/completions`, custom events on `/api/agent/chat`)
- **Markdown + syntax highlighting** — render code blocks, tables, lists
- **File upload** — `.txt`, `.md`, `.json`, `.csv`, `.py`, `.pdf`, etc. Extracted text is injected as context
- **Agentic tool calling** — model can call tools (calculator, time_now, web_fetch, read_file, list_files) via ChatML-style inline tags. ReAct loop, capped at 3 iterations
- **Conversation history** — multi-session, persisted in browser localStorage
- **OpenAI-compatible** — `/v1/chat/completions` works with any OpenAI client

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`   | `/`                            | Chat UI (PWA) |
| `GET`   | `/health`                       | `{status, model_loaded}` |
| `GET`   | `/api/info`                     | service info |
| `GET`   | `/v1/models`                     | OpenAI-style model list |
| `POST`  | `/v1/chat/completions`           | OpenAI-compatible (supports `stream: true` SSE) |
| `POST`  | `/api/files`                     | multipart file upload, returns file IDs |
| `GET`   | `/api/files/{id}`                | file metadata + extracted text |
| `DELETE`| `/api/files/{id}`                | remove file |
| `POST`  | `/api/agent/chat`                | UI chat — SSE with custom events: `delta`, `tool_call`, `tool_result`, `done`, `error` |

## Agentic protocol

When `enable_tools: true` is sent to `/api/agent/chat`, the system prompt tells the model it can emit:

```
<|tool_call|>{"name":"calculator","arguments":{"expression":"2+3*4"}}<|/tool_call|>
```

The server parses, executes the tool, appends a `tool_result` block to the conversation, and re-prompts the model. The loop runs up to `MAX_TOOL_ITERS=3` times before forcing a final answer.

**Honest caveat:** a 0.5B model is unreliable at structured tool calling. Expect it to work for simple cases (`calculator`, `time_now`) and break on complex chains. The framework is in place so when you swap to a bigger Qwen / Llama model later, agentic behavior will work without code changes.

## Limitations of the 0.5B model

- **No vision** — file upload accepts images but the model can't actually see them; the upload endpoint just records a stub
- **No audio** — audio uploads are stubbed
- **Short context** — `MODEL_CTX=4096`; large file attachments get truncated to 4k chars per file
- **Slow on CPU** — ~5–15 tok/s on botkeep's 1.5 vCore free tier; expect 2–8 s per short reply

## Deploy to botkeep

1. Dashboard → New application → GitHub source → pick `dhrubonai/dolphin-chat-ui`
2. **Application name**: `Moviebox API` (or whatever you want — this is just your slot name)
3. Language: Python 3.11+
4. Startup: `bash start.sh`
5. Plan: Free
6. Persistent volume: `/data` (≥1 GB)
7. Env vars (optional):
   - `MODEL_CTX=4096`
   - `N_THREADS=2`
   - `MAX_TOOL_ITERS=3`
8. Deploy → wait for `pip install` (~30 s) → first chat triggers model download (~60 s)

## Local dev

```bash
pip install -r requirements.txt
DATA_DIR=./data PORT=8080 python -m uvicorn app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080 on your phone or desktop browser
```

## Files

```
.
├── app.py                       # FastAPI server: streaming, files, tools, static mount
├── requirements.txt             # pinned deps (adds python-multipart + pypdf)
├── start.sh                     # botkeep entrypoint
├── README.md
└── static/
    ├── index.html               # mobile-first PWA shell
    ├── styles.css               # dark mode, safe-area aware
    ├── app.js                   # streaming chat client, session mgmt, file upload
    ├── manifest.json            # PWA manifest
    ├── sw.js                    # service worker for offline shell
    └── favicon.svg              # app icon
```
