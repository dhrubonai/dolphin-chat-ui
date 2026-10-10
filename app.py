"""
Dolphin 2.9.3 Qwen2 0.5B — chat UI + OpenAI-compatible inference server.

Single-process FastAPI app that:
  - serves a mobile-first PWA from /static/
  - exposes OpenAI-compatible /v1/chat/completions (with stream=true SSE support)
  - exposes /api/files for file uploads (text extraction for .txt/.md/.csv/.py/.pdf/etc.)
  - exposes /api/agent/chat for the UI: SSE with custom events for tool calls
    and the final answer (ReAct loop, capped at MAX_TOOL_ITERS iterations)

Designed to deploy on botkeep's free tier (2 GB RAM / 1.5 vCore / 2 GB storage).
"""

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO = "itlwas/dolphin-2.9.3-qwen2-0.5b-Q4_K_M-GGUF"
MODEL_FILE = "dolphin-2.9.3-qwen2-0.5b-q4_k_m.gguf"
MODEL_ID   = "dolphin-2.9.3-qwen2-0.5b"

_data_root = Path("/data") if Path("/data").is_dir() and os.access("/data", os.W_OK) else Path("./data")
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_data_root)))
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / MODEL_FILE
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CTX     = int(os.environ.get("MODEL_CTX", "4096"))
N_THREADS     = int(os.environ.get("N_THREADS", "2"))
N_GPU_LAYERS  = int(os.environ.get("N_GPU_LAYERS", "0"))
MAX_TOOL_ITERS = int(os.environ.get("MAX_TOOL_ITERS", "3"))
MAX_FILE_CHARS = int(os.environ.get("MAX_FILE_CHARS", "4000"))  # truncate per file in prompt
PORT          = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# ChatML template + tool-call protocol
# ---------------------------------------------------------------------------
CHATML_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
CHATML_USER   = "<|im_start|>user\n{content}<|im_end|>\n"
CHATML_ASSIST_HEADER = "<|im_start|>assistant\n"
CHATML_END    = "<|im_end|>"
DEFAULT_STOP = ["<|im_end|>", "<|im_start|>"]

# Inline tool-call tag format:  <|tool_call|>{"name":"...","arguments":{...}}<|/tool_call|>
TOOL_CALL_OPEN  = "<|tool_call|>"
TOOL_CALL_CLOSE = "<|/tool_call|>"
TOOL_RESULT_OPEN  = "<|tool_result|>"
TOOL_RESULT_CLOSE = "<|/tool_result|>"

# Primary parser: fenced ```json block containing {"tool": "...", "args": {...}}
FENCED_JSON_TOOL_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)
# Fallback: <|tool_call|>{...}<|/tool_call|>
INLINE_TOOL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(\{.*?\})" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
# Fallback: <tool name="x" arg1="v1" .../>
XML_TOOL_RE = re.compile(r'<tool\s+name="([^"]+)"([^>]*?)/>', re.DOTALL)

AGENT_SYSTEM_PROMPT = """You are Dolphin, a helpful AI assistant with tool use.

# Tools
- calculator — evaluate a math expression. args: {"expression": "2+3*4"}
- time_now — return current UTC time. args: {}
- web_fetch — fetch a URL, return first 4k chars of text. args: {"url": "https://..."}
- read_file — read an attached file by id. args: {"file_id": "..."}
- list_files — list attached files. args: {}

# How to call a tool
Write a fenced json block (nothing else in that block), then stop and wait. Example:

```json
{"tool": "calculator", "args": {"expression": "2+3*4"}}
```

The system will reply with the tool result in a `<tool_result>...</tool_result>` block. \
You may then continue reasoning or call another tool (max 3 calls per turn).

# Rules
- Be concise — the user is on a phone.
- After all tool calls are done, write your final answer in plain text (no code blocks).
- If you don't need a tool, just answer directly without any code block.
"""


def render_chatml(messages: List[Dict[str, str]]) -> str:
    """Render messages to ChatML. Roles: system/user/assistant/tool."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(CHATML_SYSTEM.format(content=content))
        elif role == "user":
            parts.append(CHATML_USER.format(content=content))
        elif role == "assistant":
            parts.append(f"{CHATML_ASSIST_HEADER}{content}{CHATML_END}\n")
        elif role == "tool":
            # tool results injected as a system note in ChatML style
            parts.append(f"<|im_start|>system\n{TOOL_RESULT_OPEN}{content}{TOOL_RESULT_CLOSE}<|im_end|>\n")
        else:
            parts.append(CHATML_USER.format(content=content))
    parts.append(CHATML_ASSIST_HEADER)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------
_llm: Any = None
_load_lock = asyncio.Lock()


def _download_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    from huggingface_hub import hf_hub_download
    print(f"[boot] downloading {MODEL_REPO}/{MODEL_FILE} -> {MODEL_PATH}", flush=True)
    local = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(local)


def _load_model() -> Any:
    from llama_cpp import Llama
    path = _download_model()
    print(f"[boot] loading model from {path} (ctx={MODEL_CTX}, threads={N_THREADS})", flush=True)
    inst = Llama(
        model_path=str(path),
        n_ctx=MODEL_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
        seed=0,
    )
    print("[boot] model ready", flush=True)
    return inst


async def get_llm() -> Any:
    global _llm
    if _llm is None:
        async with _load_lock:
            if _llm is None:
                loop = asyncio.get_event_loop()
                _llm = await loop.run_in_executor(None, _load_model)
    return _llm


# ---------------------------------------------------------------------------
# File upload + text extraction
# ---------------------------------------------------------------------------
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".py", ".js", ".ts",
    ".tsx", ".jsx", ".html", ".htm", ".css", ".xml", ".yaml", ".yml", ".sh",
    ".bash", ".log", ".ini", ".toml", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".rb", ".php", ".swift", ".kt", ".sql", ".env",
}


def _extract_text(path: Path, ext: str) -> str:
    """Extract up to MAX_FILE_CHARS characters of text from a file."""
    if ext in TEXT_EXTS:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:MAX_FILE_CHARS]
        except Exception as e:
            return f"[read error: {e}]"
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            r = PdfReader(str(path))
            txt = "\n".join((p.extract_text() or "") for p in r.pages[:10])
            return txt[:MAX_FILE_CHARS]
        except Exception as e:
            return f"[pdf extract error: {e}]"
    # binary / unsupported: return a stub
    return f"[binary file: {path.name} — content extraction not supported]"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _tool_calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression."""
    if not re.fullmatch(r"[\d\s+\-*/().,%]+", expression or ""):
        return "error: expression contains disallowed characters"
    try:
        # ast.literal_eval handles literals; we want arithmetic — use eval
        # with restricted globals.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"= {result}"
    except Exception as e:
        return f"error: {e}"


def _tool_time_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _tool_web_fetch(url: str) -> str:
    """Fetch a URL and return first 4k chars of text."""
    import urllib.request
    if not url.startswith(("http://", "https://")):
        return "error: url must start with http(s)://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dolphin-bot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(2_000_000)  # 2 MB cap
            # try to strip HTML if it looks like HTML
            text = raw.decode("utf-8", errors="ignore")
            if "<html" in text.lower() or "<body" in text.lower():
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
            return text[:4000]
    except Exception as e:
        return f"error: {e}"


def _tool_read_file(file_id: str) -> str:
    meta_path = UPLOAD_DIR / f"{file_id}.json"
    if not meta_path.exists():
        return f"error: file {file_id} not found"
    meta = json.loads(meta_path.read_text())
    return meta.get("extracted_text", "")[:MAX_FILE_CHARS]


def _tool_list_files() -> str:
    files = []
    for p in UPLOAD_DIR.glob("*.json"):
        meta = json.loads(p.read_text())
        files.append({"id": meta["id"], "name": meta["name"], "chars": meta.get("chars", 0)})
    return json.dumps(files, indent=2)


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
        if name == "calculator":
            return _tool_calculator(arguments.get("expression", ""))
        if name == "time_now":
            return _tool_time_now()
        if name == "web_fetch":
            return _tool_web_fetch(arguments.get("url", ""))
        if name == "read_file":
            return _tool_read_file(arguments.get("file_id", ""))
        if name == "list_files":
            return _tool_list_files()
        return f"error: unknown tool '{name}'"
    except Exception as e:
        return f"error: {e}"


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
app = FastAPI(title="Dolphin Chat UI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models / data ---------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[Any] = None


class AgentChatRequest(BaseModel):
    messages: List[ChatMessage]
    file_ids: List[str] = []
    enable_tools: bool = True
    temperature: float = 0.7
    max_tokens: int = 512


# --- Basic endpoints ------------------------------------------------------
@app.get("/api/info")
async def info():
    return {
        "service": "dolphin-chat-ui",
        "model": MODEL_ID,
        "model_loaded": _llm is not None,
        "tools_enabled": True,
        "max_tool_iters": MAX_TOOL_ITERS,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _llm is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "botkeep",
        }],
    }


# --- File upload ----------------------------------------------------------
@app.post("/api/files")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        fid = uuid.uuid4().hex[:16]
        safe_name = os.path.basename(f.filename or "file")
        ext = os.path.splitext(safe_name)[1].lower()
        path = UPLOAD_DIR / f"{fid}{ext}"
        with path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        text = _extract_text(path, ext)
        meta = {
            "id": fid,
            "name": safe_name,
            "size": path.stat().st_size,
            "content_type": f.content_type or "application/octet-stream",
            "extracted_text": text,
            "chars": len(text),
            "stored_path": str(path),
        }
        (UPLOAD_DIR / f"{fid}.json").write_text(json.dumps(meta, indent=2))
        saved.append({
            "id": fid,
            "name": safe_name,
            "size": meta["size"],
            "chars": meta["chars"],
            "content_type": meta["content_type"],
        })
    return {"files": saved}


@app.get("/api/files/{file_id}")
async def get_file_meta(file_id: str):
    p = UPLOAD_DIR / f"{file_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return json.loads(p.read_text())


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    meta = UPLOAD_DIR / f"{file_id}.json"
    if not meta.exists():
        raise HTTPException(status_code=404, detail="file not found")
    m = json.loads(meta.read_text())
    if m.get("stored_path") and Path(m["stored_path"]).exists():
        Path(m["stored_path"]).unlink()
    meta.unlink()
    return {"deleted": file_id}


# --- OpenAI-compatible non-streaming --------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(
            _openai_sse_stream(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    llm = await get_llm()
    stop = _normalize_stop(req.stop)
    prompt = render_chatml([m.dict() for m in req.messages])

    try:
        out = llm(
            prompt,
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.95,
            stop=stop,
            echo=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation error: {e}")

    text = out["choices"][0]["text"]
    for s in stop:
        if text.endswith(s):
            text = text[: -len(s)]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": out.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    }


async def _openai_sse_stream(req: ChatCompletionRequest):
    """OpenAI-style SSE stream: data: {delta} ... data: [DONE]."""
    llm = await get_llm()
    stop = _normalize_stop(req.stop)
    prompt = render_chatml([m.dict() for m in req.messages])
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    loop = asyncio.get_event_loop()

    def _gen():
        try:
            for chunk in llm(
                prompt,
                max_tokens=req.max_tokens or 512,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.95,
                stop=stop,
                echo=False,
                stream=True,
            ):
                delta = chunk["choices"][0].get("text", "")
                if not delta:
                    continue
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}]})}\n\n"
            # final chunk with finish_reason
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {json.dumps(err)}\n\n"

    # run sync generator in a thread, async-iterate
    queue = asyncio.Queue()

    async def _producer():
        def _push_all():
            for chunk in _gen():
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        await loop.run_in_executor(None, _push_all)

    asyncio.create_task(_producer())
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item


def _normalize_stop(stop: Any) -> List[str]:
    if stop is None:
        return list(DEFAULT_STOP)
    if isinstance(stop, str):
        return [stop] + [s for s in DEFAULT_STOP if s != stop]
    out = list(stop)
    for s in DEFAULT_STOP:
        if s not in out:
            out.append(s)
    return out


# --- Helpers used by the agentic chat endpoint ---------------------------

def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Try several formats and return parsed tool calls."""
    calls: List[Dict[str, Any]] = []

    # 1. Fenced ```json blocks containing a "tool" field (primary format)
    for m in FENCED_JSON_TOOL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("tool"):
                calls.append({"name": obj["tool"], "arguments": obj.get("args", obj.get("arguments", {}))})
        except json.JSONDecodeError:
            pass

    # 2. Inline <|tool_call|>{...}<|/tool_call|>
    if not calls:
        for m in INLINE_TOOL_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("name"):
                    calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
            except json.JSONDecodeError:
                pass

    # 3. <tool name="x" arg1="v" .../> fallback
    if not calls:
        for m in XML_TOOL_RE.finditer(text):
            name = m.group(1)
            arg_str = m.group(2)
            args = {}
            for kv in re.finditer(r'(\w+)="([^"]*)"', arg_str):
                args[kv.group(1)] = kv.group(2)
            calls.append({"name": name, "arguments": args})

    KNOWN = {"calculator", "time_now", "web_fetch", "read_file", "list_files"}
    return [c for c in calls if c["name"] in KNOWN]


def _strip_tool_blocks(text: str) -> str:
    """Remove fenced code blocks containing tool calls + inline tool tags."""
    text = FENCED_JSON_TOOL_RE.sub("", text)
    text = INLINE_TOOL_RE.sub("", text)
    text = XML_TOOL_RE.sub("", text)
    return text.strip()


def _chunk_for_stream(text: str, chunk_size: int = 4) -> List[str]:
    """Split a final answer into small chunks for simulated streaming."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# --- Agentic chat (used by the UI) ----------------------------------------
@app.post("/api/agent/chat")
async def agent_chat(req: AgentChatRequest):
    """UI-only endpoint with custom SSE events:
       event: tool_call  data: {name, arguments}
       event: tool_result data: {name, result, ok}
       event: delta       data: {content}    (streaming tokens)
       event: done        data: {usage}
    """
    return StreamingResponse(
        _agent_sse_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _agent_sse_stream(req: AgentChatRequest):
    """Generate-then-stream agentic loop.

    For each iteration:
      1. Call the model (non-streaming) to get the full text for this turn.
      2. Parse the text for tool calls (multiple formats supported).
      3. If a tool call is found:
         - Emit `tool_call` event
         - Execute the tool, emit `tool_result` event
         - Append the assistant + tool messages to the conversation
         - Loop again (up to MAX_TOOL_ITERS)
      4. If no tool call is found (or we hit the iter cap):
         - Stream the cleaned final answer as `delta` events (chunked)
         - Emit `done`
    """
    llm = await get_llm()
    loop = asyncio.get_event_loop()

    # Build message list, prepending file context to the latest user message
    messages = [m.dict() for m in req.messages]
    if req.file_ids:
        for i, m in enumerate(messages):
            if m["role"] == "user":
                file_block = await _build_file_context(req.file_ids)
                if file_block:
                    messages[i] = {**m, "content": f"{file_block}\n\n{m['content']}"}
                break

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + messages
    stop = list(DEFAULT_STOP)

    try:
        for iteration in range(MAX_TOOL_ITERS + 1):
            prompt = render_chatml(messages)

            # 1. Generate full text (non-streaming, runs in thread to avoid blocking loop)
            def _gen():
                out = llm(
                    prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=0.95,
                    stop=stop,
                    echo=False,
                    stream=False,
                )
                return out["choices"][0]["text"]
            full_text = await loop.run_in_executor(None, _gen)

            # Strip trailing stop tokens if any leaked through
            for s in stop:
                if full_text.endswith(s):
                    full_text = full_text[: -len(s)]

            # 2. Parse for tool calls
            tool_calls = _parse_tool_calls(full_text)

            # 3. If no tool calls (or we've exhausted iterations), this is the final answer
            if not tool_calls or iteration >= MAX_TOOL_ITERS:
                final_text = _strip_tool_blocks(full_text).strip() or full_text.strip()
                # Stream the final answer in small chunks for nice UX
                for chunk in _chunk_for_stream(final_text, chunk_size=6):
                    yield _sse("delta", {"content": chunk})
                    # Tiny sleep so the browser can paint between chunks
                    await asyncio.sleep(0.015)
                yield _sse("done", {"iterations": iteration + 1, "tool_calls_used": iteration})
                return

            # 4. Tool call(s) detected — emit and execute each
            for tc in tool_calls:
                yield _sse("tool_call", {"name": tc["name"], "arguments": tc["arguments"]})
                result = await loop.run_in_executor(None, execute_tool, tc["name"], tc["arguments"])
                yield _sse("tool_result", {
                    "name": tc["name"],
                    "result": result[:2000],
                    "ok": not result.startswith("error"),
                })
                # Append assistant message (with the tool call in it) and tool result to the conversation
                messages.append({"role": "assistant", "content": full_text})
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"name": tc["name"], "result": result}),
                })
            # loop continues — model will be re-prompted with tool results

    except Exception as e:
        yield _sse("error", {"message": str(e)})




async def _build_file_context(file_ids: List[str]) -> str:
    if not file_ids:
        return ""
    parts = ["The following files were attached to this message:"]
    for fid in file_ids:
        p = UPLOAD_DIR / f"{fid}.json"
        if not p.exists():
            parts.append(f"\n[file: {fid}] (not found)\n")
            continue
        m = json.loads(p.read_text())
        body = m.get("extracted_text", "")
        parts.append(f"\n[file: {m.get('name')}] (id={fid}, chars={m.get('chars', 0)})\n{body}\n[/file]\n")
    return "\n".join(parts)


# --- Mount static UI last (catch-all) -------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
