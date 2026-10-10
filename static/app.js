// Dolphin chat UI — mobile-first, streaming, agentic, file uploads.

(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const STATE = {
    sessions: [],            // [{id, title, messages: [{role, content, files?}]}]
    currentId: null,
    abortCtrl: null,
    pendingFiles: [],       // [{id, name, size, chars}]
    busy: false,
    modelReady: false,
    tools: true,
  };

  const KEY = "dolphin-state-v1";

  // ---------------------------------------------------------------------------
  // DOM helpers
  // ---------------------------------------------------------------------------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === "class") el.className = v;
      else if (k === "dataset") Object.entries(v).forEach(([dk, dv]) => el.dataset[dk] = dv);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
      else el.setAttribute(k, v);
    });
    el.append(...kids);
    return el;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));

  // ---------------------------------------------------------------------------
  // Persistence
  // ---------------------------------------------------------------------------
  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify({
        sessions: STATE.sessions,
        currentId: STATE.currentId,
        tools: STATE.tools,
      }));
    } catch (e) { console.warn("save failed", e); }
  }
  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return;
      const d = JSON.parse(raw);
      STATE.sessions = d.sessions || [];
      STATE.currentId = d.currentId || null;
      STATE.tools = d.tools !== false;
      if (!STATE.sessions.length) STATE.currentId = null;
    } catch (e) { console.warn("load failed", e); }
  }

  // ---------------------------------------------------------------------------
  // Sessions
  // ---------------------------------------------------------------------------
  function newSession() {
    const s = { id: rid(), title: "New chat", messages: [] };
    STATE.sessions.unshift(s);
    STATE.currentId = s.id;
    save();
    renderSessionList();
    renderMessages();
    closeSidebar();
  }

  function switchSession(id) {
    STATE.currentId = id;
    save();
    renderSessionList();
    renderMessages();
    closeSidebar();
  }

  function deleteSession(id) {
    STATE.sessions = STATE.sessions.filter((s) => s.id !== id);
    if (STATE.currentId === id) STATE.currentId = STATE.sessions[0]?.id || null;
    if (!STATE.currentId) newSession(); else { renderSessionList(); renderMessages(); }
    save();
  }

  function currentSession() {
    return STATE.sessions.find((s) => s.id === STATE.currentId) || null;
  }

  function rid() { return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4); }

  // ---------------------------------------------------------------------------
  // UI: session list
  // ---------------------------------------------------------------------------
  function renderSessionList() {
    const ul = $("#sessionList");
    ul.innerHTML = "";
    if (!STATE.sessions.length) {
      ul.append(h("li", { style: "color: var(--text-faint); cursor: default;" }, "No sessions yet"));
      return;
    }
    STATE.sessions.forEach((s) => {
      const li = h("li", {
        class: "session" + (s.id === STATE.currentId ? " active" : ""),
        onclick: () => switchSession(s.id),
      }, s.title || "New chat");
      const del = h("button", {
        class: "icon-btn",
        style: "float: right; width: 28px; height: 28px;",
        onclick: (e) => { e.stopPropagation(); deleteSession(s.id); },
        "aria-label": "Delete session",
      }, "×");
      del.style.fontSize = "16px";
      li.append(del);
      ul.append(li);
    });
  }

  // ---------------------------------------------------------------------------
  // UI: messages
  // ---------------------------------------------------------------------------
  function renderMessages() {
    const wrap = $("#messages");
    wrap.innerHTML = "";
    const sess = currentSession();
    if (!sess || !sess.messages.length) {
      wrap.append(h("div", { class: "empty-state" },
        h("div", { class: "big" }, "🐬"),
        h("div", {}, "Dolphin 2.9.3 Qwen2 0.5B"),
        h("div", { style: "margin-top:8px; font-size:12px;" }, "Tiny, on-device-style AI in your browser."),
      ));
      return;
    }
    sess.messages.forEach((m) => {
      if (m.role === "user") {
        const node = $("#msg-user").content.cloneNode(true);
        node.querySelector(".msg-content").innerHTML = md(m.content);
        if (m.files?.length) {
          node.querySelector(".msg-content").append(
            h("div", { style: "margin-top:8px; font-size:12px; opacity:0.7;" },
              m.files.map((f) => "📎 " + f.name).join("  ")),
          );
        }
        wrap.append(node);
      } else if (m.role === "assistant") {
        const node = $("#msg-ai").content.cloneNode(true);
        node.querySelector(".msg-content").innerHTML = md(m.content);
        const toolsBox = node.querySelector('[data-role="tools"]');
        if (m.toolCalls?.length) {
          m.toolCalls.forEach((tc) => {
            const ti = $("#msg-tool").content.cloneNode(true);
            ti.querySelector(".tool-name").textContent = tc.name;
            ti.querySelector(".tool-args").textContent = JSON.stringify(tc.arguments || {}, null, 2);
            if (tc.result !== undefined) {
              const st = ti.querySelector(".tool-status");
              st.textContent = tc.ok ? "✓ done" : "✗ error";
              st.classList.add(tc.ok ? "ok" : "err");
              ti.querySelector(".tool-result").textContent = String(tc.result).slice(0, 800);
              ti.querySelector(".tool-result").classList.remove("hidden");
            }
            toolsBox.append(ti);
          });
        } else { toolsBox.remove(); }
        wrap.append(node);
      }
    });
    scrollDown();
  }

  function md(text) {
    if (!text) return "";
    try {
      return marked.parse(text, { breaks: true, gfm: true });
    } catch {
      return esc(text).replace(/\n/g, "<br>");
    }
  }

  function scrollDown() {
    const wrap = $("#messages");
    requestAnimationFrame(() => { wrap.scrollTop = wrap.scrollHeight; });
  }

  function appendUserMessage(text, files = []) {
    const sess = currentSession();
    if (!sess) return;
    sess.messages.push({ role: "user", content: text, files });
    if (sess.messages.length === 1) sess.title = text.slice(0, 40);
    save();
    renderMessages();
    renderSessionList();
  }

  function appendAssistantPlaceholder() {
    const sess = currentSession();
    if (!sess) return null;
    const m = { role: "assistant", content: "", toolCalls: [] };
    sess.messages.push(m);
    renderMessages();
    return m;
  }

  // ---------------------------------------------------------------------------
  // Status badge
  // ---------------------------------------------------------------------------
  function setStatus(text, cls = "") {
    const b = $("#statusBadge");
    b.textContent = text;
    b.className = "subtitle " + cls;
  }

  async function pingModel() {
    setStatus("checking…", "loading");
    for (let i = 0; i < 30; i++) {
      try {
        const r = await fetch("/health");
        const d = await r.json();
        if (d.model_loaded) { STATE.modelReady = true; setStatus("ready", "ready"); return; }
        if (i === 0) setStatus("downloading model…", "loading");
      } catch (e) {}
      await new Promise((r) => setTimeout(r, 1000));
    }
    setStatus("offline", "error");
  }

  // ---------------------------------------------------------------------------
  // File upload
  // ---------------------------------------------------------------------------
  async function uploadFiles(fileList) {
    const out = [];
    if (!fileList.length) return out;
    const fd = new FormData();
    for (const f of fileList) fd.append("files", f);
    try {
      const r = await fetch("/api/files", { method: "POST", body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      for (const f of d.files) out.push(f);
    } catch (e) {
      alert("Upload failed: " + e.message);
    }
    return out;
  }

  function renderAttachments() {
    const wrap = $("#attachments");
    wrap.innerHTML = "";
    STATE.pendingFiles.forEach((f, idx) => {
      const node = $("#attachment-chip").content.cloneNode(true);
      node.querySelector(".chip-name").textContent = "📎 " + f.name;
      node.querySelector(".chip-x").onclick = () => {
        STATE.pendingFiles.splice(idx, 1);
        renderAttachments();
      };
      wrap.append(node);
    });
  }

  // ---------------------------------------------------------------------------
  // Send — agentic SSE
  // ---------------------------------------------------------------------------
  async function send() {
    if (STATE.busy) return;
    const input = $("#prompt");
    const text = input.value.trim();
    if (!text && !STATE.pendingFiles.length) return;

    // Upload pending files first
    const fileIds = STATE.pendingFiles.map((f) => f.id);
    const fileNames = STATE.pendingFiles.map((f) => f.name);

    appendUserMessage(text || "(file attached)", STATE.pendingFiles);
    STATE.pendingFiles = [];
    renderAttachments();
    input.value = "";
    autoGrow();

    const aiMsg = appendAssistantPlaceholder();
    STATE.busy = true;
    setBusy(true);

    // Build messages payload — send last 8 messages to keep ctx small
    const sess = currentSession();
    const history = sess.messages.slice(-9, -1).map((m) => ({ role: m.role, content: m.content }));
    history.push({ role: "user", content: text });

    const body = {
      messages: history,
      file_ids: fileIds,
      enable_tools: STATE.tools,
      temperature: 0.7,
      max_tokens: 600,
    };

    try {
      STATE.abortCtrl = new AbortController();
      const r = await fetch("/api/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: STATE.abortCtrl.signal,
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      await streamAgentResponse(r, aiMsg);
    } catch (e) {
      if (e.name === "AbortError") aiMsg.content = "_⏹ stopped_";
      else aiMsg.content = "**Error:** " + e.message;
      renderMessages();
    } finally {
      STATE.busy = false;
      setBusy(false);
      STATE.abortCtrl = null;
      save();
    }
  }

  async function streamAgentResponse(resp, aiMsg) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let done = false;
    let partialEvent = null;
    let partialData = "";

    while (!done) {
      const { value, done: d } = await reader.read();
      done = d;
      if (value) buf += decoder.decode(value, { stream: !done });
      // Split on double newlines
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        handleSSEBlock(chunk, aiMsg);
      }
    }
    // handle trailing
    if (buf.trim()) handleSSEBlock(buf, aiMsg);
    renderMessages();
  }

  function handleSSEBlock(block, aiMsg) {
    let eventName = "message";
    let data = "";
    block.split("\n").forEach((line) => {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    });
    if (!data) return;
    let parsed;
    try { parsed = JSON.parse(data); } catch { parsed = { raw: data }; }

    switch (eventName) {
      case "delta": {
        aiMsg.content += parsed.content || "";
        // Update last AI bubble live
        const bubbles = $$("#messages .msg.ai");
        if (bubbles.length) {
          const last = bubbles[bubbles.length - 1];
          last.querySelector(".msg-content").innerHTML = md(aiMsg.content);
          scrollDown();
        }
        break;
      }
      case "tool_call": {
        aiMsg.toolCalls.push({ name: parsed.name, arguments: parsed.arguments, result: undefined });
        renderMessages();
        break;
      }
      case "tool_result": {
        const tc = aiMsg.toolCalls[aiMsg.toolCalls.length - 1];
        if (tc) { tc.result = parsed.result; tc.ok = parsed.ok; }
        renderMessages();
        break;
      }
      case "final": {
        // Server may emit a final payload — we already streamed deltas.
        break;
      }
      case "done": {
        break;
      }
      case "error": {
        aiMsg.content += `\n\n**Error:** ${parsed.message || "unknown"}`;
        renderMessages();
        break;
      }
    }
  }

  function setBusy(busy) {
    $("#sendBtn").classList.toggle("hidden", busy);
    $("#stopBtn").classList.toggle("hidden", !busy);
    $("#prompt").disabled = false;
    if (!busy) $("#prompt").focus();
  }

  function stop() {
    if (STATE.abortCtrl) STATE.abortCtrl.abort();
  }

  // ---------------------------------------------------------------------------
  // Input behaviour
  // ---------------------------------------------------------------------------
  function autoGrow() {
    const ta = $("#prompt");
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
  }

  // ---------------------------------------------------------------------------
  // Sidebar
  // ---------------------------------------------------------------------------
  function openSidebar() { $("#sidebar").classList.add("open"); }
  function closeSidebar() { $("#sidebar").classList.remove("open"); }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    load();
    if (!STATE.sessions.length) newSession();
    else { renderSessionList(); renderMessages(); }
    $("#toolsToggle").checked = STATE.tools;
    pingModel();

    // Bind events
    $("#newChatBtn").addEventListener("click", newSession);
    $("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
    $("#sidebarBackdrop").addEventListener("click", closeSidebar);
    $("#clearAllBtn").addEventListener("click", () => {
      if (!confirm("Clear ALL sessions and localStorage?")) return;
      localStorage.removeItem(KEY);
      STATE.sessions = [];
      STATE.currentId = null;
      newSession();
    });
    $("#toolsToggle").addEventListener("change", (e) => {
      STATE.tools = e.target.checked; save();
    });
    $("#attachBtn").addEventListener("click", () => $("#fileInput").click());
    $("#fileInput").addEventListener("change", async (e) => {
      const files = Array.from(e.target.files || []);
      if (!files.length) return;
      const uploaded = await uploadFiles(files);
      STATE.pendingFiles.push(...uploaded);
      renderAttachments();
      e.target.value = "";
    });
    $("#inputBar").addEventListener("submit", (e) => { e.preventDefault(); send(); });
    $("#stopBtn").addEventListener("click", stop);
    const ta = $("#prompt");
    ta.addEventListener("input", autoGrow);
    ta.addEventListener("keydown", (e) => {
      // Enter to send on desktop (no shift), mobile uses keyboard enterkeyhint="send"
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing && window.matchMedia("(min-width: 768px)").matches) {
        e.preventDefault();
        send();
      }
    });

    // Service worker for offline shell
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
