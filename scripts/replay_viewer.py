#!/usr/bin/env python3
"""Serve a read-only, live replay of one VISTA run directory.

The viewer reads the PNG frames and action log a run already writes; it never
touches the harness. Bind it to localhost on the run host and forward the port:

    python scripts/replay_viewer.py --run-dir runs/run_20261018T101500Z
    ssh -N -L 8765:127.0.0.1:8765 charm
    open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

FRAME_NAME = re.compile(r"^turn_(\d+)_frame_(\d+)\.png$")
FRAME_DIRS = (
    Path("player") / "screenshots",
    Path("codex_visible") / "screenshots",
)
ACTION_LOG = Path("private") / "action_log.jsonl"


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.frames_dir = next(
            (run_dir / name for name in FRAME_DIRS if (run_dir / name).is_dir()),
            run_dir / FRAME_DIRS[0],
        )
        self.action_log = run_dir / ACTION_LOG

    def manifest(self) -> dict:
        return load_json(self.run_dir / "manifest.json")

    def frames(self) -> list[dict]:
        try:
            entries = list(self.frames_dir.iterdir())
        except OSError:
            return []
        frames = []
        for path in entries:
            match = FRAME_NAME.match(path.name)
            if match is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            frames.append(
                {
                    "name": path.name,
                    "turn": int(match.group(1)),
                    "frame": int(match.group(2)),
                    "mtime": stat.st_mtime,
                    "bytes": stat.st_size,
                }
            )
        frames.sort(key=lambda item: (item["turn"], item["frame"]))
        return frames

    def actions(self) -> list[dict]:
        entries: list[dict] = []
        try:
            with self.action_log.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value, dict):
                        entries.append(value)
        except OSError:
            return []
        by_step: dict[int, dict] = {}
        for entry in entries:
            step = entry.get("step")
            if isinstance(step, int):
                by_step[step] = entry
        return [by_step[step] for step in sorted(by_step)]

    def snapshot(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "manifest": self.manifest(),
            "frames": self.frames(),
            "actions": self.actions(),
            "server_time": time.time(),
        }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VISTA replay</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214; --panel: #171b1f; --line: #2a3138;
  --text: #e6e9ec; --muted: #8b969f; --accent: #5bc0a8;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
header { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: baseline;
  padding: 12px 18px; border-bottom: 1px solid var(--line); background: var(--panel); }
header .title { font-weight: 700; color: var(--accent); }
header .meta { color: var(--muted); }
header .meta b { color: var(--text); font-weight: 500; }
main { display: grid; grid-template-columns: minmax(320px, 640px) minmax(280px, 1fr);
  gap: 18px; padding: 18px; align-items: start; }
.stage { display: flex; flex-direction: column; gap: 10px; }
.frame { position: relative; background: #050607; border: 1px solid var(--line);
  border-radius: 6px; aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center; }
.frame img { width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; display: none; }
.frame img.ready { display: block; }
.frame .empty { color: var(--muted); padding: 24px; text-align: center; }
.controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
button { background: #22282e; color: var(--text); border: 1px solid var(--line);
  border-radius: 5px; padding: 6px 10px; font: inherit; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.on { background: #1d3a33; border-color: var(--accent); color: var(--accent); }
button:disabled { opacity: .35; cursor: default; }
.slider { flex: 1 1 160px; min-width: 120px; }
input[type=range] { width: 100%; accent-color: var(--accent); }
.turn { color: var(--muted); min-width: 130px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
.panel h2 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
.kv { display: grid; grid-template-columns: 130px 1fr; gap: 3px 10px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; word-break: break-word; }
.action { margin-top: 12px; }
.action .label { color: var(--accent); font-weight: 700; }
.action .muted { color: var(--muted); }
.action pre { margin: 6px 0 0; white-space: pre-wrap; word-break: break-word; color: var(--text); }
.list { margin-top: 12px; max-height: 46vh; overflow: auto; border-top: 1px solid var(--line); }
.row { display: grid; grid-template-columns: 54px 1fr auto; gap: 8px; padding: 5px 4px;
  border-bottom: 1px solid #1e2429; cursor: pointer; }
.row:hover { background: #1c2227; }
.row.current { background: #1d3a33; }
.row .step { color: var(--muted); }
.row .act { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .res { color: var(--muted); white-space: nowrap; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--muted); font-size: 12px; }
.badge.live { border-color: var(--accent); color: var(--accent); }
.progress-wrap { display: flex; flex-direction: column; gap: 8px;
  padding: 12px 18px; border-bottom: 1px solid var(--line); background: var(--panel); }
.progress-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.progress-head .label { color: var(--muted); text-transform: uppercase;
  letter-spacing: .08em; font-size: 12px; }
.progress-head b { color: var(--accent); font-size: 18px; }
.progress-head .note { color: var(--muted); font-size: 12px; }
.progress-track { display: grid; gap: 4px; }
.progress-cell { height: 12px; border: 1px solid var(--line);
  border-radius: 3px; background: #1d2328; }
.progress-cell.done { background: var(--accent); border-color: var(--accent); }
.progress-cell.current { border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent) inset; }
.progress-scale { display: flex; justify-content: space-between;
  color: var(--muted); font-size: 11px; }
.hint { color: var(--muted); font-size: 12px; margin-top: 10px; }
</style>
</head>
<body>
<header>
  <span class="title">VISTA replay</span>
  <span class="meta" id="h-game"></span>
  <span class="meta" id="h-model"></span>
  <span class="meta" id="h-mode"></span>
  <span class="badge" id="h-status">connecting</span>
</header>
<div class="progress-wrap">
  <div class="progress-head">
    <span class="label">Levels completed</span>
    <b id="progress-count">0 / 8</b>
    <span class="note" id="progress-note">waiting for data</span>
  </div>
  <div class="progress-track" id="progress-track"></div>
  <div class="progress-scale"><span>0</span><span id="progress-total">8</span></div>
</div>
<main>
  <section class="stage">
    <div class="frame"><div class="empty" id="empty">Waiting for the first frame…</div><img id="img" alt="game frame"></div>
    <div class="controls">
      <button id="first" title="First frame">|&lt;</button>
      <button id="prev" title="Previous frame">&#9664;</button>
      <button id="play" title="Play / pause">&#9654; play</button>
      <button id="next" title="Next frame">&#9654;|</button>
      <button id="last" title="Latest frame">&#9654;&#9654;|</button>
      <button id="follow" class="on" title="Follow the newest frame">live</button>
      <span class="turn" id="turn">no frames</span>
      <input class="slider" id="seek" type="range" min="0" max="0" value="0">
      <select id="speed" title="Playback speed">
        <option value="1">1 fps</option>
        <option value="2">2 fps</option>
        <option value="4" selected>4 fps</option>
        <option value="8">8 fps</option>
        <option value="16">16 fps</option>
      </select>
    </div>
    <div class="hint">space play/pause &middot; arrows step &middot; f live</div>
  </section>
  <aside>
    <div class="panel">
      <h2>Run</h2>
      <dl class="kv" id="run-kv"></dl>
      <div class="action" id="action"></div>
      <div class="list" id="list"></div>
    </div>
  </aside>
</main>
<script>
const $ = (id) => document.getElementById(id);
let snap = { frames: [], actions: [], manifest: {} };
let index = 0;
let follow = true;
let playing = false;
let timer = null;
let started = false;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function currentFrame() { return snap.frames[index] || null; }

function actionForTurn(turn) {
  let byStep = null;
  for (const entry of snap.actions) {
    const resultTurn = entry && entry.result && entry.result.turn;
    if (resultTurn === turn) return entry;
    if (entry && entry.step === turn - 1) byStep = entry;
  }
  return byStep;
}

function describeAction(entry) {
  if (!entry) return { name: '—', detail: '' };
  const parsed = entry.validation && entry.validation.parsed;
  let name = (parsed && parsed.action) || 'invalid';
  let detail = '';
  if (parsed && parsed.action === 'ACTION6' && parsed.x !== undefined) {
    detail = `x=${parsed.x} y=${parsed.y}`;
  }
  if (entry.validation && entry.validation.ok === false) {
    detail = entry.validation.error || 'rejected';
  }
  return { name, detail };
}

function renderHeader() {
  const m = snap.manifest || {};
  const live = !m.finished_at;
  $('h-status').textContent = live ? 'live' : (m.terminal ? 'finished' : 'stopped');
  $('h-status').className = 'badge' + (live ? ' live' : '');
  $('h-game').innerHTML = m.game_id ? `game <b>${esc(m.game_id)}</b>` : '';
  const model = m.resolved_model_ids && m.resolved_model_ids.length
    ? m.resolved_model_ids[m.resolved_model_ids.length - 1] : m.requested_model;
  $('h-model').innerHTML = model
    ? `model <b>${esc(model)}</b>${m.reasoning_effort ? ` / ${esc(m.reasoning_effort)}` : ''}` : '';
  $('h-mode').innerHTML = m.operation_mode ? `mode <b>${esc(m.operation_mode)}</b>` : '';

  const rows = [
    ['run', m.game_id], ['runtime', m.runtime], ['model', m.requested_model],
    ['effort', m.reasoning_effort], ['mode', m.operation_mode],
    ['scorecard', m.scorecard_id], ['steps', m.steps_attempted],
    ['frames', snap.frames.length], ['run dir', snap.run_dir],
  ].filter(([, value]) => value !== undefined && value !== null && value !== '');
  $('run-kv').innerHTML = rows.map(([key, value]) =>
    `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('');
}

function currentProgress() {
  const m = snap.manifest || {};
  const fallback = m.resume_progress || m.progress || { completed: 0, total: 8 };
  const frame = currentFrame();
  if (!frame) return fallback;
  const entry = actionForTurn(frame.turn);
  const result = entry && entry.result;
  const progress = result && result.progress;
  if (progress && typeof progress.completed === 'number') return progress;
  return fallback;
}

function renderProgress() {
  const progress = currentProgress();
  const total = Number(progress.total) || 8;
  const completed = Math.max(0, Math.min(total, Number(progress.completed) || 0));
  $('progress-count').textContent = `${completed} / ${total}`;
  $('progress-total').textContent = String(total);
  $('progress-track').style.gridTemplateColumns = `repeat(${total}, 1fr)`;
  const cells = [];
  for (let level = 1; level <= total; level += 1) {
    let cls = 'progress-cell';
    if (level <= completed) cls += ' done';
    else if (level === completed + 1) cls += ' current';
    cells.push(`<div class="${cls}" title="Level ${level}"></div>`);
  }
  $('progress-track').innerHTML = cells.join('');
  const m = snap.manifest || {};
  const notes = [];
  if (completed >= total) notes.push('all levels complete');
  else notes.push(`level ${completed + 1} in progress`);
  const resumed = m.resume_progress || {};
  if (m.resume_from_run && typeof resumed.completed === 'number') {
    notes.push(`resumed at ${resumed.completed}/${resumed.total || total}`);
  } else if (m.resume_from_run) {
    notes.push('resumed from previous run');
  }
  $('progress-note').textContent = notes.join(' · ');
}

let imageKey = null;

function renderFrame() {
  const frame = currentFrame();
  const img = $('img');
  const key = frame ? `${frame.name}:${frame.mtime}` : null;
  if (!frame) {
    imageKey = null;
    img.className = '';
    img.removeAttribute('src');
    $('empty').style.display = '';
    $('turn').textContent = 'no frames';
  } else {
    $('empty').style.display = 'none';
    if (key !== imageKey) {
      imageKey = key;
      img.src = `frames/${encodeURIComponent(frame.name)}?v=${frame.mtime}`;
    }
    img.className = 'ready';
    $('turn').textContent = `turn ${frame.turn} · frame ${frame.frame}`;
  }
  $('seek').max = Math.max(0, snap.frames.length - 1);
  $('seek').value = String(index);
  $('first').disabled = index <= 0;
  $('prev').disabled = index <= 0;
  $('next').disabled = index >= snap.frames.length - 1;
  $('last').disabled = index >= snap.frames.length - 1;
  $('play').textContent = playing ? '\\u23f8 pause' : '\\u25b6 play';
  $('follow').className = follow ? 'on' : '';
  renderAction();
  renderList();
  renderProgress();
}

function renderAction() {
  const frame = currentFrame();
  const entry = frame ? actionForTurn(frame.turn) : null;
  const { name, detail } = describeAction(entry);
  if (!frame || !entry) {
    $('action').innerHTML = '<span class="muted">No action recorded for this frame.</span>';
    return;
  }
  const result = entry.result || {};
  const progress = result.progress || {};
  const parts = [
    `<span class="label">${esc(name)}</span>`,
    detail ? `<span class="muted">${esc(detail)}</span>` : '',
    `<span class="muted">state=${esc(result.state ?? '?')}` +
      (progress.total ? ` · level ${esc(progress.completed ?? '?')}/${esc(progress.total)}` : '') +
      (result.error ? ` · error=${esc(result.error)}` : '') + '</span>',
  ].filter(Boolean);
  const requested = entry.requested && Object.keys(entry.requested).length
    ? `<pre>${esc(JSON.stringify(entry.requested))}</pre>` : '';
  $('action').innerHTML = parts.join(' ');
  $('action').insertAdjacentHTML('beforeend', requested);
}

let listKey = null;

function renderList() {
  const frame = currentFrame();
  const currentTurn = frame ? frame.turn : -1;
  const key = `${snap.actions.length}:${currentTurn}`;
  if (key === listKey) return;
  listKey = key;
  const rows = snap.actions.map((entry) => {
    const { name } = describeAction(entry);
    const result = entry.result || {};
    const progress = result.progress || {};
    const turn = entry.step + 1;
    const cls = turn === currentTurn ? 'row current' : 'row';
    const res = progress.total ? `${result.state || ''} ${progress.completed}/${progress.total}` : (result.state || '');
    return `<div class="${cls}" data-step="${esc(entry.step)}">` +
      `<span class="step">#${esc(entry.step)}</span>` +
      `<span class="act">${esc(name)}</span>` +
      `<span class="res">${esc(res)}</span></div>`;
  }).join('');
  const list = $('list');
  list.innerHTML = rows || '<div class="row"><span></span><span class="muted">No actions yet.</span><span></span></div>';
  list.querySelectorAll('.row[data-step]').forEach((row) => {
    row.addEventListener('click', () => {
      const turn = Number(row.dataset.step) + 1;
      let target = snap.frames.findIndex((item) => item.turn >= turn);
      if (target < 0 && snap.frames.length) target = snap.frames.length - 1;
      if (target >= 0) { follow = false; index = target; renderFrame(); }
    });
  });
}

function jump(next) {
  follow = false;
  index = Math.max(0, Math.min(snap.frames.length - 1, next));
  renderFrame();
}

function stopPlaying() {
  playing = false;
  if (timer) { clearInterval(timer); timer = null; }
  renderFrame();
}

function startPlaying() {
  if (timer) clearInterval(timer);
  playing = true;
  follow = false;
  const fps = Number($('speed').value) || 4;
  timer = setInterval(() => {
    if (index >= snap.frames.length - 1) {
      const live = !snap.manifest.finished_at;
      if (!live) { stopPlaying(); return; }
      return;
    }
    index += 1;
    renderFrame();
  }, 1000 / fps);
  renderFrame();
}

async function poll() {
  try {
    const response = await fetch('api/state', { cache: 'no-store' });
    const next = await response.json();
    const grew = next.frames.length > snap.frames.length;
    snap = next;
    renderHeader();
    if (!started && snap.frames.length) {
      started = true;
      follow = !snap.manifest.finished_at;
      index = follow ? snap.frames.length - 1 : 0;
    } else if (follow && snap.frames.length) {
      index = snap.frames.length - 1;
    } else if (grew) {
      index = Math.min(index, snap.frames.length - 1);
    }
    renderFrame();
  } catch (error) {
    $('h-status').textContent = 'disconnected';
    $('h-status').className = 'badge';
  }
}

$('first').onclick = () => jump(0);
$('prev').onclick = () => jump(index - 1);
$('next').onclick = () => { stopPlaying(); jump(index + 1); };
$('last').onclick = () => {
  const live = !snap.manifest.finished_at && !playing;
  jump(snap.frames.length - 1);
  follow = live;
  renderFrame();
};
$('play').onclick = () => { playing ? stopPlaying() : startPlaying(); };
$('follow').onclick = () => {
  follow = !follow;
  if (follow) { stopPlaying(); index = Math.max(0, snap.frames.length - 1); }
  renderFrame();
};
$('seek').oninput = (event) => jump(Number(event.target.value));
$('speed').onchange = () => { if (playing) startPlaying(); };
window.addEventListener('keydown', (event) => {
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
  if (event.key === ' ') { event.preventDefault(); playing ? stopPlaying() : startPlaying(); }
  else if (event.key === 'ArrowLeft') { event.preventDefault(); jump(index - 1); }
  else if (event.key === 'ArrowRight') { event.preventDefault(); jump(index + 1); }
  else if (event.key === 'f' || event.key === 'F') { $('follow').click(); }
});
poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    artifacts: RunArtifacts

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            body = json.dumps(self.artifacts.snapshot(), separators=(",", ":")).encode()
            self._send(body, "application/json")
        elif path.startswith("/frames/"):
            self._frame(path.removeprefix("/frames/"))
        else:
            self._send(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def _frame(self, name: str) -> None:
        if not FRAME_NAME.match(name):
            self._send(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        path = self.artifacts.frames_dir / name
        try:
            body = path.read_bytes()
        except OSError:
            self._send(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        self._send(body, "image/png", cache="no-store")

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        cache: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"not a directory: {run_dir}")
    Handler.artifacts = RunArtifacts(run_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {run_dir}")
    print(f"Open http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
