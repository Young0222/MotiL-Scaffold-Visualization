"""
Minimal FastAPI backend for the Scaffold Visualization Demo.

Start with:
    uvicorn server:app --reload --port 8000

Then open index.html in a browser (or serve it via the same uvicorn).
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Allow the HTML page opened as a local file (file://) to call this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request schema ────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    data_path: str
    checkpoint_path: str
    dataset_type: str = "classification"
    panel_label: str = "BACE"
    style: str = "reference"
    top_k: int = 6
    show_counts: bool = False
    min_scaffold_size: int = 8
    output_dir: str = "scaffold_visualization_v2/outputs_custom_demo"
    python_path: str = sys.executable  # default: same interpreter as the server


# ── /run endpoint — streams stdout/stderr back as text ───────────────────────

@app.post("/run")
async def run(req: RunRequest):
    cmd = [
        req.python_path,
        "scaffold_visualization_v2/plot_scaffold_tsne.py",
        "--data-path", req.data_path,
        "--checkpoint-path", req.checkpoint_path,
        "--dataset-type", req.dataset_type,
        "--panel-label", req.panel_label,
        "--style", req.style,
        "--top-k", str(req.top_k),
        "--min-scaffold-size", str(req.min_scaffold_size),
        "--output-dir", req.output_dir,
    ]
    if req.show_counts:
        cmd.append("--show-counts")

    async def event_stream():
        yield f"[server] Running:\n{shlex.join(cmd)}\n\n"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for line in proc.stdout:
            yield line.decode(errors="replace")
        await proc.wait()
        if proc.returncode == 0:
            # Tell the client where to find the result image.
            stem = Path(req.data_path).stem
            png_rel = str(Path(req.output_dir) / f"{stem}_scaffold_tsne.png")
            yield f"\n__DONE__:{png_rel}\n"
        else:
            yield f"\n__ERROR__:exit code {proc.returncode}\n"

    return StreamingResponse(event_stream(), media_type="text/plain")


# ── /file endpoint — serve any local absolute path as a download/view ─────────

@app.get("/file")
def serve_file(path: str):
    p = Path(path)
    if not p.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(str(p))


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}
