"""MuseTalk lip-sync microservice — isolated Python/CUDA environment.

Wraps the persistent stdin/stdout musetalk_worker.py behind HTTP so the
thin API never loads MuseTalk weights.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("musetalk")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MUSETALK_PATH = Path(os.environ.get("MUSETALK_PATH", "/app/models/MuseTalk"))
AVATAR_ENGINE = os.environ.get("AVATAR_ENGINE", "musetalk")
AVATAR_RESOLUTION = int(os.environ.get("AVATAR_RESOLUTION", "512"))
AVATAR_FPS = int(os.environ.get("AVATAR_FPS", "25"))
WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "musetalk_worker.py"

app = FastAPI(title="AvatarAI MuseTalk", version="1.0.0")

_worker_proc: Optional[asyncio.subprocess.Process] = None
_worker_lock = asyncio.Lock()
_engine = AVATAR_ENGINE
_cuda = False


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@app.on_event("startup")
async def startup():
    global _engine, _cuda
    _cuda = _cuda_available()
    if _engine == "musetalk":
        if not (MUSETALK_PATH / "scripts" / "inference.py").exists():
            logger.warning(
                "MuseTalk not found at %s — falling back to simple engine",
                MUSETALK_PATH,
            )
            _engine = "simple"
        elif not WORKER_SCRIPT.exists():
            logger.warning("musetalk_worker.py missing — falling back to simple")
            _engine = "simple"
    logger.info("MuseTalk service ready (engine=%s, cuda=%s)", _engine, _cuda)


class AnimateRequest(BaseModel):
    image: str
    audio: str
    output: str
    engine: Optional[str] = Field(None, description="Override: musetalk | simple")


class AnimateResponse(BaseModel):
    output_path: str
    engine: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "musetalk",
        "engine": _engine,
        "worker_alive": _worker_proc is not None
        and _worker_proc.returncode is None,
        "cuda": _cuda,
        "musetalk_path": str(MUSETALK_PATH),
        "models_present": (MUSETALK_PATH / "scripts" / "inference.py").exists(),
    }


async def _ensure_worker() -> asyncio.subprocess.Process:
    global _worker_proc

    if _worker_proc is not None and _worker_proc.returncode is None:
        return _worker_proc

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(MUSETALK_PATH) + (":" + existing if existing else "")

    use_float16 = _cuda
    logger.info("Starting persistent MuseTalk worker…")
    # Don't PIPE stderr — worker progress/warnings can fill the OS pipe
    # buffer and deadlock while we wait on stdout for READY.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(WORKER_SCRIPT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
        cwd=str(MUSETALK_PATH),
        env=env,
    )

    init_msg = (
        json.dumps(
            {
                "unet_model_path": str(MUSETALK_PATH / "models" / "musetalkV15" / "unet.pth"),
                "unet_config": str(MUSETALK_PATH / "models" / "musetalkV15" / "musetalk.json"),
                "whisper_dir": str(MUSETALK_PATH / "models" / "whisper"),
                "vae_type": str(MUSETALK_PATH / "models" / "sd-vae"),
                "use_float16": use_float16,
            }
        )
        + "\n"
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(init_msg.encode())
    await proc.stdin.drain()

    timeout = 300 if _cuda else 600
    deadline = asyncio.get_event_loop().time() + timeout
    ready = False
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            proc.kill()
            raise RuntimeError("MuseTalk worker timed out while loading models")
        try:
            ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("MuseTalk worker timed out while loading models")

        if proc.returncode is not None and not ready_line:
            raise RuntimeError(
                f"Worker exited during startup (code={proc.returncode})"
            )
        if not ready_line:
            raise RuntimeError("Worker closed stdout before READY")

        line = ready_line.decode(errors="replace").strip()
        if line.startswith("READY"):
            ready = True
            break
        # MuseTalk's load_all_model prints progress to stdout — ignore until READY
        if line:
            logger.info("worker: %s", line)

    if not ready:
        proc.kill()
        raise RuntimeError("Worker failed to become READY")

    logger.info("MuseTalk worker ready")
    _worker_proc = proc
    return proc


async def _worker_infer(
    image_path: str, audio_path: str, output_path: str, coord_cache: Optional[str]
) -> str:
    global _worker_proc
    async with _worker_lock:
        proc = await _ensure_worker()
        assert proc.stdin and proc.stdout

        job = (
            json.dumps(
                {
                    "image": str(Path(image_path).resolve()),
                    "audio": str(Path(audio_path).resolve()),
                    "output": str(Path(output_path).resolve()),
                    "coord_cache": coord_cache,
                }
            )
            + "\n"
        )

        try:
            proc.stdin.write(job.encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            proc.kill()
            _worker_proc = None
            raise RuntimeError(f"MuseTalk worker pipe is dead: {e}") from e

        infer_timeout = 180 if _cuda else 300
        deadline = asyncio.get_event_loop().time() + infer_timeout
        result = None
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                proc.kill()
                _worker_proc = None
                raise RuntimeError(f"MuseTalk inference timed out after {infer_timeout}s")
            try:
                result_line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                proc.kill()
                _worker_proc = None
                raise RuntimeError(f"MuseTalk inference timed out after {infer_timeout}s")

            if not result_line:
                proc.kill()
                _worker_proc = None
                raise RuntimeError("MuseTalk worker exited before returning a result")

            line = result_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                logger.info("worker: %s", line)
                continue
            if isinstance(parsed, dict) and "status" in parsed:
                result = parsed
                break
            logger.info("worker: %s", line)

        if result is None:
            raise RuntimeError("MuseTalk worker returned no result")
        if result["status"] != "ok":
            raise RuntimeError(result.get("msg", "Unknown worker error"))
        return output_path


async def _animate_simple(image_path: str, audio_path: str, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").is_file() else "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        "-vf",
        (
            f"fps={AVATAR_FPS},"
            f"scale={AVATAR_RESOLUTION}:{AVATAR_RESOLUTION}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={AVATAR_RESOLUTION}:{AVATAR_RESOLUTION}:(ow-iw)/2:(oh-ih)/2"
        ),
        output_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Simple animation failed: {stderr.decode(errors='replace')}")
    return output_path


@app.post("/animate", response_model=AnimateResponse)
async def animate(body: AnimateRequest):
    if not Path(body.image).is_file():
        raise HTTPException(404, f"Image not found: {body.image}")
    if not Path(body.audio).is_file():
        raise HTTPException(404, f"Audio not found: {body.audio}")

    engine = body.engine or _engine
    Path(body.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        if engine == "musetalk":
            avatar_id = hashlib.md5(str(Path(body.image).resolve()).encode()).hexdigest()
            coord_cache = str(MUSETALK_PATH / "results" / "coords" / f"{avatar_id}.pkl")
            os.makedirs(os.path.dirname(coord_cache), exist_ok=True)
            await _worker_infer(body.image, body.audio, body.output, coord_cache)
            return AnimateResponse(output_path=body.output, engine="musetalk")
        else:
            await _animate_simple(body.image, body.audio, body.output)
            return AnimateResponse(output_path=body.output, engine="simple")
    except Exception as e:
        logger.error("Animation failed (%s): %s — falling back to simple", engine, e)
        await _animate_simple(body.image, body.audio, body.output)
        return AnimateResponse(output_path=body.output, engine="simple")
