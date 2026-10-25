"""Read-only runtime diagnostics shared by API views and voice commands."""

import asyncio
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.cuda import configure_cuda_dll_directories
from app.services.llm import OllamaModelMissingError, OllamaResponseError, OllamaUnavailableError


def _check(status: str, detail: str, suggestion: str | None = None) -> dict[str, str]:
    result = {"status": status, "detail": detail}
    if suggestion:
        result["suggestion"] = suggestion
    return result


def _memory() -> tuple[int | None, int | None]:
    if sys.platform != "win32":
        return None, None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return status.total_physical, status.available_physical


def _gpu() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        devices = []
        for row in rows:
            name, total, free, driver = [part.strip() for part in row.split(",", 3)]
            devices.append({"name": name, "memory_total_mb": int(total), "memory_free_mb": int(free), "driver": driver})
        return {"status": "PASS" if devices else "UNKNOWN", "devices": devices, "detail": f"Detected {len(devices)} NVIDIA GPU(s)." if devices else "No NVIDIA GPU was reported."}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {"status": "UNKNOWN", "devices": [], "detail": "Physical GPU details are unavailable from nvidia-smi."}


async def collect_diagnostics(
    request: Any | None = None, *, llm: Any | None = None, stt: Any | None = None,
    tts: Any | None = None, verify_inference: bool = False,
) -> dict[str, Any]:
    """Collect bounded, safe checks without exposing private paths or environment values."""
    configure_cuda_dll_directories()
    try:
        import ctranslate2
    except ImportError:
        ctranslate2 = None

    if request is not None:
        llm, stt, tts = request.app.state.ollama, request.app.state.stt, request.app.state.tts
    if llm is None or stt is None or tts is None:
        raise ValueError("Diagnostics require the configured LLM, STT, and TTS services.")
    settings = stt.settings
    checks: dict[str, dict[str, str]] = {
        "backend": _check("PASS", "FastAPI is responding."),
        "stt": _check("PASS" if stt.available else "FAIL", "Faster-Whisper model and dependency are available." if stt.available else "Faster-Whisper is not ready.", "Install the configured local STT model and dependencies." if not stt.available else None),
        "tts": _check("PASS" if tts.available else "FAIL", "English Piper synthesis files and dependency are available." if tts.available else "English Piper synthesis is not ready.", "Install the configured Piper voice and matching config." if not tts.available else None),
    }
    try:
        models = await llm.list_models(timeout_seconds=3.0)
        expected = settings.ollama_model if ":" in settings.ollama_model else f"{settings.ollama_model}:latest"
        installed = expected in {name if ":" in name else f"{name}:latest" for name in models}
        checks["ollama"] = _check("PASS", "Ollama is reachable.")
        checks["model"] = _check("PASS" if installed else "FAIL", f"Configured model '{settings.ollama_model}' is installed." if installed else f"Configured model '{settings.ollama_model}' is missing.", f"Run ollama pull {settings.ollama_model}." if not installed else None)
    except (OllamaUnavailableError, OllamaResponseError) as exc:
        installed = False
        checks["ollama"] = _check("FAIL", str(exc), "Start the local Ollama service.")
        checks["model"] = _check("UNKNOWN", "Model availability cannot be checked while Ollama is unavailable.")

    if verify_inference and installed:
        try:
            text, _ = await asyncio.wait_for(llm.chat([{"role": "user", "content": "Reply only with READY."}], num_predict=8), timeout=15)
            checks["inference"] = _check("PASS" if text.strip() else "FAIL", "The configured model completed a bounded inference request." if text.strip() else "The model returned an empty response.")
        except (TimeoutError, OllamaUnavailableError, OllamaModelMissingError, OllamaResponseError) as exc:
            checks["inference"] = _check("FAIL", f"Inference readiness check failed: {exc}", "Check Ollama logs, model installation, and available memory.")
    else:
        checks["inference"] = _check("UNKNOWN", "A live inference probe was not requested." if installed else "Inference cannot be checked until the model is available.")

    try:
        cuda_devices = ctranslate2.get_cuda_device_count() if ctranslate2 is not None else 0
    except RuntimeError:
        cuda_devices = 0
    libraries = {}
    for label, filename in (("cublas12", "cublas64_12.dll"), ("cudnn9", "cudnn64_9.dll")):
        try:
            ctypes.WinDLL(filename)
            libraries[label] = True
        except (OSError, AttributeError):
            libraries[label] = False
    cuda_verified = stt.last_successful_device == "cuda"
    checks["cuda"] = _check("PASS" if cuda_verified else "WARNING" if cuda_devices else "NOT CONFIGURED", "CUDA STT inference has been verified in this process." if cuda_verified else "A CUDA device may be visible, but successful STT inference has not been verified." if cuda_devices else "No CUDA device is available to CTranslate2.")

    total_ram, available_ram = _memory()
    disk = shutil.disk_usage(Path(__file__).resolve().anchor)
    gpu = _gpu()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "system": {
            "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "Unknown",
            "logical_cpu_count": os.cpu_count(),
            "ram_total_bytes": total_ram,
            "ram_available_bytes": available_ram,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "python": platform.python_version(),
            "platform": platform.system(),
            "gpu": gpu,
        },
        "stt": {
            "configured_device": settings.stt_device,
            "active_device": stt.active_device,
            "compute_type": settings.stt_compute_type,
            "active_compute_type": stt.active_compute_type,
            "last_successful_device": stt.last_successful_device,
            "fallback_reason": stt.fallback_reason,
        },
        "cuda": {
            "device_count": cuda_devices,
            "cublas12_found": libraries["cublas12"],
            "cudnn9_found": libraries["cudnn9"],
            "inference_verified": cuda_verified,
            "detail": checks["cuda"]["detail"],
        },
        "languages": {
            code: {"stt": stt.available, "llm": installed, "tts": tts.available_for(code)}
            for code in ("en", "hi")
        },
        "ollama_loaded_models": await _loaded_models(llm),
    }


async def _loaded_models(llm: Any) -> list[dict[str, int | str]]:
    try:
        return await llm.loaded_models()
    except OllamaUnavailableError:
        return []


def status_summary(report: dict[str, Any]) -> str:
    checks = report["checks"]
    labels = (("backend", "Backend"), ("ollama", "Ollama"), ("model", "Qwen model"), ("inference", "model inference"), ("stt", "speech recognition"), ("tts", "speech output"), ("cuda", "CUDA acceleration"))
    parts = [f"{label}: {checks[key]['status']}" for key, label in labels]
    return f"System status at {report['timestamp']}: " + "; ".join(parts) + ". Open Settings for full hardware details and corrective suggestions."
