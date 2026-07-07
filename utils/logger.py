"""
utils/logger.py

Structured progress logger for the cleaning pipeline.

Features
--------
* Per-page section headers with timestamps.
* Sub-step logging with indentation.
* ETA estimation based on a rolling average of completed page times.
* Optional GPU memory reporting after load/unload events.
* Coloured output (ANSI) when stdout is a TTY; plain text otherwise.

All methods are safe to call from any thread (uses a module-level lock).
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import timedelta
from typing import Optional


_lock = threading.Lock()
_IS_TTY = sys.stdout.isatty()


# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled when not in a TTY)
# ---------------------------------------------------------------------------

class _Colour:
    RESET  = "\033[0m"  if _IS_TTY else ""
    BOLD   = "\033[1m"  if _IS_TTY else ""
    DIM    = "\033[2m"  if _IS_TTY else ""
    GREEN  = "\033[32m" if _IS_TTY else ""
    YELLOW = "\033[33m" if _IS_TTY else ""
    CYAN   = "\033[36m" if _IS_TTY else ""
    RED    = "\033[31m" if _IS_TTY else ""
    GREY   = "\033[90m" if _IS_TTY else ""


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _print(*args: object) -> None:
    with _lock:
        print(*args, flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def page_start(page_name: str, index: int, total: int) -> None:
    """Print a page section header."""
    _print(
        f"\n{_Colour.BOLD}{_Colour.CYAN}"
        f"[{_now()}] ── Page {index}/{total}: {page_name}"
        f"{_Colour.RESET}"
    )


def step(message: str) -> None:
    """Print a single pipeline step."""
    _print(f"  {_Colour.GREEN}▸{_Colour.RESET} {message}")


def sub_step(message: str) -> None:
    """Print a subordinate step (e.g. per-patch progress)."""
    _print(f"    {_Colour.GREY}· {message}{_Colour.RESET}")


def warning(message: str) -> None:
    """Print a warning."""
    _print(f"  {_Colour.YELLOW}⚠ WARNING:{_Colour.RESET} {message}")


def error(message: str) -> None:
    """Print an error."""
    _print(f"  {_Colour.RED}✖ ERROR:{_Colour.RESET} {message}")


def page_done(page_name: str, elapsed_s: float) -> None:
    """Print a page completion line."""
    _print(
        f"  {_Colour.GREEN}✔ Done{_Colour.RESET}  "
        f"{_Colour.DIM}{page_name} ({elapsed_s:.1f}s){_Colour.RESET}"
    )


def eta(completed: int, total: int, elapsed_s: float) -> None:
    """Print an ETA estimate based on average time per completed page."""
    if completed == 0:
        return
    avg = elapsed_s / completed
    remaining_s = avg * (total - completed)
    eta_str = str(timedelta(seconds=int(remaining_s)))
    _print(
        f"  {_Colour.DIM}ETA: {eta_str} "
        f"(avg {avg:.1f}s/page, {total - completed} remaining){_Colour.RESET}"
    )


def gpu_memory(label: str, mb: Optional[float]) -> None:
    """Print GPU memory usage if available."""
    if mb is None:
        return
    _print(f"  {_Colour.DIM}GPU mem ({label}): {mb:.0f} MiB{_Colour.RESET}")


def pipeline_start(total_pages: int) -> None:
    """Print the pipeline banner."""
    _print(
        f"\n{_Colour.BOLD}{'═' * 60}{_Colour.RESET}\n"
        f"{_Colour.BOLD}  AI Image Cleaner  —  {total_pages} page(s) to process{_Colour.RESET}\n"
        f"{_Colour.BOLD}{'═' * 60}{_Colour.RESET}"
    )


def pipeline_done(total_pages: int, elapsed_s: float, output_zip: str) -> None:
    """Print the pipeline completion summary."""
    _print(
        f"\n{_Colour.BOLD}{_Colour.GREEN}{'═' * 60}{_Colour.RESET}\n"
        f"{_Colour.BOLD}{_Colour.GREEN}  ✔ Pipeline complete{_Colour.RESET}\n"
        f"  Pages processed : {total_pages}\n"
        f"  Total time      : {elapsed_s:.1f}s\n"
        f"  Output          : {output_zip}\n"
        f"{_Colour.BOLD}{_Colour.GREEN}{'═' * 60}{_Colour.RESET}\n"
    )
