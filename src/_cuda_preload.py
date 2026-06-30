"""Preload the cu12 ``libcusolver.so.11`` so TensorFlow 2.21 can find it on Linux GPU boxes.

Why this exists
---------------
On the Linux GPU machine the venv ends up carrying **both** the cu12 and cu13 NVIDIA
wheels side by side (torch pulls a cu13 build; ``tensorflow[and-cuda]`` and other deps
pull cu12 ones). TensorFlow 2.21 was built against the cu12 SONAMEs and probes for
``libcusolver.so.11`` at startup. But the cu13 cusolver wheel only ships
``libcusolver.so.12``, and the cu12 lib dir is not on the dynamic-loader path, so TF's
``dlopen("libcusolver.so.11")`` fails and it silently falls back to CPU
(``tf.config.list_physical_devices('GPU')`` returns ``[]``).

The file we need *is* installed — ``nvidia-cusolver-cu12`` ships
``site-packages/nvidia/cusolver/lib/libcusolver.so.11``. Preloading it into the global
symbol table with ``RTLD_GLOBAL`` before TF imports makes TF's later ``dlopen`` by SONAME
succeed, regardless of ``LD_LIBRARY_PATH``. This is the only mechanism that also works
under ``uv run`` and Jupyter kernels, which bypass ``.venv/bin/activate``.

This is a no-op on macOS / non-CUDA setups (the wheel/file simply isn't there), so it is
safe to call unconditionally and early.
"""
from __future__ import annotations


def preload_cusolver() -> bool:
    """Preload cu12 ``libcusolver.so.11`` if present. Returns True if loaded, else False."""
    import ctypes
    import glob
    import os
    import sys

    if sys.platform != "linux":
        return False

    candidates: list[str] = []

    # Preferred: ask the installed cu12 cusolver wheel where its libs live.
    try:
        import nvidia.cusolver as _cusolver  # type: ignore

        pkg_dir = os.path.dirname(os.path.abspath(_cusolver.__file__))
        candidates.append(os.path.join(pkg_dir, "lib", "libcusolver.so.11"))
    except Exception:
        pass

    # Fallback: scan site-packages for any nvidia/cusolver/lib/libcusolver.so.11.
    for site_dir in {os.path.dirname(os.path.dirname(__file__))} | set(sys.path):
        candidates.extend(
            glob.glob(os.path.join(site_dir, "nvidia", "cusolver", "lib", "libcusolver.so.11"))
        )

    for so in candidates:
        if not os.path.exists(so):
            continue
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue
    return False
