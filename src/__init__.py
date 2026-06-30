# from src import data_extraction

# Linux GPU fix: preload cu12 libcusolver.so.11 before anything imports TensorFlow,
# otherwise TF 2.21 can't find it and silently falls back to CPU. No-op off Linux.
# See src/_cuda_preload.py for the full explanation.
try:
    from ._cuda_preload import preload_cusolver as _preload_cusolver

    _preload_cusolver()
except Exception:
    pass
