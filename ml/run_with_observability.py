from __future__ import annotations

import faulthandler
import importlib
import json
import multiprocessing
import os
import signal
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_observability.py <python-script> [args...]")

    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

    # CUDA is initialized before PyTorch's loaders are iterated. Linux's default
    # `fork` start method would clone that live CUDA process when each loader
    # lazily creates workers, which is unsafe and can wedge WSL's dxg bridge.
    multiprocessing.set_start_method("spawn", force=True)

    print(json.dumps({
        "event": "observability-ready",
        "pid": os.getpid(),
        "stackDumpSignal": "SIGUSR1" if hasattr(signal, "SIGUSR1") else None,
        "multiprocessingStartMethod": multiprocessing.get_start_method(),
        "script": str(script),
    }), flush=True)

    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    module = importlib.import_module(script.stem)
    entrypoint = getattr(module, "main", None)
    if not callable(entrypoint):
        raise RuntimeError(f"observed script has no callable main(): {script}")
    entrypoint()


if __name__ == "__main__":
    main()
