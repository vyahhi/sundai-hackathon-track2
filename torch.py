import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


def _prepend_env_flag(name: str, flag: str) -> None:
    current = os.environ.get(name, "").strip()
    if flag in current:
        return
    os.environ[name] = f"{flag} {current}".strip()


def _replace_opt_level(flags: list[str], target: str) -> list[str]:
    replaced = []
    changed = False
    for flag in flags:
        if flag.startswith("-O"):
            replaced.append(target)
            changed = True
        else:
            replaced.append(flag)
    if not changed:
        replaced.append(target)
    return replaced


max_jobs = os.environ.get("SUNDAI_MAX_JOBS")
if max_jobs:
    os.environ["MAX_JOBS"] = max_jobs

nvcc_threads = os.environ.get("SUNDAI_NVCC_THREADS")
if nvcc_threads:
    _prepend_env_flag("NVCC_PREPEND_FLAGS", f"--threads {nvcc_threads}")

current_dir = Path(__file__).resolve().parent
search_path = []
for entry in sys.path:
    if not entry:
        continue
    try:
        if Path(entry).resolve() == current_dir:
            continue
    except OSError:
        pass
    search_path.append(entry)

spec = importlib.machinery.PathFinder.find_spec("torch", search_path)
if spec is None or spec.loader is None:
    raise ImportError("Unable to locate the real torch package")

module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)

try:
    from torch.utils import cpp_extension
except Exception:
    cpp_extension = None


if cpp_extension is not None:
    _real_load = cpp_extension.load

    def _patched_load(*args, **kwargs):
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        if name == "sol_kernels":
            cpp_opt = os.environ.get("SUNDAI_SOL_CPP_OLEVEL", "1")
            cuda_opt = os.environ.get("SUNDAI_SOL_CUDA_OLEVEL", "1")
            if cpp_opt:
                kwargs["extra_cflags"] = _replace_opt_level(
                    list(kwargs.get("extra_cflags") or []),
                    f"-O{cpp_opt}",
                )
            if cuda_opt:
                kwargs["extra_cuda_cflags"] = _replace_opt_level(
                    list(kwargs.get("extra_cuda_cflags") or []),
                    f"-O{cuda_opt}",
                )
        return _real_load(*args, **kwargs)

    cpp_extension.load = _patched_load
