"""Run a function for every language, in several processes, within the available memory."""

import multiprocessing
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from glob import glob

import psutil
import pyarrow.parquet as pq

from .utils import SAMPLED_PICKLE_DIR

# Estimated peak memory per language of each pipeline step: GB per unit of the
# step's input, measured on the largest languages and rounded up. Inputs are the
# subsampled treebank pickle (MB on disk) and the language's largest feature table
# (million cells, rows x columns, from the parquet metadata).
GB_PER_PICKLE_MB = 0.03
GB_PER_MILLION_FEATURE_CELLS = 0.11
BASE_GB = 0.5

STEP_MEMORY = {
    "extract_features": {"pickle": GB_PER_PICKLE_MB},
    "fit_validation_classifiers": {"pickle": GB_PER_PICKLE_MB},
    "fit_dts": {"features": GB_PER_MILLION_FEATURE_CELLS},
    # a lexical DT may be fitted while the treebank is loaded
    "create_pairs": {"pickle": GB_PER_PICKLE_MB, "features": GB_PER_MILLION_FEATURE_CELLS},
}


def _pickle_mb(lang: str, data_dir: str) -> float:
    path = os.path.join(data_dir, SAMPLED_PICKLE_DIR, f"{lang.replace(' ', '_')}.pickle")
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


def _feature_million_cells(lang: str, data_dir: str) -> float:
    cells = 0
    for path in glob(os.path.join(data_dir, "treebank_features", "*", f"{lang.replace(' ', '_')}.parquet")):
        metadata = pq.read_metadata(path)
        cells = max(cells, metadata.num_rows * metadata.num_columns)
    return cells / 1e6


def estimate_memory(step: str, lang: str, data_dir: str) -> float:
    """Estimated peak memory (GB) of running `step` for one language."""
    sizes = {"pickle": _pickle_mb, "features": _feature_million_cells}
    return BASE_GB + sum(
        coefficient * sizes[source](lang, data_dir) for source, coefficient in STEP_MEMORY[step].items()
    )


@contextmanager
def _single_threaded_workers():
    """One thread per worker process (set before they start), as the processes already use all cores."""
    variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    previous = {var: os.environ.get(var) for var in variables}
    os.environ.update({var: "1" for var in variables})
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def parse_workers(value: str) -> int:
    """argparse type for --workers: a number, or "auto" for one per CPU core (memory permitting)."""
    return (os.cpu_count() or 1) if value == "auto" else int(value)


def map_languages(
    fn,
    langs: list[str],
    workers: int = 1,
    on_result=None,
    step: str | None = None,
    memory_reserve: float = 6.0,
    **kwargs,
) -> dict:
    """
    Call fn(lang, **kwargs) for every language, in at most `workers` processes, and
    return {lang: result}. fn must be a module-level function.

    With `step` (a key of STEP_MEMORY; kwargs must then include data_dir), a
    language is only started while the estimated memory of the running languages
    and its own fit in the memory that was available at the start, minus
    memory_reserve; a language that does not fit waits for others to finish, and
    runs on its own if it needs more than that.

    on_result(lang, result), if given, is called in the main process as soon as a
    language is done. A language that fails is reported and left out, so that the
    other languages still finish.
    """
    results = {}

    def finish(lang, get_result):
        try:
            results[lang] = get_result()
        except Exception:
            print(f"[{lang}] FAILED\n{traceback.format_exc()}", flush=True)
            return
        if on_result is not None:
            on_result(lang, results[lang])

    if workers <= 1:
        for lang in langs:
            finish(lang, lambda: fn(lang, **kwargs))
        return results

    estimates = {lang: estimate_memory(step, lang, kwargs["data_dir"]) if step else 0.0 for lang in langs}
    budget = psutil.virtual_memory().available / 1e9 - memory_reserve
    queue = list(langs)
    running = {}  # future -> lang

    # A fresh process per language: a process keeps the memory it has used, which
    # the memory budget does not account for.
    context = multiprocessing.get_context("spawn")
    with _single_threaded_workers(), ProcessPoolExecutor(
        max_workers=workers, mp_context=context, max_tasks_per_child=1
    ) as executor:
        while queue or running:
            in_use = sum(estimates[lang] for lang in running.values())
            # Start the waiting languages that fit (smaller ones may go ahead of a
            # large one); a language too large for the budget runs on its own.
            for lang in list(queue):
                if len(running) >= workers:
                    break
                if in_use + estimates[lang] <= budget or not running:
                    queue.remove(lang)
                    running[executor.submit(fn, lang, **kwargs)] = lang
                    in_use += estimates[lang]
                    print(
                        f"[{lang}] started (~{estimates[lang]:.1f} GB; {in_use:.1f} of {budget:.1f} GB in use)",
                        flush=True,
                    )

            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                finish(running.pop(future), future.result)

    return results
