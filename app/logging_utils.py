from time import perf_counter


def elapsed_since(started_at: float) -> float:
    return perf_counter() - started_at
