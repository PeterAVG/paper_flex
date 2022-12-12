from functools import wraps
from time import time
from typing import Any, Callable, Generator, List


def timing(print_output: bool = False) -> Callable:
    def timing_decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrap(*args: Any, **kw: Any) -> Any:
            ts = time()
            result = f(*args, **kw)
            te = time()
            text = "func:%r took: %2.4f sec" % (f.__name__, te - ts)
            text += f" | Result: {result})" if print_output else ""
            print(text)
            return result

        return wrap

    return timing_decorator


def chunks(lst: List[Any], n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
