from functools import wraps
from pathlib import Path
from typing import Callable, List
import inspect


def ensure_paths(*path_args):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            arg_names = func.__code__.co_varnames

            for idx, arg in enumerate(args):
                if arg_names[idx] in path_args and isinstance(arg, str):
                    args[idx] = Path(arg)

            for arg_name in path_args:
                if arg_name in kwargs and isinstance(kwargs[arg_name], str):
                    kwargs[arg_name] = Path(kwargs[arg_name])

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_arguments(**valid_options):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Fill in default values

            for arg_name, valid_vals in valid_options.items():
                if arg_name in bound_args.arguments:
                    if bound_args.arguments[arg_name] not in valid_vals:
                        raise ValueError(
                            f"Invalid {arg_name}: '{bound_args.arguments[arg_name]}'. "
                            f"Supported options are: {valid_vals}"
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@validate_arguments(
    group={1, 2},
    orthogonalization={"all-but-one", "odd-even"},
)
def _get_orthogonalized_run_labels(
    run_labels: List[str], group: int, orthogonalization: str
):
    if orthogonalization == "all-but-one":
        if len(run_labels) == 2:
            labels = run_labels if group == 2 else run_labels[::-1]
        else:
            if group == 1:
                labels = [f"orth{run}" for run in run_labels]
            else:
                labels = run_labels
    else:
        if group == 1:
            labels = ["odd", "even"]
        else:
            labels = ["even", "odd"]
    return labels
