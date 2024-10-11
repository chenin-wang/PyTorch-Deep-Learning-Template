import torch
from typing import Any, Callable, Optional, Sequence, Union
from numpy.typing import NDArray
from functools import partial, wraps
import numpy as np
import inspect
import random
import types
from .timers import Timer, MultiLevelTimer

__all__ = [
    "opt_args_deco",
    "delegates",
    "map_container",
    "retry_new_on_error",
    "allow_np",
]


def opt_args_deco(deco: Callable) -> Callable:
    """Meta-decorator to make implementing of decorators with optional arguments more intuitive

    Recall: Decorators are equivalent to applying functions sequentially
        >>> func = deco(func)

    If we want to provide optional arguments, it would be the equivalent of doing:
        >>> func = deco(foo=10)(func)
    I.e. in this case, deco is actually a function that RETURNS a decorator (a.k.a. a decorator factory)

    In practice, this is typically implemented with two nested functions as opposed to one.
    Also, the "factory" must always be called, "func = deco()(func)", even if no arguments are provided.
    This is ugly, obfuscated and makes puppies cry. No one wants puppies to cry.

    This decorator "hides" one level of nesting by using the 'partial' function.
    If no optional parameters are provided, we proceed as a regular decorator using the default parameters.
    If any optional kwargs are provided, this returns the decorator that is then applied to the function (this is
    equivalent to the "deco(foo=10)" portion of the second example).

    Example (before):
    ```
        def stringify(func=None, *, prefix='', suffix=''):
            if func is None:
                return partial(stringify, prefix=prefix, suffix=suffix)

            @wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                return f'{prefix}{out}{suffix}'
            return wrapper
    ```

    Example (after):
    ```
        @opt_args_deco
        def stringify(func, prefix='', suffix=''):
            @wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                return f'{prefix}{out}{suffix}'
            return wrapper
    ```

    :param deco: (Callable) Decorator function with optional parameters to wrap.
    :return: (Callable) If `func` is provided: decorated func, otherwise: decorator to apply to `func`.
    """

    @wraps(deco)
    def wrapper(f: Optional[Callable] = None, **kwargs) -> Callable:
        # If only optional arguments are provided --> return decorator
        if f is None:
            return partial(deco, **kwargs)

        # Soft-enforcing that we provide the optional arguments as keyword only
        if not isinstance(f, (types.FunctionType, types.MethodType)):
            raise TypeError(
                f"Positional argument must be a function or method, got {f} of type {type(f)}"
            )

        # Pass kwargs to allow for programmatic decorating --> return decorated function
        return deco(f, **kwargs)

    return wrapper


def delegates(to: Optional[Callable] = None, keep: bool = False):
    """From https://www.fast.ai/2019/08/06/delegation/
    Decorator to replace `**kwargs` in signature with params from `to`.

    This can be used to decorate either a class
    ```
        @delegates()
        class Child(Parent): ...
    ```
    or a function
    ```
        @delegates(parent)
        def func(a, **kwargs): ...
    ```

    :param to: (Callable) Callable containing the params to copy
    :param keep: (bool) If `True`, keep `**kwargs` in the signature.
    :return: (Callable) The decorated class or function with the updated signature.
    """

    def wrapper(f: Union[type, Callable]) -> Callable:
        to_f, from_f = (f.__base__.__init__, f.__init__) if to is None else (to, f)
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)

        args = sigd.pop("args", None)
        if args:
            sigd2 = {
                k: v
                for k, v in inspect.signature(to_f).parameters.items()
                if v.default == inspect.Parameter.empty and k not in sigd
            }
            sigd.update(sigd2)

        kwargs = sigd.pop("kwargs", None)
        if kwargs:
            sigd2 = {
                k: v
                for k, v in inspect.signature(to_f).parameters.items()
                if v.default != inspect.Parameter.empty and k not in sigd
            }
            sigd.update(sigd2)

        if keep and args:
            sigd["args"] = args
        if keep and kwargs:
            sigd["kwargs"] = kwargs

        from_f.__signature__ = sig.replace(parameters=list(sigd.values()))
        return f

    return wrapper


def map_container(f: Callable) -> Callable:
    """Decorator to recursively apply a function to arbitrary nestings of `dict`, `list`, `tuple` & `set`

    NOTE: `f` can have an arbitrary signature, but the first arg must be the item we want to apply `f` to.

    Example:
    ```
        @map_apply
        def square(n, bias=0):
            return (n ** 2) + bias

        x = {'a': [1, 2, 3], 'b': 4, 'c': {1: 5, 2: 6}}
        print(map_apply(x))

        ===>
        {'a': [1, 4, 9], 'b': 16, 'c': {1: 25, 2: 36}}

        print(map_apply(x, bias=2))

        ===>
        {'a': [3, 6, 11], 'b': 18, 'c': {1: 27, 2: 38}}
    ```
    """

    @wraps(f)
    def wrapper(x: Any, *args, **kwargs) -> Any:
        if isinstance(x, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in x.items()}

        elif isinstance(x, list):
            return [wrapper(v, *args, **kwargs) for v in x]

        elif isinstance(x, tuple):
            return tuple(wrapper(v, *args, **kwargs) for v in x)

        elif isinstance(x, set):
            return {wrapper(v, *args, **kwargs) for v in x}

        else:  # Base case, single item
            return f(x, *args, **kwargs)

    return wrapper


@opt_args_deco
def retry_new_on_error(
    __getitem__: Callable,
    exc: Union[BaseException, Sequence[BaseException]] = Exception,
    silent: bool = False,
    max: Optional[int] = None,
    use_blacklist: bool = False,
) -> Callable:
    """Decorator to wrap a BaseDataset __getitem__ function, and retry a different index if there is an error.

    The idea is to provide a way of ignoring missing/corrupt data without having to blacklist files,
    change number of items and do "hacky" workarounds.
    Obviously, the less data we have, the less sense this decorator makes, since we'll start duplicating more
    and more items (although if we're augmenting our data, it shouldn't be too tragic).
    Obviously as well, for debugging/evaluation it probably makes more sense to disable this decorator.

    NOTE: This decorator assumes we follow the BaseDataset format
        - We return three dicts (x, y, meta)
        - Errors are logged in meta['errors']
        - A 'log_timings' flag indicates the presence of a 'MultiLevelTimer' in self.timer

    :param __getitem__: (Callable) Dataset `__getitem__` method to decorate.
    :param exc: (tuple|Exception) Expected exceptions to catch and retry on.
    :param silent: (bool) If `False`, log error info to `meta`.
    :param max: (None|int) Maximum number of retries for a single item.
    :param use_blacklist: (bool) If `True`, keep a list of items to avoid.
    :return: (tuple[dict]) x, y, meta returned by `__getitem__`.
    """
    n = 0
    blacklist = set()

    # Multiple exceptions must be provided as tuple
    exc = exc or tuple()
    if isinstance(exc, list):
        exc = tuple(exc)

    @wraps(__getitem__)
    def wrapper(cls, item):
        nonlocal n

        try:
            x, y, m = __getitem__(cls, item)
            if not silent and "errors" not in m:
                m["errors"] = ""
        except exc as e:
            n += 1
            if max and n >= max:
                raise RuntimeError("Exceeded max retries when loading dataset item...")

            if use_blacklist:
                blacklist.add(item)
            if cls.log_time:
                cls.timer.reset()

            new = item
            while new == item or new in blacklist:  # Force new item
                new = random.randrange(len(cls))

            x, y, m = wrapper(cls, new)
            if not silent:
                m["errors"] += f'{" - " if m["errors"] else ""}{(item, e)}'

        n = 0  # Reset!
        return x, y, m

    return wrapper


@map_container
def to_torch(
    x: Any, /, permute: bool = True, device: Optional[torch.device] = None
) -> Any:
    """Convert given input to torch.Tensors

    :param x: (Any) Arbitrary structure to convert to tensors (see `map_apply`).
    :param permute: (bool) If `True`, permute to PyTorch convention (b, h, w, c) -> (b, c, h, w).
    :param device: (torch.device) Device to send tensors to.
    :return: (Any) Input structure, converted to tensors.
    """
    # Classes that should be ignored
    if isinstance(x, (str, Timer, MultiLevelTimer)):
        return x

    # NOTE: `as_tensor` allows for free numpy conversions
    x = torch.as_tensor(x, device=device)

    if permute and x.ndim > 2:
        dim = [-1, -3, -2]  # Transpose last 3 dims as (2, 0, 1)
        dim = list(range(x.ndim - 3)) + dim  # Keep higher dimensions the same
        x = x.permute(dim)

    return x


@map_container
def to_numpy(x: Any, /, permute: bool = True) -> Any:
    """Convert given input to numpy.ndarrays.

    :param x: (Any) Arbitrary structure to convert to ndarrays (see map_apply).
    :param permute: (bool) If `True`, permute from PyTorch convention (b, c, h, w) -> (b, h, w, c).
    :return: (Any) Input structure, converted to ndarrays.
    """
    # Classes that should be ignored
    if isinstance(x, (np.ndarray, str, Timer, MultiLevelTimer)):
        return x

    if permute and x.ndim > 2:
        dim = [-2, -1, -3]  # Transpose last 3 dims as [1, 2, 0]
        dim = list(range(x.ndim - 3)) + dim  # Keep higher dimensions the same
        x = x.permute(dim)

    return x.detach().cpu().numpy()


@opt_args_deco
def allow_np(fn: Optional[Callable], permute: bool = False) -> Callable:
    """Decorator to allow for numpy.ndarray inputs in a torch function.

    Main idea is to implement the function using torch ops and apply this decorator to also make it numpy friendly.
    Since numpy.ndarray and torch.Tensor share memory (when on CPU), there shouldn't be any overhead.

    The decorated function can have an arbitrary signature. We enforce that there should only be either np.ndarray
    or torch.Tensor inputs. All other args (int, float, str...) are left unchanged.
    """
    ann = fn.__annotations__
    for k, type in ann.items():
        if type == torch.Tensor:
            ann[k] = Union[NDArray, type]

    @wraps(fn)
    def wrapper(*args, **kwargs):
        all_args = args + tuple(kwargs.values())
        any_np = any(isinstance(arg, np.ndarray) for arg in all_args)
        any_torch = any(isinstance(arg, torch.Tensor) for arg in all_args)
        if any_torch and any_np:
            raise ValueError("Must pass only np.ndarray or torch.Tensor!")

        if any_np:
            args, kwargs = to_torch((args, kwargs), permute=permute)
        out = fn(*args, **kwargs)
        if any_np:
            out = to_numpy(out, permute=permute)

        return out

    return wrapper
