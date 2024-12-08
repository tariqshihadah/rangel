import numpy as np
import base

def _method_require(**requirements):
    """
    Callable decorator to require that a class meets certain attribute or 
    property requirements.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for key, value in requirements.items():
                if getattr(self, key) != value:
                    raise ValueError(
                        f"The {func.__name__} method is only available "
                        f"for {self.__class__.__name__} instances with {key}={value}."
                    )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def _prepare_data_array(data, name, ndim=1, dtype=None, copy=None):
    """
    Function for validating input data as a 1D scalar np.array using the given 
    name and optional dtype and copy arguments.
    """
    # Initialize data as a numpy array
    try:
        try:
            data = np.asarray(data, dtype=dtype, copy=copy)
        except TypeError: # numpy 1.X compatibility
            data = np.array(data, dtype=dtype, copy=False)
        assert data.ndim == ndim
    except AssertionError:
        if dtype is None:
            raise ValueError(
                f"Invalid input data for `{name}`. Must be a 1D array-like object."
            )
        else:
            raise ValueError(
                f"Invalid input data for `{name}`. Must be a 1D array-like object with dtype={dtype}. Provided array has shape={data.shape} and dtype={data.dtype}."
            )
    return data

