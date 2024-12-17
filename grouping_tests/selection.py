import numpy as np
import base, utility

def _validate_any_selector(rng, selector, ignore=False):
    """
    Function for validating input selector as a slice, boolean array, or array
    of indices aligned with the input range's actual or generic index values.
    """
    # Identify the selector type
    if isinstance(selector, slice):
        pass
    elif isinstance(selector, (list, tuple, np.ndarray)):
        selector = np.asarray(selector)
        if selector.dtype == bool:
            selector = _validate_boolean_selector(rng, selector)
        else:
            selector = _validate_index_selector(rng, selector, ignore)
    else:
        raise ValueError(
            "Input selector must be a slice object, boolean array, or an "
            "array of event indices.")
    return selector

def _validate_slice_selector(rng, selector):
    """
    Function for validating input selector as a slice object.
    """
    # Validate input
    if not isinstance(selector, slice):
        raise ValueError(
            "Input selector must be a slice object.")
    return selector

def _validate_boolean_selector(rng, selector):
    """
    Function for validating input selector as a boolean array.
    """
    # Validate input
    try:
        selector = np.asarray(selector)
    except:
        raise ValueError(
            "Input selector must be an array-like object.")
    if not selector.ndim == 1:
        raise ValueError(
            "Input selector must be a 1D array-like object.")
    if not len(selector) == rng.num_events:
        raise ValueError(
            "Input selector must be the same length as the number of events. "
            f"Expected {rng.num_events}, received {len(selector)}.")
    if not selector.dtype == bool:
        raise ValueError(
            "Input selector must be a boolean array.")
    return selector

def _validate_index_selector(rng, selector, ignore=False):
    """
    Function for validating input selector as an array of indices.
    """
    # Validate input
    try:
        selector = np.asarray(selector)
    except:
        raise ValueError(
            "Input selector must be an array-like object.")
    if not selector.ndim == 1:
        raise ValueError(
            "Input selector must be a 1D array-like object.")
    if ignore and not np.issubdtype(selector.dtype, np.integer):
        raise ValueError(
            "When ignoring the set index, input selector must be an array of "
            "integers.")
    
    # Apply to index values
    if not ignore:
        # Ensure that all values are present in the index
        selector_test = np.in1d(selector, rng._index)
        if not np.all(selector_test):
            missing_values = selector[~selector_test]
            raise ValueError(
                f"Index values not found: {missing_values}")
        # Sort the event index values
        sorter = np.argsort(rng._index)
        # Apply the selector to the sorted index values
        selector = np.searchsorted(rng._index[sorter], selector)
    else:
        # Ensure that all values are within the range of the number of events
        selector_test = (selector >= 0) & (selector < rng.num_events)
        if not np.all(selector_test):
            missing_values = selector[~selector_test]
            raise ValueError(
                f"Index values out of range: {missing_values}")
    return selector

def _apply_selector(rng, selector, inplace=False):
    """
    Apply a selector to the input events.
    """
    # Apply selection
    rc = rng if inplace else rng.copy()
    try:
        rc._index = rng._index[selector]
        rc._groups = rng._groups[selector] if rng._groups is not None else None
        rc._locs = rng._locs[selector] if rng._locs is not None else None
        rc._begs = rng._begs[selector] if rng._begs is not None else None
        rc._ends = rng._ends[selector] if rng._ends is not None else None
    except:
        raise ValueError(
            f"Invalid selection: {selector}")
    return None if inplace else rc

def select(rng, selector, ignore=False, inplace=False):
    """
    Select events by index or slice. Use ignore=True to use a generic 
    0-based index, ignoring the current index values.
    """
    selector = _validate_any_selector(rng, selector, ignore)
    return _apply_selector(rng, selector, inplace)

def select_slice(rng, slice_, inplace=False):
    """
    Select events by slice.
    """
    selector = _validate_slice_selector(rng, slice_)
    return _apply_selector(rng, selector, inplace)

def select_mask(rng, mask, inplace=False):
    """
    Select events by boolean mask.
    """
    selector = _validate_boolean_selector(rng, mask)
    return _apply_selector(rng, selector, inplace)

def select_index(rng, index, ignore=False, inplace=False):
    """
    Select events by index values.
    """
    selector = _validate_index_selector(rng, index, ignore)
    return _apply_selector(rng, selector, inplace)