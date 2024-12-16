import numpy as np
import base, utility

def dissolve(rng, return_index=False):
    """
    Merge consecutive ranges. For best results, input events should be sorted.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    if not rng.is_linear:
        raise ValueError("Input object must be a linear Rangel instance.")
    if rng.is_empty:
        return base.Rangel()
    
    # Identify edges of dissolvable events
    consecutive_strings = rng.consecutive_strings()
    string_number, string_start = \
        np.unique(consecutive_strings, return_index=True)
    
    # Initialize new event edges
    index = []
    groups = [rng.groups[0] if rng.groups is not None else None]
    begs = [rng.begs[0]]
    ends = []
    
    # Get min and max bounds of dissolved events
    if len(string_start) > 1:
        for i, j in zip(string_start[:-1], string_start[1:]):
            ends.append(rng.ends[j - 1])
            begs.append(rng.begs[j])
            index.append(rng.index[i:j])
            groups.append(rng.groups[j] if rng.groups is not None else None)
    
    # Add final end point
    ends.append(rng.ends[-1])
    index.append(rng.index[string_start[-1]:])
    
    # Prepare output class instance
    res = rng.from_similar(
        index=None,
        groups=groups if rng.groups is not None else None,
        begs=begs, 
        ends=ends, 
        closed=rng.closed
    )

    # Return results
    if return_index:
        return res, index
    return res

def extend(rng, extend_begs=0, extend_ends=0, inplace=False):
    """
    Extend the range of events by a specified amount in either or both directions.

    Parameters
    ----------
    rng : Rangel
        Input range of events.
    extend_begs : float or array-like, optional
        Amount to extend the beginning and end of each event range. If an array-like
        is provided, it must be the same length as the number of events in the 
        collection. Positive values extend ranges to the left, negative values to
        the right. Default is 0.
    extend_ends : float or array-like, optional
        Amount to extend the end of each event range. If an array-like is provided,
        it must be the same length as the number of events in the collection. Positive
        values extend ranges to the right, negative values to the left. Default is 0.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    extend_begs = utility._validate_scalar_or_array_input(rng, extend_begs, 'extend_begs')
    extend_ends = utility._validate_scalar_or_array_input(rng, extend_ends, 'extend_ends')

    # Select object to modify
    rng = rng if inplace else rng.copy()

    # Select methodology
    if rng.is_point:
        rng._begs = rng.locs - extend_begs
        rng._ends = rng.locs + extend_ends
    else:
        rng._begs = rng._begs - extend_begs
        rng._ends = rng._ends + extend_ends
    
    # Return results
    return None if inplace else rng

def shift(rng, shift, inplace=False):
    """
    Shift the range of events by a specified amount.

    Parameters
    ----------
    rng : Rangel
        Input range of events.
    shift : float or array-like
        Amount to shift all events. If an array-like is provided, it must
        be the same length as the number of events in the collection. Positive
        values shift events to the right, negative values to the left.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    shift = utility._validate_scalar_or_array_input(rng, shift, 'shift')

    # Select object to modify
    rng = rng if inplace else rng.copy()

    # Select methodology
    if rng.is_located:
        rng._locs = rng._locs + shift
    if rng.is_linear:
        rng._begs = rng._begs + shift
        rng._ends = rng._ends + shift
    
    # Return results
    return None if inplace else rng

def round(rng, decimals=None, factor=None, inplace=False):
    """
    Round the bounds and locations of events to a specified number of decimal 
    places or using a specified rounding factor.

    Parameters
    ----------
    rng : Rangel
        Input range of events.
    decimals : int, optional
        Number of decimal places to round to. If an array-like is provided, it must
        be the same length as the number of events in the collection. Default 
        is None.
    factor : float, optional
        Rounding factor. If provided, the bounds and locations of events will be
        rounded to the nearest multiple of this factor. Default is None.
    inplace : bool, optional
        If True, modify the input object in place. Default is False.
    """
    # Validate input
    if not isinstance(rng, base.Rangel):
        raise TypeError("Input object must be a Rangel class instance.")
    if decimals is not None:
        if not isinstance(decimals, int):
            raise TypeError("'decimals' must be an integer.")
        _rounder = lambda x: np.round(x, decimals=decimals)
    elif factor is not None:
        factor = utility._validate_scalar_or_array_input(rng, factor, 'factor', nonzero=True)
        _rounder = lambda x: np.round(x / factor, decimals=0) * factor
    else:
        raise ValueError("Either 'decimals' or 'factor' must be provided.")
    
    # Select object to modify
    rng = rng if inplace else rng.copy()

    # Select methodology
    
    if rng.is_located:
        rng._locs = _rounder(rng._locs)
    if rng.is_linear:
        rng._begs = _rounder(rng._begs)
        rng._ends = _rounder(rng._ends)

    # Return results
    return None if inplace else rng