import numpy as np
import base

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