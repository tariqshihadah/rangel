import numpy as np
import base

def overlay(left, right, normalize=True, norm_by='right'):
    """
    Compute the overlay of two collections of events.

    Parameters
    ----------
    left, right : Rangel
        Input Rangel instances to overlay.
    normalize : bool, default True
        Whether overlapping lengths should be normalized to give a 
        proportional result with a float value between 0 and 1.
    norm_by : str, default 'right'
        How overlapping lengths should be normalized. Only applied if
        `normalize` is True.
        - 'right' : Normalize by the length of the right events.
        - 'left' : Normalize by the length of the left events.
    """
    _norm_by_options = {'right', 'left'}
    
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input objects must have the same grouping status.")

    # Compute overlap lengths
    lefts = left.ends.reshape(-1, 1) - right.begs.reshape(1, -1)
    rights = right.ends.reshape(1, -1) - left.begs.reshape(-1, 1)

    # Compare against event lengths
    overlap = np.minimum(lefts, rights)
    lengths = np.minimum(
        left.lengths.reshape(-1, 1),
        right.lengths.reshape(1, -1)
    )
    np.minimum(overlap, lengths, out=overlap)
    np.clip(overlap, 0, None, out=overlap)

    # Normalize if necessary
    if normalize:
        # Get denominator
        if norm_by == 'right':
            denom = right.lengths.reshape(1, -1)
        elif norm_by == 'left':
            denom = left.lengths.reshape(-1, 1)
        else:
            raise ValueError(
                f"Invalid 'norm_by' parameter value provided ({norm_by}). Must be one "
                f"of {_norm_by_options}.")
        # Normalize
        denom = np.where(denom==0, np.inf, denom)
        np.divide(overlap, denom, out=overlap)

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = np.equal(left.groups.reshape(-1, 1), right.groups.reshape(1, -1))
        np.multiply(overlap, mask, out=overlap)
    
    return overlap

def intersection_point_point(left, right):
    """
    Identify intersections between two collections of point events.
    """
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input objects must have the same grouping status.")

    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_locs = right.locs.reshape(1, -1)
    
    # Test for intersection of locations
    res = np.equal(left_locs, right_locs)

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = np.equal(left.groups.reshape(-1, 1), right.groups.reshape(1, -1))
        np.logical_and(res, mask, out=res)
    
    return res

def intersection_point_linear(left, right, enforce_edges=True):
    """
    Identify intersections between a collection of point events and a collection 
    of linear events.
    """
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input objects must have the same grouping status.")

    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_begs = right.begs.reshape(1, -1)
    right_ends = right.ends.reshape(1, -1)

    # Initialize result array
    res = np.zeros((left_locs.shape[0], right_begs.shape[1]), dtype=bool)

    # Test for intersection of locations
    right_closed_base = right.closed_base
    # - Test 1
    if right_closed_base in ['left', 'both']:
        np.greater_equal(left_locs, right_begs, out=res)
    else:
        np.greater(left_locs, right_begs, out=res)
    # - Test 2
    if right_closed_base in ['right', 'both']:
        np.less_equal(left_locs, right_ends, out=res, where=res)
    else:
        np.less(left_locs, right_ends, out=res, where=res)

    # Test for modified edges
    if right.closed_mod and enforce_edges:
        # Get mask of modified edges to overwrite unmodified edges
        mask = right.modified_edges.reshape(1, -1)
        if right_closed_base == 'left':
            np.equal(left_locs, right_ends, out=res, where=mask & ~res)
        elif right_closed_base == 'right':
            np.equal(left_locs, right_begs, out=res, where=mask & ~res)
    
    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        np.equal(left.groups.reshape(-1, 1), right.groups.reshape(1, -1), out=mask)
        np.logical_and(res, mask, out=res)
    
    return res

def intersection_linear_linear(left, right, enforce_edges=True):
    """
    Identify intersections between two collections of linear events.
    """
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input objects must have the same grouping status.")

    # Reshape arrays for broadcasting
    left_begs = left.begs.reshape(-1, 1)
    left_ends = left.ends.reshape(-1, 1)
    right_begs = right.begs.reshape(1, -1)
    right_ends = right.ends.reshape(1, -1)

    # Initialize result array
    res = np.zeros((left_begs.shape[0], right_begs.shape[1]), dtype=bool)
    step = np.zeros((left_begs.shape[0], right_begs.shape[1]), dtype=bool)

    # Initialize result array with linear intersections
    np.greater(left_ends, right_begs, out=res)
    np.less(left_begs, right_ends, out=step)
    res &= step

    # Test edges if necessary
    if enforce_edges:
        # Identify if edge cases need testing
        test_edges = not (
            ((left.closed == 'neither') or (right.closed == 'neither')) or \
            ((left.closed == 'left') and (right.closed == 'left')) or \
            ((left.closed == 'right') and (right.closed == 'right'))
        )
        if test_edges:
            # Identify which edge cases need testing
            test_begs_ends = (left.closed != 'right') and (right.closed != 'left')
            test_ends_begs = (left.closed != 'left') and (right.closed != 'right')

            # Identify modified edges if needed
            if left.closed_mod:
                left_mod = left.modified_edges.reshape(-1, 1)
            if right.closed_mod:
                right_mod = right.modified_edges.reshape(1, -1)

            # - Test 1: left_begs == right_ends
            mask = np.invert(res)
            if test_begs_ends:
                # Create mask for where edge cases are relevant
                if left.closed == 'right_mod':
                    np.logical_and(mask, left_mod, out=mask)
                if right.closed == 'left_mod':
                    np.logical_and(mask, right_mod, out=mask)
                # Apply test
                np.equal(left_begs, right_ends, out=step)
                np.logical_and(step, mask, out=step)
                np.logical_or(res, step, out=res)

            # - Test 2: left_ends == right_begs
            np.invert(res, out=mask)
            if test_ends_begs:
                # Create mask for where edge cases are relevant
                if left.closed == 'left_mod':
                    np.logical_and(mask, left_mod, out=mask)
                if right.closed == 'right_mod':
                    np.logical_and(mask, right_mod, out=mask)
                # Apply test
                np.equal(left_ends, right_begs, out=step)
                np.logical_and(step, mask, out=step)
                np.logical_or(res, step, out=res)

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        np.equal(left.groups.reshape(-1, 1), right.groups.reshape(1, -1), out=mask)
        np.logical_and(res, mask, out=res)
    
    return res
