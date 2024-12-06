import numpy as np

def intersection_point_point(left, right):
    """
    Identify intersections between two collections of point events.
    """
    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_locs = right.locs.reshape(1, -1)
    
    # Test for intersection of locations
    res = np.equal(left_locs, right_locs)
    return res

def intersection_point_linear(left, right, enforce_edges=True):
    """
    """
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
    
    return res

def intersection_linear_linear(left, right, enforce_edges=True):
    """
    """
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
                res |= step & mask

            # - Test 2: left_ends == right_begs
            np.invert(res, out=mask)
            if test_ends_begs:
                # Create mask for where edge cases are relevant
                if left.closed == 'left_mod':
                    np.logical_and(mask, left_mod, out=mask)
                if right.closed == 'right_mod':
                    np.logical_and(mask, right_mod, out=mask)
                # Apply test
                np.equal(left_ends, right_begs, out=res)
                res |= step & mask

    return res
