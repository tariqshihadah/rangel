def intersection_point_point(left, right):
    """
    Identify intersections between two collections of point events.
    """
    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_locs = right.locs.reshape(1, -1)
    
    # Test for intersection of locations
    res = left_locs == right_locs
    return res

def intersection_point_linear(left, right):
    """
    """
    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_begs = right.begs.reshape(1, -1)
    right_ends = right.ends.reshape(1, -1)

    # Initialize result array
    res = np.empty((left_locs.shape[0], right_begs.shape[1]), dtype=bool)

    # Test for intersection of locations
    right_closed_base = right.closed_base
    if right_closed_base in ['left', 'both']:
        np.greater_equal(left_locs, right_begs, out=res)
    else:
        np.greater(left_locs, right_begs, out=res)
    if right_closed_base in ['right', 'both']:
        np.less_equal(left_locs, right_ends, out=res)
    else:
        np.less(left_locs, right_ends, out=res)

    # Test for modified edges
    if right.closed_mod:
        # Get mask of modified edges to overwrite unmodified edges
        mask = right.modified_edges
        if right_closed_base == 'left':
            np.equal(left_locs, right_ends, out=res, where=mask)
        elif right_closed_base == 'right':
            np.equal(left_locs, right_begs, out=res, where=mask)
    
    return res

def intersection_linear_linear(left, right):
    """
    """
    pass