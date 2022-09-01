import numpy as np
from rangel import RangeCollection
from scipy import stats

def rasterize(
    data,
    values=None,
    size=1,
    blur_size=0,
    blur_style='linear',
    bounds=None,
    fill='cut', 
    closed='left_mod',
    rc=None,
    **kwargs
):
    """
    Digitize and buffer events over a defined range, extending them across 
    uniform steps within the range and scaling their values relative to their 
    digital distance from their intersecting step location.
    
    Parameters
    ----------
    data : 1d or 2d array-like
        The ranges of the locations of target events being analyzed. All ranges 
        should fall within the defined bounds (if provided) to be considered in 
        the analysis.
    values : numeric or 1d array-like, optional
        The value(s) associated with each event being analyzed. If not 
        provided, all values will default to be 1.
    size : positive numeric, default 1
        The length of each pixel used to perform the analysis.
    blur_size : int, default 0
        The number of pixels to blur events across based on the blur style.
    blur_style : str or callable, default 'linear'
        The scaling function to be called at each blurring step to scale 
        original values. If a callable is provided, it must accept a single 
        integer input for the zero-indexed pixel number, returning a single 
        float scaling value. Predefined blurring functions can be called using 
        the following labels:
        
        Options
        -------
        linear : linearly scale down values from the original value to zero 
            at the first index outside the blurred pixel range
        norm_static : Tk
        norm_scale : Tk
        none : do not scale down original values
        
    bounds : two-value numeric tuple, optional
        The values at which to begin and end the pixelation analysis. If not 
        provided, will default to the min and max location values in the events 
        data, respectively. If a predefined range collection is provided, that 
        will supersede these input parameters.        
    rc : RangeCollection, optional
        A predefined valid consecutive range collection instance which defines 
        the analysis domain and which will be used to intersect with input 
        events.
    **kwargs
        Keyword arguments to be passed to the constructor for the generated 
        range collection if rc is not provided. The collection is generated 
        using the RangeCollection.from_steps method.
            
    Created:  2022-06-16
    """
    ##################
    # VALIDATE INPUT #
    ##################
    
    # Validate input range collection
    if not isinstance(data, RangeCollection):
        raise TypeError("Input data must be range collection type.")
    
    # Validate blur style
    if blur_style in [None, 'none'] or blur_size==0:
        blur_function = lambda n: 1
    elif blur_style in ['linear','lin']:
        blur_function = lambda n: (blur_size - n + 1) / (blur_size + 1)
    elif blur_style in ['norm_static']:
        blur_function = lambda n: stats.norm.pdf(n)
    elif blur_style in ['norm_scale']:
        blur_function = lambda n: stats.norm.pdf(n * 3, scale=blur_size)
    elif callable(blur_style):
        blur_function = blur_style
    else:
        raise ValueError(
            "Input blur_style must be callable or label of valid predefined "
            "scaling function.")
    
    # Validate values data
    if not values is None:
        try:
            # Extend single scalar value
            values = np.full(data.num_ranges, float(values))
        except:
            try:
                # Coerce array-like to array
                values = np.asarray(values, dtype=float).flatten()
            except:
                raise TypeError(
                    "Could not convert input values to np.ndarray of dtype "
                    "float.")
            # Check for appropriate size
            if not len(values) == data.num_ranges:
                raise ValueError(
                    "Number of event values must be equal to the number of "
                    f"events provided. Provided: {data.num_ranges} events, "
                    f"{len(values)} values.")
    
    # Validate or create analysis range collection
    if rc is None:
        if bounds is None:
            beg, end = np.min(data.begs), np.max(data.ends)
        else:
            try:
                beg, end = bounds
            except:
                raise ValueError(
                    "If used, bounds must be provided as two-value tuple of "
                    "scalars defining the bounds of the analysis.")
        # Create a new analysis range collection
        rc = RangeCollection.from_steps(
            beg, end, length=size, steps=1, **kwargs)
    elif not isinstance(rc, RangeCollection):
        raise ValueError("Input range collection is not valid.")
            
    ####################
    # PERFORM ANALYSIS #
    ####################
    
    # Intersect events data with analysis range collection to determine initial
    # pixel scores
    intx = rc.intersecting(
        beg=data.begs, end=data.ends, squeeze=False) * 1 # coerce integer
    
    # Blur events data by stepping through the blur range and applying the blur 
    # function to scale the initial pixel scores and add them
    scale = blur_function(0)
    data = intx * scale
    for step in range(1, blur_size + 1):
        # Create and blur buffered data
        scale = blur_function(step)
        buff = np.pad(intx, ((step, step), (0, 0)), mode='constant') * scale
        # Apply buffered data
        forw = buff[:-step * 2, :]
        back = buff[step * 2:, :]
        data += forw + back
        
    # Normalize data scores by target range lengths
    term = rc.lengths.reshape(-1,1)
    data = np.multiply(data, term)
    
    # Normalize data scores to a single unit per record
    denom = data.sum(axis=0)
    data = np.divide(data, denom, out=np.zeros_like(data), where=denom!=0)
    
    # Apply values if used
    if not values is None:
        data = np.multiply(data, values)
    
    # Return results
    return rc, data
