"""
===============================================================================

Module featuring RangeCollection object class for the management of range data 
and optimized performance of various simple and complex range operations, 
including range and point overlays, equitable and score-based separation of 
overlapping ranges, generation of random range data, and range data comparisons 
among others.

Collections can be initialized using a variety of constructor class methods 
including from_array, from_breaks, from_steps, and others. Data is managed with 
numpy arrays, requiring minimal other dependencies.

Classes
-------
RangeCollection

Dependencies
------------
numpy, matplotlib, scipy, copy, warnings

Development
-----------
Developed by:
Tariq Shihadah, tariq.shihadah@gmail.com

Created:
10/22/2019

Modified:
1/11/2021

===============================================================================
"""


################
# DEPENDENCIES #
################


import numpy as np
import matplotlib.pyplot as plt
import copy, warnings
from scipy import stats
from rangel.util.logic import _ArrayLogicManager


####################
# RANGE COLLECTION #
####################


class RangeCollection(object):
    """
    An object class to manage, modify, and analyze numerical range information 
    with an efficient, flexible, numpy-based approach. Range collections allow
    for indexing based on continuous ranges, analyzing and separating 
    overlapping ranges, sorting based on range bounds or reference points, and
    more.

    RangeCollections can be initialized using various constructor class methods 
    including from_array, from_breaks, from_steps, and others.
    
    Parameters
    ----------
    begs : array-like
        A 1D array or list of numerical values representing the begin points 
        of each range within the collection. If begs and ends is provided, 
        breaks should not be provided.
    ends : array-like
        A 1D array or list of numerical values representing the end points of
        each range within the collection. If begs and ends is provided, 
        breaks should not be provided.
    centers : array-like or {'true_centers', 'begs', 'ends'}, optional
        A 1D array or list of numerical values representing the effective 
        center points of each range within the collection. This does not 
        necessarily need to represent the true center of each range. Can 
        provide a 1D array-like with a number of elements equal to the number 
        of ranges in the collection, or a string indicating which collection 
        property to set as the static center values. If None is provided, the 
        collection will default to dynamically calculated true center values.
    closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
            'neither'}, default 'right'
        Whether collection intervals are closed on the left-side, right-side, 
        both or neither.

        Options
        -------
        left : ranges are always closed on the left and never closed on the 
            right.
        left_mod : ranges are always closed on the left and only closed on the 
            right when the next range is not consecutive or overlapping.
        right : ranges are always closed on the right and never closed on the 
            right.
        right_mod : ranges are always closed on the right and only closed on 
            the left when the previous range is not consecutive or overlapping.
        both : ranges are always closed on both sides
        neither : ranges are never closed on either side
            
    sort : boolean, default False
        Whether to automatically sort the ranges by begin points or to leave
        them in the order they are input.
    snap_centers : boolean, default False
        Whether to snap center values which fall outside the bounds of an 
        associated range to the nearest bound. Only applicable when an array 
        of center values is provided.
    copy : boolean, default True
        Whether to copy all input information to the range collection.
    force_monotonic : boolean, default True
        Whether to automatically force all ranges to be increasing, with each
        end point value being greater than or equal to its associated begin 
        point value.
    keys : Not implemented.
    """

    # Define class variables
    _ops_closed = {'left','left_mod','right','right_mod','both','neither'}
    _ops_closed_base = {'left','right','both','neither'}
    _ops_centers = {'true_centers','begs','ends'}
    display_max = 10
    
    def __init__(
        self, begs=None, ends=None, centers=None, closed='right', sort=False, 
        snap_centers=False, copy=None, force_monotonic=True, keys=None):

        # Process ranges
        if begs is None:
            begs = []
        if ends is None:
            ends = begs
        # Address copy parameter changes between numpy 1.x and 2.x
        try:
            begs = np.array(begs, dtype=float, copy=copy).flatten()
            ends = np.array(ends, dtype=float, copy=copy).flatten()
        except ValueError:
            begs = np.array(begs, dtype=float, copy=True).flatten()
            ends = np.array(ends, dtype=float, copy=True).flatten()
        self._begs = begs.copy()
        self._ends = ends.copy()
        # Set centers
        self.set_centers(centers, inplace=True, snap=snap_centers, copy=copy)

        # Set closed parameters
        self.set_closed(closed, inplace=True)
        
        # Reset keys
        self.reset_keys(keys=keys, inplace=True) 
        
        # Validate input
        if force_monotonic:
            self.set_monotonic(increasing=True, inplace=True)
        else:
            self._monotonic = False
        if sort:
            self.sortranges(by='begs', ascending=True, inplace=True)
        
    def __getitem__(self, index):
        rc = self.copy()
        rc._select_self(index)
        return rc
    
    def __len__(self):
        return self.num_ranges
    
    def __iter__(self):
        return iter(self.pairs)
    
    def __str__(self):
        text = f"""\
RangeCollection({self.num_ranges:,.0f} numeric ranges, closed={self.closed}, \
centers={self.center_type})"""
        return text
    
    def __repr__(self):
        # If no ranges present, return self as a string
        if self.num_ranges == 0:
            return str(self)
        # Determine number of records to show
        if self.num_ranges > self.display_max:
            # Define head/skip/tail selections
            display_head = (self.display_max // 2) + (self.display_max % 2)
            display_tail = (self.display_max // 2)
            display_skip = self.num_ranges - self.display_max
            # Define bool mask
            display_select = np.array(
                [True]  * display_head + 
                [False] * display_skip + 
                [True]  * display_tail)
        else:
            # Default head/skip/tail selections
            display_head = self.num_ranges
            display_tail = display_skip = 0
            display_select = np.array([True] * self.num_ranges)
        # Determine numbers of left and right digits to display
        ld = len(str(int(self.arr[display_select].max())))
        rd = 3
        # Create formatter
        records = []
        closed = self.closed
        # Iterate over selected features and create strings
        feature_gen = zip(
            self.begs[display_select],
            self.ends[display_select],
            self._mod_locs[display_select])
        for beg, end, mod in feature_gen:
            # Determine left and right brackets
            lb = '[' if (closed in ['left','left_mod','both']) or mod else '('
            rb = ']' if (closed in ['right','right_mod','both']) or mod else ')'
            # Format string
            record = f'{lb}{beg: >{ld+rd+1}.{rd}f}, {end: >{ld+rd+1}.{rd}f}{rb}'
            records.append(record)
        # Create skipped record label if required
        if display_skip > 0:
            # Label skipped records
            spacer_label = '{:,.0f} records'.format(display_skip)
            # Format label
            spaces = max(ld*2 + rd*2 + 6 - len(spacer_label), 6)
            spacer = '.' * (spaces // 2) + spacer_label + \
                '.' * (spaces // 2 + spaces % 2)
            records = \
                records[:display_head] + [spacer] + records[-display_tail:]
        # Format full text string and return
        return '\n'.join(records) + '\n' + str(self)

    @property
    def num_ranges(self):
        return self.begs.size
    
    @property
    def total_length(self):
        return self.lengths.sum()
    
    @property
    def begs(self):
        return self._begs
    
    @property
    def ends(self):
        return self._ends
    
    @property
    def keys(self):
        return self._keys
    
    @property
    def index(self):
        return np.asarray(range(self.num_ranges))
    
    @property
    def rng(self):
        return np.stack((self.begs, self.ends), axis=0).reshape((2,-1))
    
    @property
    def arr(self):
        return np.stack((self.begs, self.ends, self.centers), axis=1)

    @property
    def values(self):
        return np.stack((self.begs, self.ends, self.centers), axis=0)
    
    @property
    def pairs(self):
        return self.rng.transpose()
    
    @property
    def lengths(self):
        return self.ends - self.begs
    
    @property
    def left_lengths(self):
        return self.centers - self.begs
    
    @property
    def right_lengths(self):
        return self.ends - self.centers
    
    @property
    def ratios(self):
        return self.right_lengths / self.left_lengths
    
    @property
    def centers(self):
        """
        The center values of all ranges in the collection. If not otherwise 
        defined at initialization or through set_centers, will be equivalent to 
        true_centers.
        """
        if not self.center_type=='data':
            return getattr(self, self._centers)
        else:
            return self._centers
    
    @centers.setter
    def centers(self, centers):
        self.set_centers(centers, inplace=True, copy=False)

    @property
    def center_type(self):
        if isinstance(self._centers, str):
            return self._centers
        else:
            return 'data'

    @property
    def true_centers(self):
        return (self.begs + self.ends)/2
    
    @property
    def groups(self):
        """
        Group all consecutive ranges by unique numbers.
        """
        groups = np.insert(np.cumsum(~self.are_consecutive(False)),0,0)
        return groups

    @property
    def closed(self):
        """
        Collection parameter for whether intervals are closed on the left-side, 
        right-side, both or neither.
        """
        return self._closed

    @closed.setter
    def closed(self, closed):
        self.set_closed(closed, inplace=True)

    @property
    def closed_base(self):
        return self._closed_base
    
    @property
    def monotonic(self):
        return self._monotonic
    
    @classmethod
    def _breaks_to_ranges(cls, breaks):
        """
        Take an array of break points or a list of arrays of break points and 
        convert them to begin and end points to define unique ranges.
        """
        # Check for valid input
        if not isinstance(breaks, (list, np.ndarray)):
            raise TypeError(
                "Input breaks must be an array or a list of arrays of valid "
                "breaks data.")
        elif len(breaks) == 0:
            return np.array([]), np.array([])

        # Array of numerical break points
        if np.issubdtype(type(breaks[0]), np.number):
            breaks = np.asarray(breaks, dtype=float)
            begs = breaks[:-1]
            ends = breaks[1:]
            
        # List of arrays of numerical break points
        elif isinstance(breaks[0], (list, np.ndarray)):
            # Initialize result arrays
            begs = np.asarray([], dtype=float)
            ends = np.asarray([], dtype=float)
            for breaks_i in breaks:
                # Array of numerical break points
                if len(breaks_i) == 0:
                    continue
                if np.issubdtype(type(breaks_i[0]), np.number):
                    breaks_i = np.asarray(breaks_i, dtype=float)
                    begs = np.concatenate((begs, breaks_i[:-1]), axis=None)
                    ends = np.concatenate((ends, breaks_i[1:]), axis=None)
                else:
                    raise TypeError(
                        "Multiple sets of break points must be provided as a "
                        "list of arrays or array-like.")
        else:
            raise TypeError(
                "Input breaks must be an array or a list of arrays of valid "
                "breaks data.")
        
        # Return results as a tuple
        return begs, ends        
    
    def _select_self(self, index):
        """
        Use a static index to set new object attributes. Equivalent to 
        __getitem__ operating inplace.
        """
        # Array indexer
        if not isinstance(index, slice):
            # Flatten to avoid dimension loss
            index = np.asarray(index).flatten()
        # Select from data arrays
        try:
            self._begs = self._begs[index]
            self._ends = self._ends[index]
            self._keys = self._keys[index]
        except:
            raise IndexError(
                f"Unable to index range data with provided indexer: {index}")
        # Determine center
        if self.center_type=='data':
            self._centers = self._centers[index]
        
    def _validate_centers(self, centers=None, snap=False, copy=False):
        """
        Ensure that each range center falls within the bounds of
        its respective range.
        """
        # Validate no input
        if centers is None:
            return 'true_centers'

        # Validate string input
        elif isinstance(centers, str):
            # Check for valid value
            if not centers in self._ops_centers:
                raise ValueError(
                    f"Centers as string must be one of {self._ops_centers}.")
            return centers

        # Validate array input
        # Enforce array class
        try:
            centers = np.array(centers, copy=copy)
        except:
            raise ValueError("Cannot convert centers data to array.")

        # Test for array length
        if len(centers) != self.num_ranges:
            raise ValueError(
                "Centers information must have an equal number of elements to "
                "the number of ranges in the collection.")
        
        # Test for numeric dtype
        if not np.issubdtype(centers.dtype, np.number):
            raise TypeError("Input center values must be numeric.")

        # Test for NaN
        if np.isnan(centers).sum() > 0:
            raise ValueError("Input center values cannot contain NaN.")

        # Test the centers against the bounds of the ranges
        if snap:
            # Snap values to nearest bound
            centers = np.max([centers, self.begs], axis=0)
            centers = np.min([centers, self.ends], axis=0)
        elif not np.all((centers >= self.begs) & (centers <= self.ends)):
            raise ValueError(
                "Range centers must fall within the begin and end points of "
                "the input ranges.")

        # Return validated centers array
        return centers
    
    def _validate_begend(self):
        """
        Ensure that each range begin and end point is a valid
        numerical value.
        """
        # Test for numeric dtype
        if not np.issubdtype(self.begs.dtype, np.number):
            raise TypeError("Input begin values must be numeric.")
        if not np.issubdtype(self.ends.dtype, np.number):
            raise TypeError("Input end values must be numeric.")
        
        # Test for NaN
        if np.isnan(self.begs).sum() > 0:
            raise ValueError("Input begin values cannot contain NaN.")
        if np.isnan(self.ends).sum() > 0:
            raise ValueError("Input end values cannot contain NaN.")
        return
    
    def _validate_monotonic(self):
        """
        Confirm that the range collection is monotonic.
        """
        if self._monotonic:
            return
        elif self.is_monotonic():
            self._monotonic = True
            return
        else:
            raise ValueError("Range collection is not monotonic.")
    
    @classmethod
    def from_centers(cls, centers, lengths=1.0, **kwargs):
        """
        Create a range collection from an array of centers and (optionally) 
        length values.
        
        Parameters
        ----------
        centers : array-like
            An array-like of numerical values representing the centers of 
            ranges being defined.
        lengths : scalar or array-like, default 1.0
            A single value or array-like of numerical values representing the 
            lengths of ranges being defined.
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        
        Returns
        -------
        rc : RangeCollection
            RangeCollection object instance created based on the provided 
            constructor parameters.
        """
        # Validate inputs
        try:
            centers = np.asarray(centers, dtype=float).flatten()
        except:
            raise ValueError(
                "Invalid centers input values. Must be provided as an array-"
                "like of scalar values.")
        try:
            lengths = np.asarray(lengths, dtype=float).flatten()
        except:
            raise ValueError(
                "Invalid lengths input values. Must be provided as a single "
                "scalar value or an array-like of scalar values.")
        if lengths.size == 1:
            lengths = np.tile(lengths, centers.size)
        elif lengths.size != centers.size:
            raise ValueError(
                f"If provided as an array, lengths (size={lengths.size:,.0f}) "
                f"must be equal in size to centers array "
                f"(size={centers.size:,.0f}).")

        # Define range begin and end points and create collection
        delta = lengths / 2
        begs = centers - delta
        ends = centers + delta
        rc = cls(begs=begs, ends=ends, centers=centers, **kwargs)
        return rc

    @classmethod
    def from_array(cls, arr, ignore_centers=False, **kwargs):
        """
        Create a range collection from an array defining the begin and end
        bounds of each range, as well as (optionally) the centers.
        
        Parameters
        ----------
        arr : array-like
            A 1-2D array-like of numerical values representing the begin and 
            end bounds of each range, and (optionally) the center values. 
            Arrays must be of shape (x), (x,1), (x,2), or (x,3)
        ignore_centers : bool, default False
            Whether to ignore centers data if provided within the data array.
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        
        Returns
        -------
        rc : RangeCollection
            RangeCollection object instance created based on the provided 
            constructor parameters.
        """
        # Enforce array
        try:
            arr = np.array(arr)
        except:
            raise ValueError("Input array must be coercible to np.ndarray.")
        # Check dimensions
        if arr.ndim == 1:
            begs = arr
            ends = None
            centers = None
        elif np.shape(arr)[1] == 1:
            begs = arr[:,0]
            ends = None
            centers = None
        elif np.shape(arr)[1] == 2:
            begs = arr[:,0]
            ends = arr[:,1]
            centers = None
        elif np.shape(arr)[1] == 3:
            begs = arr[:,0]
            ends = arr[:,1]
            centers = arr[:,2] if not ignore_centers else None
        else:
            raise ValueError(
                "Input array must have shape of (x), (x,1), (x,2), or (x,3)")
        rc = cls(begs=begs, ends=ends, centers=centers, **kwargs)
        return rc
            
    @classmethod
    def from_breaks(cls, breaks, **kwargs):
        """
        Create a range collection from a list of break point information.
        
        Parameters
        ----------
        breaks : array-like or list of array-like of numerical values
            Numerical information representing the range breakpoints to be used 
            to compute begin and end points for consecutive ranges automatically.
            To generate non-consecutive ranges, provide a list of array-like 
            objects which will each be parsed separately.
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        
        Returns
        -------
        rc : RangeCollection
            RangeCollection object instance created based on the provided 
            constructor parameters.
        """
        # Convert breaks to ranges
        begs, ends = cls._breaks_to_ranges(breaks)
        rc = cls(begs=begs, ends=ends, **kwargs)
        return rc

    @classmethod
    def from_bounds(cls, beg, end, num_ranges, **kwargs):
        """
        Create a collection of a specified number of consecutive ranges between 
        a begin and end point.

        Parameters
        ----------
        beg, end : scalar
            Numerical values defining the begin and end points of the 
            consecutive range collection.
        num_ranges : int
            The number of total ranges to be created in the collection between 
            the begin and end points.
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        """
        # Create array data for ranges
        breaks = np.linspace(beg, end, num=num_ranges, endpoint=True)
        # Create range collection
        rc = cls.from_breaks(breaks, **kwargs)
        return rc
    
    @classmethod
    def from_steps(cls, beg, end, length=1, steps=1, fill=None, **kwargs):
        """
        Create a collection of ranges between a begin and end point using a 
        fixed range length and a specified number of steps to define a grid to 
        fix ranges to.
        
        Parameters
        ----------
        beg, end : scalar
            Numerical values defining the begin and end points of the stepped 
            range collection.
        length : numerical, default 1.0
            A fixed length for all ranges being defined.
        steps : int, default 1
            A number of steps per range length. The resulting step length will 
            be equal to length / steps. For non-overlapped ranges, use a steps 
            value of 1.
        fill : {'none','cut','left','right','extend','balance'}, default 'cut'
            How to fill a gap at the end of the input range.

            Options
            -------
            none : no range will be generated to fill the gap at the end of the 
                input range.
            cut : a truncated range will be created to fill the gap with a 
                length less than the full range length.
            left : the final range will be anchored on the end value and will 
                extend the full range length to the left. 
            right : the final range will be anchored on the grid defined by the 
                step value, extending the full range length to the right, 
                beyond the defined end value.
            extend : the final range will be anchored on the grid defined by 
                the step value, extending beyond the step length to the right
                bound of the range.
            balance : if the final range is greater than or equal to half the 
                target range length, perform the cut method; if it is less, 
                perform the extend method.

            Schematics
            ----------
            bounds : [------------------------]
            none :   
                     [---------|              ]
                     [         |---------|    ]
            cut : 
                     [---------|              ]
                     [         |---------|    ]
                     [                   |----]
            left :   
                     [---------|              ]
                     [         |---------|    ]
                     [              |---------]
            right :  
                     [---------|              ]
                     [         |---------|    ]
                     [                   |----]----|
            extend :
                     [---------|              ]
                     [         |--------------]
        
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        
        Returns
        -------
        rc : RangeCollection
            RangeCollection object instance created based on the provided 
            constructor parameters.
        """
        # Validate inputs
        # - length
        try:
            length = float(length)
        except ValueError:
            raise TypeError("Length of ranges must be a numeric value.")
        # - steps
        try:
            steps = int(steps)
            step = length / steps
        except ValueError:
            raise TypeError("Number of steps must be an integer value.")
        # - fill
        fill_options = {'none','cut','left','right','extend','balance'}
        if fill is None:
            fill = 'cut'
        elif not fill in fill_options:
            raise TypeError(f"Input fill option must be one of {fill_options}")
            
        # Build default ranges
        vertices = np.arange(beg,end,step)
        begs = vertices[:-steps]
        ends = vertices[steps:]
        
        # Address minimal ranges
        num_ranges = len(begs)
        if num_ranges:
            last_beg = begs[-1]
            last_end = ends[-1]
        else:
            last_beg = beg - step
            last_end = beg - step + length
        
        # Address fill option
        if fill == 'balance':
            if end - last_end >= length / 2:
                fill = 'cut'
            else:
                fill = 'extend'
        if last_end + step > end:
            if fill == 'none':
                pass
            elif fill == 'cut':
                begs = np.append(begs,last_beg+step)
                ends = np.append(ends,end)
            elif fill == 'left':
                begs = np.append(begs,end-length)
                ends = np.append(ends,end)
            elif fill == 'right':
                begs = np.append(begs,last_beg+step)
                ends = np.append(ends,last_end+step)
            elif fill == 'extend':
                if num_ranges > 0:
                    ends[-1] = end
                elif end > beg:
                    begs = np.array([beg])
                    ends = np.array([end])
                else:
                    pass
        else:
            begs = np.append(begs,last_beg+step)
            ends = np.append(ends,end)
        
        # Generate RangeCollection
        rc = cls(begs=begs, ends=ends, **kwargs)
        return rc

    @classmethod
    def union(cls, objs, fill_gaps=False, return_index=False, null_index=None, 
        **kwargs):
        """
        Combine multiple RangeCollection instances into a single instance, 
        creating least common intervals among all collections. If requested, 
        a list of index arrays for each range can be provided to relate each 
        new range to its parent range(s).

        Parameters
        ----------
        objs : list-like of RangeCollection instances
            A selection of RangeCollection object instances to be combined 
            into a single instance based on the input parameters.
        fill_gaps : bool, default False
            Whether to fill gaps in merged collection with ranges. These ranges 
            would not be associated with any parent collection.
        return_index : bool, default False
            Whether to return a list of index arrays for each collection which 
            relates each new range to its parent range(s) within the collection 
            with a range index.
        null_index : int, optional
            Value to use in returned indices for new ranges which do not 
            intersect with a provided collection. If not provided, will use 
            np.nan.
        **kwargs
            Keyword arguments to be passed to the initialization function for 
            the new RangeCollection instance.
        """
        # Validate null index
        if null_index is None:
            null_index = np.nan
        elif not isinstance(null_index, int):
            raise ValueError("If provided, null_index must be an integer.")
        # Validate input
        for obj in objs:
            if not isinstance(obj, cls):
                raise TypeError(
                    "Input objects must be list-like of RangeCollection "
                    "instances.")

        # Create range collection from all unique range begin and end values
        unique = np.unique(np.concatenate([obj.rng for obj in objs], axis=1))
        rc = RangeCollection.from_breaks(unique, **kwargs)
        
        # If requested, return indices for each original range
        if return_index or not fill_gaps:
            masks = []
            indices = []
            for obj in objs:
                # Check for range content
                if obj.num_ranges > 0:
                    # Intersect all parent collection ranges with new collection
                    mask = rc.set_closed('neither').intersecting(
                        obj.begs, obj.ends, closed='neither', squeeze=False)
                    masks.append(mask.any(axis=1))
                    # Determine parent range; if none, return an index of 
                    # null_index
                    if mask.size == 0:
                        indices.append(np.full(rc.num_ranges, null_index))
                    else:
                        argmax = mask.argmax(axis=1)
                        indices.append(
                            np.where(mask.max(axis=1), argmax, null_index))
                # If a range is empty, append default data
                else:
                    masks.append(np.zeros(rc.num_ranges, dtype=bool))
                    indices.append(np.full(rc.num_ranges, null_index))
        
        # Remove gaps if requested
        if not fill_gaps:
            select = np.any(masks, axis=0)
            rc = rc[select]
            indices = [index[select] for index in indices]

        # Return result
        return (rc, indices) if return_index else rc
    
    @classmethod
    def concatenate(cls, objs=[], reset_centers=True, sort=True, **kwargs):
        """
        Concatenate multiple RangeCollection instances into a single instance 
        without modifying each collection's range data.
        
        Parameters
        ----------
        objs : list-like of RangeCollection instances
            A selection of RangeCollection object instances to be combined 
            into a single instance based on the input parameters.
        reset_centers : boolean, default True
            Whether to reset the centers of the resulting RangeCollection, 
            resulting in the use of dynamic true centers.
        sort : boolean, default True
            Whether to sort the resulting RangeCollection instance upon 
            initialization.
        **kwargs
            Keyword arguments to be passed to the initialization function for 
            the new RangeCollection instance.
        """
        # Validate input
        for obj in objs:
            if not isinstance(obj, cls):
                raise TypeError(
                    "Input objects must be list-like of RangeCollection "
                    "instances.")
        
        # Determine range parameters
        arr = np.concatenate([obj.arr for obj in objs], axis=0)
        keys = np.concatenate([obj.keys for obj in objs])
        # Combine collections
        rc = cls.from_array(arr, keys=keys, sort=sort, **kwargs)
        # Modify new collection
        if reset_centers:
            rc.reset_centers(inplace=True)
        return rc
    
    @classmethod
    def random_int(cls, size=10, beg_bounds=(0,10), end_bounds=(0,10), 
                   random_center=False, **kwargs):
        """
        Create a randomly-generated collection of integer ranges based on the 
        defined parameters.

        Parameters
        ----------
        size : int, default 10
            The number of ranges to be generated for the collection.
        beg_bounds, end_bounds : tuple, default (0,10)
            Two-element tuples of min and max begin and end values to use when 
            generating the collection.
        random_center : bool, default False
            Whether to generate random center values for the generated 
            collection. If False, true_centers will be used.
        """
        # Validate input
        if type(beg_bounds) is int:
            beg_bounds = (0, beg_bounds)
        
        if end_bounds is None:
            end_bounds = beg_bounds
        elif type(end_bounds) is int:
            end_bounds = (0, end_bounds)
                    
        # Define bounds based on user input
        begs_ = np.random.randint(beg_bounds[0], beg_bounds[1], size)
        ends_ = np.random.randint(end_bounds[0], end_bounds[1], size)
        begs = np.min([begs_, ends_], axis=0)
        ends = np.max([begs_, ends_], axis=0)
        
        # Generate and return the range collection
        rc = cls(begs=begs, ends=ends, **kwargs)
        
        # Define centers based on user input
        if random_center:
            rc.randomize_centers(inplace=True)
        
        return rc
    
    @classmethod
    def random_float(cls, size=10, beg_bounds=(0,10), end_bounds=(0,10), 
                     random_center=False, **kwargs):
        """
        Create a randomly-generated collection of float ranges based on the 
        defined parameters.

        Parameters
        ----------
        size : int, default 10
            The number of ranges to be generated for the collection.
        beg_bounds, end_bounds : tuple, default (0,10)
            Two-element tuples of min and max begin and end values to use when 
            generating the collection.
        random_center : bool, default False
            Whether to generate random center values for the generated 
            collection. If False, true_centers will be used.
        """
        # Validate input
        if type(beg_bounds) in [int, float]:
            beg_bounds = (0, beg_bounds)
        
        if end_bounds is None:
            end_bounds = beg_bounds
        elif type(end_bounds) in [int, float]:
            end_bounds = (0, end_bounds)
                    
        # Define bounds based on user input
        begs_ = np.random.random(size) * \
                (beg_bounds[1] - beg_bounds[0]) + beg_bounds[0]
        ends_ = np.random.random(size) * \
                (end_bounds[1] - end_bounds[0]) + end_bounds[0]
        begs = np.min([begs_, ends_], axis=0)
        ends = np.max([begs_, ends_], axis=0)
        
        # Generate the range collection
        rc = cls(begs=begs, ends=ends, **kwargs)
        
        # Define centers based on user input
        if random_center:
            rc.randomize_centers(inplace=True)
        
        return rc
    
    @classmethod
    def random_consecutive(cls, size=10, beg=0, end=10, **kwargs):
        """
        Create a randomly-generated collection of consecutive float ranges 
        based on the defined parameters.

        Parameters
        ----------
        size : int, default 10
            The number of ranges to be generated for the collection.
        beg, end : scalar, default 0, 10
            Begin and end points to define the collection of consecutive ranges 
            being generated.
        """
        # Define breaks
        breaks = np.random.random(size=size)
        breaks = breaks.cumsum() / breaks.sum() * (end-beg) + beg
        breaks = np.concatenate([[beg], breaks])
        
        # Generate and return the range collection
        return cls.from_breaks(breaks=breaks, **kwargs)

    def append(self, begs, ends, centers=None):
        """
        Append the input begin, end, and centers data to the collection. If 
        centers in the target collection are defined dynamically, do not 
        provide centers data.

        Parameters
        ----------
        begs, ends : numeric or array-like
            A single numerical value or array-like of the same representing 
            the begin and end points of each range to be appended to the 
            collection.
        centers : numeric or array-like, optional
            A single numerical value or array-like of the same representing 
            the center points of each range to be appended to the collection. 
            If the centers data of the target collection are defined 
            dynamically (e.g., rc.center_type!='data'), this data should not be 
            provided and the parameter should be left as None.
        """
        # Validate input begs and ends
        if not type(begs) is type(ends):
            raise TypeError("Input begs and ends must have the same type.")
        if isinstance(begs, (int, float)):
            begs = np.array([begs])
        else:
            try:
                begs = np.array(begs)
            except:
                raise TypeError("Input begs must be number or array-like.")
        if isinstance(ends, (int, float)):
            ends = np.array([ends])
        else:
            try:
                ends = np.array(ends)
            except:
                raise TypeError("Input ends must be number or array-like.")
        # Validate equal length arrays
        if not len(begs) == len(ends):
            raise ValueError("Input begs and ends must be equal in length.")
        # Validate centers
        if self.center_type == 'data':
            if centers is None:
                raise ValueError(
                    "Centers data must be provided when the target collection "
                    "has centers data (e.g., rc.center_type=='data').")
            elif isinstance(centers, (int, float)):
                centers = np.array([centers])
            else:
                try:
                    centers = np.array(centers)
                except:
                    raise TypeError(
                        "Input centers must be number or array-like.")
            # Validate equal length arrays
            if not len(begs) == len(centers):
                raise ValueError(
                    "Input begs, ends, and centers must be equal in length.")
        elif not centers is None:
            raise ValueError(
                "Centers data cannot be provided when the target collection "
                "uses a dynamic centers definition (e.g., "
                "rc.center_type!='data').")
        
        # Combine data
        begs = np.append(self._begs, begs)
        ends = np.append(self._ends, ends)
        if not centers is None:
            centers = np.append(self._centers, centers)
        else:
            centers = self._centers
        # Generate new collection and return
        rc = self.__class__(
            begs=begs, ends=ends, centers=centers, closed=self.closed)
        return rc
    
    def reset_keys(self, keys=None, inplace=False):
        """
        Reset key values to enumerate ranges within the collection.
        """
        if keys is None:
            keys = np.arange(0,self.num_ranges,1,dtype=int)
        else:
            if len(keys) != self.num_ranges:
                raise ValueError(
                    "Keys array must have a length equal to the number of "
                    "ranges in the collection.")
            else:
                keys = np.asarray(keys, dtype=int)
        
        if inplace:
            self._keys = keys
        else:
            rc = self.copy()
            rc._keys = keys
            return rc
        
    def reset_centers(self, inplace=False):
        """
        Reset center values to dynamically computed true centers.
        """
        if inplace:
            self.set_centers(centers=None, inplace=True)
            return
        else:
            rc = self.copy()
            rc.set_centers(centers=None, inplace=True)
            return rc
        
    def randomize_centers(self, inplace=False):
        """
        Reset center values with random float values between [beg,end).
        """
        # Generate random centers
        centers = np.random.uniform(size=self.num_ranges) * \
            (self.ends-self.begs)+self.begs
        if inplace:
            self.set_centers(centers=centers, inplace=True)
            return
        else:
            rc = self.copy()
            rc.set_centers(centers=centers, inplace=True)
            return rc
    
    def set_closed(self, closed='right', inplace=False):
        """
        Change whether ranges are closed on left, right, both, or neither side. 
        
        Parameters
        ----------
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, default 'right'
            Whether collection intervals are closed on the left-side, 
            right-side, both or neither.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        if closed in self._ops_closed:
            if inplace:
                self._closed = closed
                self._closed_base = closed.replace('_mod','')
                self._set_mod_locs()
            else:
                rc = self.copy()
                rc._closed = closed
                rc._closed_base = closed.replace('_mod','')
                rc._set_mod_locs()
                return rc
        else:
            raise ValueError(
                "Collection's closed parameter must be one of "
                f"{self._ops_closed}.")
    
    def set_centers(self, centers=None, snap=False, inplace=False, copy=True):
        """
        Change the set range center information.
        
        Parameters
        ----------
        centers : np.array or {'true_centers', 'begs', 'ends'}, optional
            What to set as the new range centers information. Can provide 
            a 1D array-like with a number of elements equal to the number of 
            ranges in the collection, or a string indicating which collection
            property to set as the static center values. If None is provided,
            the collection will default to dynamically calculated true center
            values.
        snap : boolean, default False
            Whether to snap center values which fall outside the bounds of 
            an associated range to the nearest bound.
        """
        # Validate input
        centers = self._validate_centers(centers=centers, snap=snap, copy=copy)
        
        # Set the new centers information
        if inplace:
            self._centers = centers
        else:
            rc = self.copy()
            rc._centers = centers
            return rc
        
    def sample(self, size=1, seed=None):
        """
        Return a random simple of the range collection of the 
        given size and state.
        """
        # Validate input size
        if size > self.num_ranges:
            raise ValueError("Input size cannot be greater than the number \
of ranges in the collection.")
        elif size < 1:
            raise ValueError("Input size must be at least one.")
        
        # Select random sample of given size
        
        index = np.random.RandomState(seed=seed)\
                    .randint(0, self.num_ranges, size=size)
        return self[index]
    
    def copy(self, deep=False):
        """
        Create an exact copy of the range collection object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
    
    def iterranges(self):
        """
        """
        for i in range(self.num_ranges):
            yield self[i]
    
    def plot(self, ax=None, figsize=None, centers=True, closed=True,
             one_line=False, voffset=0, *args, **kwargs):
        """
        Create a simple plot of the range collection using matplotlib's pyplot 
        interface. All provided **kwargs are passed directly to the ax.plot() 
        axis method call.
        """
        # Create figure and subplot
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
        # Iterate over all ranges and plot individually
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, rng in enumerate(self.iterranges()):
            # Select range color
            color = colors[i % len(colors)]
            # Plot the range line
            ax.plot([rng.begs, rng.ends], 
                ([voffset, voffset] if one_line else [-i+voffset, -i+voffset]),
                color=color, **kwargs)
            # Plot center points
            if centers:
                ax.scatter(rng.centers, voffset if one_line else -i+voffset,
                    color=color, **kwargs)
            # Plot closed edges
            if closed:
                left_closed = \
                    (self._closed in ['left','both','left_mod']) or \
                    (self._closed in ['right_mod'] and self._mod_locs[i])
                right_closed = \
                    (self._closed in ['right','both','right_mod']) or \
                    (self._closed in ['left_mod'] and self._mod_locs[i])
                if left_closed:
                    ax.scatter(rng.begs, voffset if one_line else -i+voffset,
                        marker=4, s=120, color=color, **kwargs)
                if right_closed:
                    ax.scatter(rng.ends, voffset if one_line else -i+voffset,
                        marker=5, s=120, color=color, **kwargs)
        return ax
    
    def search(self, loc=None, by='begs', choose='first'):
        """
        Select all locations which match the input location value in terms of 
        the chosen search-by property.
        
        Parameters
        ----------
        loc : numeric
            The numeric location value to search for within the range 
            collection.
        by : {'centers', 'begs', 'ends', 'true_centers'}, default 'begs'
            Which range property to consider when searching for the target 
            range.
        choose : {'first', 'last', 'all'}, default 'first'
            Which range to return information for if multiple ranges are found 
            which intersect with the provided location.
        
        NOTE: Previously named 'findby'
        """
        # Validate selected reference point information
        by_options = {'centers', 'begs', 'ends', 'true_centers'}
        if not by in by_options:
            raise ValueError(f"Reference 'by' must be one of {by_options}.")
        else:
            ref = getattr(self, by)
        
        # Find relevant range indices
        index = np.where(ref==loc)[0]

        # Make selection
        if choose == 'first':
            return self[index[0]]
        elif choose == 'last':
            return self[index[-1]]
        elif choose == 'all':
            return self[index]
        else:
            raise ValueError("Choose parameter must be first, last, or all.")
        
    def locate(self, loc, choose='first', snap=None, return_dist=False, *args, 
               **kwargs):
        """
        Get the index of the range which intersects the input location value
        as well as the proportional position along that range where it falls.
        
        Parameters
        ----------
        loc : numeric
            The numeric location value to search for within the range 
            collection.
        choose : {'first', 'last', 'all'}, default 'first'
            Which range to return information for if multiple ranges are found 
            which intersect with the provided location.
        snap : {None, 'near', 'left', 'right'}, default None
            If the input location does not fall within any ranges, snap to the 
            nearest match based on distance, choosing the closest range to the 
            left, right, or either side ('near'). If None, a value error will 
            be raised when no intersecting ranges are found.
        return_dist : boolean, default False
            Whether to return the positive or negative distance of the input
            range location value from the selected range index and position.
            If the location falls within a valid range, the distance will 
            always be zero. If it does not fall within a valid range, the 
            distance will be equal to the positive distance to the nearest
            range begin point or the negative distance to the nearest range
            end point, depending on the selected snapping parameter.
            
        Returns
        -------
        index : int or array of ints
            The index or indices of the range or ranges which intersect the 
            input location value.
        position : float or array of floats
            The proportional distance along the range or ranges where the 
            input location value falls.
        """
        self._validate_monotonic()
        # Get index of ranges which intersect the input location
        index = np.where(self.intersecting(beg=loc))[0]
        
        # Check for at least one range
        if len(index) == 0:
            if snap is None:
                raise ValueError(
                    f"Location ({loc}) does not fall within any valid ranges.")
            
            # Find the nearest match if requested
            #  - Analyze gaps
            diff_right = self.begs - loc
            diff_left  = self.ends - loc
            index = (np.argmin(np.where(diff_right < 0, np.inf, diff_right)),
                     np.argmax(np.where(diff_left > 0, -np.inf, diff_left)))
            
            # Find the nearest, either direction
            if snap == 'near':
                dist = (diff_right[index[0]], diff_left[index[1]])
                select = np.argmin(np.abs(dist))
                dist = (dist[select],) if return_dist else ()
                return (index[select], [0.0, 1.0][select]) + dist
            
            # Find the nearest, to the right
            elif snap == 'right':
                dist = (diff_right[index[0]],) if return_dist else ()
                return (index[0], 0.0) + dist
            
            # Find the nearest, to the left
            elif snap == 'left':
                dist = (diff_left[index[1]],) if return_dist else ()
                return (index[1], 1.0) + dist
        
        # At least one range intersected, make selection
        else:
            # Compute fractional distances
            rc = self[index]
            position = (loc-rc.begs) / np.where(rc.lengths>0,
                                                rc.lengths, np.inf)
            
            # Assign distance value if requested
            if return_dist:
                dist = (0,)
            else:
                dist = ()
            
            # Choose results if requested
            if choose == 'first':
                return (index[0], position[0]) + dist
            elif choose == 'last':
                return (index[-1], position[-1]) + dist
            elif choose == 'all':
                return (index, position) + dist
            else:
                raise ValueError(
                    "Choose parameter must be 'first', 'last', or 'all'.")
    
    def project(self, index, dist):
        """
        Get the location value on the indexed range at the given proportional
        distance along that range.
        """
        self._validate_monotonic()
        # Validate distance value
        if dist < 0 or dist > 1:
            raise ValueError(f"Distance ({dist}) must be between 0 and 1.")
            
        # Retrieve reference range
        try:
            rc = self[index]
        except IndexError:
            raise IndexError(f"Invalid index ({index}); must be integer \
between 0 and {self.num_ranges - 1}.")

        val = rc.begs[0] + rc.lengths[0] * dist
        return val
    
    def is_before(self, loc=None, **kwargs):
        """
        Get boolean mask for ranges which fall entirely before the given range 
        location.
        """
        self._validate_monotonic()
        
        # Validate input points
        if loc is None:
            raise ValueError("No input point provided.")
        
        # Test for intersecting
        if self._closed in ['right', 'both']:
            t1 = self.ends  < loc
        elif self._closed in ['left', 'neither']:
            t1 = self.ends <= loc
        
        return t1
    
    def is_behind(self, loc=None, **kwargs):
        """
        Get boolean mask for ranges which fall entirely behind the given range 
        location.
        """
        self._validate_monotonic()
        
        # Validate input points
        if loc is None:
            raise ValueError("No input point provided.")
        
        # Test for intersecting
        if self._closed in ['left', 'both']:
            t1 = self.begs  > loc
        elif self._closed in ['right', 'neither']:
            t1 = self.ends >= loc
        
        return t1

    def snap(self, loc, **kwargs):
        """
        Snap a provided location value to fall within the bounds of a range 
        within the collection. If the location falls within the bounds of at 
        least one range, the value will be returned unchanged.

        Note: the process assumes ranges are closed on both sides.

        Parameters
        ----------
        loc : scalar
            Location value to snap to a range.
        """
        # Check if already intersecting
        if self.set_closed('both').intersecting(beg=loc, closed='both').any():
            return loc
        else:
            # Snap to bounds
            bounds = self.pairs.flatten()
            return bounds[np.abs(bounds - loc).argmin()]
    
    def intersecting(self, beg=None, end=None, other=None, closed='both', 
                     squeeze=True, **kwargs):
        """
        Get boolean mask for ranges which intersect the given range values. If 
        multiple ranges given, return a mask array with a second dimension 
        equal to the number of ranges provided.

        Parameters
        ----------
        beg, end : numerical or array-like, optional
            The begin and end locations of the range or ranges to be tested. If 
            a single range is to be tested, provide a numeric value. If 
            multiple, provide an array-like with a single begin and end value 
            for each range. If no end parameter provided, point locations will 
            be assumed and end will be set equal to beg. Not required if other 
            parameter is used.
        other : RangeCollection, optional
            Other RangeCollection instance to be intersected with this one. Can 
            be provided instead of beg, end, and closed parameters and will 
            take precedence over other input.
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, default 'right'
            Whether collection intervals are closed on the left-side, 
            right-side, both or neither.
        squeeze : boolean, default True
            Whether to reduce the dimensions of the output array to 1D if only 
            a single begin/end location was provided. If False, output array 
            will always be 2D.

        Returns
        -------
        mask : np.ndarray
            A numpy array with a first dimension equal to the number of ranges 
            in the collection and a second dimension equal to the number of 
            ranges being tested against the collection. If a single numerical 
            value is provided for the begin and end test range, a 1d-array will 
            be returned.
        """
        self._validate_monotonic()
        
        # Validate input events
        if not other is None:
            if not isinstance(other, RangeCollection):
                raise TypeError(
                    "If provided, input other parameter must be valid "
                    "RangeCollection instance.")
            rc = other
        else:
            rc = self.__class__(beg, end, closed=closed, sort=False, )
        
        # Reshape range data for testing
        left_begs = self.begs.reshape(-1,1)
        left_ends = self.ends.reshape(-1,1)
        right_begs = rc.begs.reshape(1,-1)
        right_ends = rc.ends.reshape(1,-1)

        # Initial testing
        logic = np.greater(left_ends, right_begs) & \
            np.less(left_begs, right_ends)

        # If edges are in play for both sets of ranges, adjust accordingly
        if not 'neither' in [self._closed_base, rc._closed_base]:
            # Prepare edge logic arrays
            left_edges  = _ArrayLogicManager(left_begs, right_ends)
            right_edges = _ArrayLogicManager(left_ends, right_begs)

            # Adjust for standard edges
            if self._closed_base in ['left','both'] and \
                rc._closed_base in ['right','both']:
                logic |= left_edges.equal
            if self._closed_base in ['right','both'] and \
                rc._closed_base in ['left','both']:
                logic |= right_edges.equal

            # Adjust for modified edges
            # - Check left range collection
            if self._closed in ['left_mod'] and \
                rc._closed_base in ['left','both']:
                logic[self._mod_locs,:] |= right_edges.equal[self._mod_locs,:]
            elif self._closed in ['right_mod'] and \
                rc._closed_base in ['right','both']:
                logic[self._mod_locs,:] |= left_edges.equal[self._mod_locs,:]
            # - Check right range collection
            if rc._closed in ['left_mod'] and \
                self._closed_base in ['left','both']:
                logic[:,rc._mod_locs] |= left_edges.equal[:,rc._mod_locs]
            elif rc._closed in ['right_mod'] and \
                self._closed_base in ['right','both']:
                logic[:,rc._mod_locs] |= right_edges.equal[:,rc._mod_locs]
        
        # Combine test results and squeeze if requested
        if logic.shape[1] == 1 and squeeze:
            logic = logic.flatten()

        # Return final test results
        return logic

    def _set_mod_locs(self):
        """
        Get indexes of ranges with modified edges. Only applicable when 
        self.closed in ['left_mod','right_mod'].
        """
        # Require minimum ranges
        if self.num_ranges == 0:
            mod_locs = np.zeros(self.begs.shape, dtype=bool)
        # Modify test for specific closed cases
        elif self.closed in ['left_mod']:
            # Identify ends of group ranges which will be modified
            mod_locs = self.are_overlapping(all_=False, when_one=np.array([], dtype=bool), enforce_edges=True)
            mod_locs = np.append(~mod_locs, True)
        elif self.closed in ['right_mod']:
            # Identify ends of group ranges which will be modified
            mod_locs = self.are_overlapping(all_=False, when_one=np.array([], dtype=bool), enforce_edges=True)
            mod_locs = np.append(True, ~mod_locs)
        else:
            mod_locs = np.zeros(self.begs.shape, dtype=bool)
        self._mod_locs = mod_locs

    @property
    def mod_locs(self):
        """
        Get indexes of ranges with modified edges. Only applicable when 
        self.closed in ['left_mod','right_mod'].
        """
        return self._mod_locs

    def is_concentric(self, center=None):
        """
        Get boolean mask for ranges which have a center equal to the given 
        center value.

        Parameters
        ----------
        center : scalar
            Location of center test point.
        """
        self._validate_monotonic()
        return (self.centers == center)
    
    def is_inside(self, beg=None, end=None):
        """
        Get boolean mask for ranges which fall entirely within the given range 
        values.

        Parameters
        ----------
        beg, end (optional) : scalar
            Begin and end locations of the overlaid range.
        """
        self._validate_monotonic()
        if beg is None:
            raise ValueError("Must provide begin location.")
        elif end is None:
            end = beg
        return ((self.begs >= beg) & (self.ends <  end)) | \
               ((self.begs >  beg) & (self.ends <= end))
    
    def is_outside(self, beg=None, end=None):
        """
        Get boolean mask for ranges which entirely eclipse the given range 
        values.

        Parameters
        ----------
        beg, end (optional) : scalar
            Begin and end locations of the overlaid range.
        """
        self._validate_monotonic()
        if beg is None:
            raise ValueError("Must provide begin location.")
        elif end is None:
            end = beg
        return (self.begs < beg) & (self.ends > end)
    
    def is_on_left(self, beg=None, end=None):
        """
        Get boolean mask for ranges which overlap only the left side of the 
        given range values.

        Parameters
        ----------
        beg, end (optional) : scalar
            Begin and end locations of the overlaid range.
        """
        self._validate_monotonic()
        if beg is None:
            raise ValueError("Must provide begin location.")
        elif end is None:
            end = beg
        return (self.begs < beg) & (self.ends > beg) & (self.ends <= end)
    
    def is_on_right(self, beg=None, end=None):
        """
        Get boolean mask for ranges which overlap only the right side of the 
        given range values.

        Parameters
        ----------
        beg, end (optional) : scalar
            Begin and end locations of the overlaid range.
        """
        self._validate_monotonic()
        if beg is None:
            raise ValueError("Must provide begin location.")
        elif end is None:
            end = beg
        return (self.ends > end) & (self.begs < end) & (self.begs >= beg)
    
    def is_same(self, beg=None, end=None):
        """
        Get boolean mask for ranges which have the same end points as the given 
        range values.

        Parameters
        ----------
        beg, end (optional) : scalar
            Begin and end locations of the overlaid range.
        """
        self._validate_monotonic()
        if beg is None:
            raise ValueError("Must provide begin location.")
        elif end is None:
            end = beg
        return (self.begs == beg) & (self.ends == end)
    
    def are_overlapping(self, all_=True, when_one=True, sort=False, 
            enforce_edges=False, **kwargs):
        """
        Whether all or any ranges are overlapping the next range in the 
        collection.

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of overlapping ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            overlapping, False if any adjacent ranges are not overlapping. If 
            False, will return an array of shape num_ranges - 1 of boolean 
            values indicating whether each range is overlapping the next.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        enforce_edges : bool, default False
            Whether to consider ranges which have a common vertex as 
            overlapping. This is independent of the collection's closed state.
        """
        # Validate input
        if self.num_ranges == 1:
            return when_one
        elif self.num_ranges == 0:
            raise ValueError("No ranges in collection.")

        # Check for overlapping
        if enforce_edges:
            res = (self.begs[1:] <= self.ends[:-1])
        else:
            res = (self.begs[1:] < self.ends[:-1])
        if all_:
            return res.all()
        else:
            return res
    
    def are_consecutive(self, all_=True, when_one=True, sort=False):
        """
        Whether all or any ranges are consecutive with the next range in the 
        collection.

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of consecutive ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            consecutive, False if any adjacent ranges are not consecutive. If 
            False, will return an array of shape num_ranges - 1 of boolean 
            values indicating whether each range is consecutive to the next.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        """
        # Validate input
        if self.num_ranges == 1:
            return np.array([when_one])
        elif self.num_ranges == 0:
            raise ValueError("No ranges in collection.")
        
        # Check for consecutive ranges
        res = (self.begs[1:] == self.ends[:-1])
        if all_:
            return res.all()
        else:
            return res
    
    def are_monotonic(self, all_=True, when_one=True):
        """
        Whether all ranges are increasing (i.e., self.begs >= self.ends).

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of monotonic ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            monotonic, False if any adjacent ranges are not monotonic. If 
            False, will return an array of shape num_ranges of boolean values 
            indicating whether each range is monotonic or not.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        """
        # Validate input
        if self.num_ranges == 1:
            return when_one
        elif self.num_ranges == 0:
            raise ValueError("No ranges in collection.")
        
        # Check for monotonic
        res = (self.ends >= self.begs)
        if all_:
            return res.all()
        else:
            return res
    
    def overlay(self, beg=None, end=None, normalize=True, how='right', 
                norm_zero=None, squeeze=True, validate=True, **kwargs):
        """
        Compute overlap of the input bounds with respect to collection ranges.
        
        Parameters
        ----------
        beg, end : scalar or array of scalars
            Begin and end locations of the overlaid range(s).
        normalize : boolean, default True
            Whether overlapping lengths should be normalized range length to 
            give a proportional result.
        how : {'right','left','sum'}, default 'right'
            How overlapping lengths should be normalized. Only applied when 
            normalize=True.

            right : Normalize overlaps by the length of each provided overlay 
                range.
            left : Normalize overlaps by the length of each of the collection's 
                ranges being overlaid.
            sum : Normalize overlaps by the sum of the lengths of all overlaps 
                for each provided overlay range. If there are gaps in the 
                collection's ranges or overlaps between the collection's 
                ranges, this will allow the sum of the overlaps to still equal 
                1.0, except where no overlaps occur.
        norm_zero : float, optional
            A number to substitute for instances where the normalizing factor 
            (denominator) is equal to zero, e.g., when the overlay range has a 
            length of zero and how='right'. If not provided, all instances of 
            zero division will return float value 0.0.
        squeeze : boolean, default True
            Whether to reduce the dimensions of the output array to 1D if only 
            a single begin/end location was provided. If False, output array 
            will always be 2D.
        validate : boolean, default True
            Whether to validate the input begin and end location information. 
            Unless externally validated, always use True.

        Modified: 12/21/2020
        """
        self._validate_monotonic()
        #----------------#
        # VALIDATE INPUT #
        #----------------#
        # Validate input events
        if validate:
            beg, end = self._validate_beg_end(beg, end)
        # Validate normalization parameters
        try:
            how_ops = {'right','left','sum'}
            how = str(how).lower()
            assert(how in how_ops)
        except:
            raise ValueError("Invalid input for normalization how parameter. "
                f"Must be one of {how_ops}.")
        if norm_zero is None:
            norm_zero = 0.0
        # Address deprecated parameters
        if 'minus_gaps' in kwargs:
            warnings.warn("Use of 'minus_gaps' is deprecated, instead use "
                "equivalent how='sum' for True or how='right' for False.", 
                DeprecationWarning)
            if kwargs['minus_gaps']:
                how = 'sum'
            else:
                how = 'right'
        
        #-----------------#
        # PERFORM OVERLAY #
        #-----------------#
        # Compute the overlap lengths
        lefts  = self.ends.reshape((-1,1)) - beg.reshape((1,-1))
        rights = end.reshape((1,-1)) - self.begs.reshape((-1,1))
        # Compare against range lengths and bounds
        bounds  = np.tile(end-beg, (self.num_ranges,1))
        lengths = np.tile(self.lengths, (beg.size,1)).T
        overlap = np.stack([lefts, rights, lengths, bounds], axis=0)
        overlap = np.nanmin(overlap, axis=0).clip(min=0)
        
        # Normalize lengths to get weights
        if normalize:
            if how == 'sum':
                denom = np.tile(overlap.sum(axis=0), (self.num_ranges,1))
            elif how == 'right':
                denom = bounds
            elif how == 'left':
                denom = lengths
            # Address zero division and apply to normalize
            overlap = overlap / np.where(denom==0, np.inf, denom)
            overlap = np.where(denom==0, norm_zero, overlap)
        
        # Shape array based on input and return
        if squeeze and beg.size==1:
            return overlap[:,0]
        else:
            return overlap

    def _validate_beg_end(self, beg, end, arr=True):
        """
        Validate input begin and end data for use in overlay, intersecting, 
        and similar methods.
        """
        if beg is None:
            raise ValueError("No input range(s) provided.")
        elif end is None:
            end = beg
        if arr:
            try:
                beg = np.asarray(beg, dtype=float).flatten()
                end = np.asarray(end, dtype=float).flatten()
            except ValueError:
                raise ValueError("Input begin and end locations must be "
                    "provided as single numerical values or array-likes of "
                    "numerical values.")
        # Return validate begin and end data
        return beg, end

    def _validate_closed_other(self, closed):
        """
        Modify the collection's closed parameter based on the input closed 
        parameter. This is used to account for instances of intersecting other 
        ranges with this collection when those ranges have a closed parameter 
        other than both.
        """
        # Validate option
        if not closed in self._ops_closed:
            raise ValueError(
                "Input closed parameter must be one of "
                f"{self._ops_closed}.")
                
        # Invert function
        def invert(op):
            if op == 'left':
                return 'right'
            elif op == 'right':
                return 'left'
            else:
                return op

        # Simplify combinations of closed parameters
        if closed == 'both':
            return self._closed
        elif (closed == 'neither') or (self._closed == 'neither'):
            return 'neither'
        elif closed != self._closed_base:
            return self._closed_base
        elif self._closed == 'both':
            return invert(closed)
        elif 'mod' in self._closed:
            return invert(closed) + '_end'
        else:
            return 'neither'
    
    def sortranges(self, by='begs', ascending=True, return_index=False, 
                   return_inverse=False, inplace=False):
        """
        Sort the ranges by range collection property values.
        
        Parameters
        ----------
        by : {'begs', 'ends', 'centers', true_centers'}, default 'begs'
            The range collection property by which all ranges should be sorted.
        ascending : boolean, default True
            Whether sorting should be done in ascending order. When False, 
            ranges will be sorted in descending order.
        return_index : boolean, default False
            Whether to return an array of the indices used to perform the sort
            in addition to the sorted range collection.
        return_inverse : boolean, default False
            Whether to return an array of the indices which represent the 
            inverse of the performed sort in addition to the sorted range 
            collection.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        if not type(by) is list:
            by = [by]
        if type(ascending) is bool:
            ascending = [ascending for x in range(len(by))]
        elif type(ascending) is list and not len(ascending) == len(by):
            raise ValueError("'ascending' parameter must be single boolean "
                "value or must be list of same length as 'by'.")
        
        # Get the arrays for lexsort
        ascending = [1 if x else -1 for x in ascending[::-1]] # Reverse order
        by = [ascending[i] * getattr(self, x) for i, x in enumerate(by[::-1])]
        
        index = np.lexsort(by)
        
        if inplace:
            self._select_self(index)
        else:
            rc = self[index]
            if return_index:
                if return_inverse:
                    return rc, index, np.argsort(index)
                else:
                    return rc, index
            if return_inverse:
                return rc, np.argsort(index)
            else:
                return rc
        
    def find_inside(self, enforce_edges=False):
        """
        Find all ranges which fall inside of at least one other range.
        
        Parameters
        ----------
        """
        self._validate_monotonic()

        # Sort ranges by begin and end location
        rc, inv = self.sortranges(by=['begs', 'ends'],
                                  ascending=[True, False],
                                  return_inverse=True)
        
        # Find dominating range extents
        cummax = np.maximum.accumulate(rc.ends)
        unique, uindex, uinv = np.unique(cummax, return_index=True,
                                         return_inverse=True)
        cummin = rc.begs[uindex[uinv]]
        
        # Identify all inside ranges
        if enforce_edges:
            index = (((rc.begs >= cummin) & (rc.ends <  cummax)) |
                     ((rc.begs >  cummin) & (rc.ends <= cummax)))
        else:
            index = (rc.begs > cummin) & (rc.ends < cummax)
        
        # Unsort index
        return np.where(index[inv])[0]
    
    def find_unique(self, keepfirst=True):
        """
        """
        self._validate_monotonic()
        
        # Analyze unique combinations of begin and end points
        unique, uindex, ucounts = np.unique(
            self.values, axis=1, 
            return_index=True,
            return_counts=True
        )
        
        # Ignore the first instance of non-unique ranges
        if not keepfirst:
            index = uindex[ucounts == 1]
        else:
            index = uindex
        
        return index
        
    def find_same(self, keepfirst=True):
        """
        """
        self._validate_monotonic()
        
        # Get inverse of all unique ranges
        index_unique = self.find_unique(keepfirst=keepfirst)
        
        # Invert the unique indexes
        index = np.ones(self.num_ranges, dtype=bool)
        index[index_unique] = False
        return np.where(index)[0]
        
    def find_nonconcentric(self, keepfirst=True):
        """
        """
        self._validate_monotonic()
        
        unique, uindex, ucounts = np.unique(
            self.centers, 
            return_index=True,
            return_counts=True
        )
               
        # Ignore the first instance of concentric ranges
        if not keepfirst:
            index = uindex[ucounts == 1]
        else:
            index = uindex
        
        return index
        
    def find_concentric(self, keepfirst=True):
        """
        """
        self._validate_monotonic()
        
        # Get inverse of all unique ranges
        index_nonconcentric = self.find_nonconcentric(keepfirst=keepfirst)
        
        # Invert the unique indexes
        index = np.ones(self.num_ranges, dtype=bool)
        index[index_nonconcentric] = False
        return np.where(index)[0]
        
    def set_monotonic(self, increasing=True, inplace=False, **kwargs):
        """
        Arrange all ranges to be increasing or decreasing.

        Parameters
        ----------
        increasing : boolean, default True
            Whether to set range values to be increasing. When False, range
            values will be set to decreasing.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        rng = np.sort(self.rng, axis=0)
        
        if increasing:
            begs = rng[0]
            ends = rng[1]
        else:
            begs = rng[1]
            ends = rng[0]
        
        if inplace:
            self._begs = begs
            self._ends = ends
            self._monotonic = True
            return
        else:
            rc = self.__class__(begs, ends, self._centers, copy=True,
                                force_monotonic=False)
            rc._monotonic = True
            return rc
        
    def drop(self, index, inplace=False):
        """
        Remove ranges at the given indices
        """
        index = np.setdiff1d(self.index, index)
        if inplace:
            self._select_self(index)
            return
        else:
            return self[index].copy()
        
    def drop_short(self, length=0, inplace=False):
        """
        Remove ranges that are less than or equal to a given length.
        """
        index = self.lengths <= length
        return self.drop(np.where(index)[0], inplace=inplace)        
        
    def drop_long(self, length=0, inplace=False):
        """
        Remove ranges that are greater than or equal to a given length.
        """
        index = self.lengths >= length
        return self.drop(np.where(index)[0], inplace=inplace)        
        
    def prepare(self, inplace=False, **kwargs):
        """
        Prepare range collection by making it monotonic and 
        eliminating duplicate and inside ranges.
        """
        rc = self.set_monotonic(**kwargs)\
                 .eliminate_same(**kwargs)\
                 .eliminate_inside(**kwargs)
        if inplace:
            self._begs = rc._begs
            self._ends = rc._ends
            return
        else:
            return rc
        
    def eliminate_inside(self, enforce_edges=False, drop=False, inplace=False, 
                         **kwargs):
        """
        Remove left and right wings from all ranges which fall 
        completely inside of another range.
        """
        self._validate_monotonic()
        # Identify all inside ranges
        index = self.find_inside(enforce_edges=enforce_edges)
        return self.eliminate_index(index, drop=drop, inplace=inplace)
        
    def eliminate_same(self, keepfirst=True, drop=False, inplace=False, 
                       **kwargs):
        """
        Remove left and right wings from all ranges which
        are identical to at least one other, keeping only
        the first.
        """
        self._validate_monotonic()
        # Identify all similar ranges
        index = self.find_same(keepfirst=keepfirst)
        return self.eliminate_index(index, inplace=inplace)

    def eliminate_concentric(self, keepfirst=True, drop=False, inplace=False, 
                             **kwargs):
        """
        Remove left and right wings from all ranges which
        have an identical center to at least one other, keeping only
        the first.
        """
        # Identify all similar ranges
        index = self.find_concentric(keepfirst=keepfirst)
        return self.eliminate_index(index, inplace=inplace)
        
    def eliminate_index(self, index, drop=False, inplace=False, **kwargs):
        """
        Remove left and right wings from all ranges indicated by
        the given array index.
        """
        if inplace:
            self._begs[index] = self.centers[index]
            self._ends[index] = self.centers[index]
            if drop:
                self.drop_short(inplace=True)
            return
        else:
            rc = self.copy(deep=True)
            begs = self._begs.copy()
            ends = self._ends.copy()
            begs[index] = self.centers[index]
            ends[index] = self.centers[index]
            rc._begs = begs
            rc._ends = ends
            if drop:
                rc.drop_short(inplace=True)
            return rc

    def round(self, decimals=0, factor=1, round_centers=False, inplace=False):
        """
        Round the bounds of all ranges to the specified number of decimals 
        or to a specified rounding factor.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimals to round range bound values to.
        factor : scalar, default 1
            Rounding factor to apply to the range bound values. For example, 
            use `factor=0.5` (and `decimals=0`) to round each value to the 
            nearest 0.5.
        round_centers : bool, default False
            Whether to round explicitly defined centers. Only applicable when 
            `self.center_type=='data'`.
        """
        # Perform rounding
        begs = np.round(self.begs / factor, decimals=decimals) * factor
        ends = np.round(self.ends / factor, decimals=decimals) * factor
        if round_centers and (self.center_type=='data'):
            centers = \
                np.round(self.centers / factor, decimals=decimals) * factor
    
        # Apply update
        rc = self if inplace else self.copy(deep=True)
        rc._begs = begs
        rc._ends = ends
        if round_centers and (self.center_type=='data'):
            rc._centers = centers

        # If not inplace, return updated object
        if not inplace:
            return rc
        else:
            return

    def separate(self, by=None, eliminate_inside=False, drop_short=False, 
                 inplace=False, **kwargs):
        """
        Address overlapping ranges by distributing overlaps between adjacent 
        ranges and eliminating eclipsed ranges. Distributions are made equally
        and are based a specified reference point of the center, begin, end, 
        or true center points of each range. Separations are done in a single 
        computation and are not iterative and therefore provide efficient, 
        consistent results for collections of overlapping ranges.
        
        Parameters
        ----------
        by : str {'centers', 'begs', 'ends', 'true_centers'}, optional
            The reference point of each range to be used when distributing 
            overlaps between ranges.
        eliminate_inside : boolean, default False
            Whether to automatically eliminate ranges which are entirely 
            overlapped by other ranges.
        drop_short : boolean, default False
            Whether to automatically drop ranges which have a length of zero or 
            which have otherwise been eliminated.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        
        Returns
        -------
        res : RangeCollection
            The resulting RangeCollection object with all overlapping ranges 
            separated based on the provided parameters.
        """
        self._validate_monotonic()
        #----------------#
        # VALIDATE INPUT #
        #----------------#
        # Validate selected reference point information
        if not by in [None, 'centers', 'begs', 'ends', 'true_centers']:
            raise ValueError("Separate 'by' must be either 'centers', "
                "'begs', 'ends', or 'true_centers' or None.")

        #--------------------#
        # PREPARE RANGE DATA #
        #--------------------#
        # Prepare sorted ranges for processing
        rc = self.copy(deep=True)
        if not by is None:
            rc.set_centers(centers=by, inplace=True)
        rc, inv = rc.sortranges(
            by=['centers', 'lengths'],
            ascending=[True, False],
            inplace=False,
            return_inverse=True
        )
        
        # Eliminate concentric, same, and inside ranges
        rc = rc.eliminate_concentric(**kwargs).eliminate_same(**kwargs)
        if eliminate_inside:
            rc = rc.eliminate_inside(**kwargs)
        index = np.where(rc.lengths > 0)[0]
        
        #---------------#
        # MODIFY RANGES #
        #---------------#
        # Identify the new begin and end points based on computed
        # midpoints and existing begin and end points
        rights    = rc.ends[index[:-1]].copy()
        lefts     = rc.begs[index[1:]].copy()
        centers_l = rc.centers[index[:-1]].copy()
        centers_r = rc.centers[index[1:]].copy()
        
        # Compute midpoints between consecutive centers
        center_mids = (centers_l + centers_r) / 2
        center_mids_valid = (rights >= center_mids) & (lefts <= center_mids)
        
        # Compute midpoints between consecutive termini
        termini_mids = (rights + lefts)/2
        termini_mids = np.min([np.max([termini_mids, centers_l], axis=0),
                               centers_r], axis=0)
        termini_mids_valid = (
            (rights >= termini_mids) &
            (lefts <= termini_mids) &
            (termini_mids >= centers_l)
        )
        
        # Apply termini mids
        rights[termini_mids_valid] = termini_mids[termini_mids_valid]
        lefts[termini_mids_valid]  = termini_mids[termini_mids_valid]
        
        # Apply center mids
        rights[center_mids_valid] = center_mids[center_mids_valid]
        lefts[center_mids_valid]  = center_mids[center_mids_valid]

        # Assign the new begin and end points to the processed ranges
        rc.reset_centers(inplace=True)
        rc._ends[index[:-1]] = rights
        rc._begs[index[1:]]  = lefts
        rc = rc[inv]

        if inplace:
            self._begs = rc._begs
            self._ends = rc._ends
            self.reset_centers(inplace=True)
            # Drop short if requested
            if drop_short:
                self.drop_short(length=0, inplace=True)
            return
        else:
            # Drop short if requested
            if drop_short:
                rc.drop_short(length=0, inplace=True)
            return rc

    def prioritize(self, scores=None, by=None, ascending=True, 
                   drop_short=False, inplace=False, **kwargs):
        """
        Address overlapping ranges by prioritizing ranges either based on a 
        range property (e.g., length, begin point location, etc.) or based on 
        a provided sequence of rankings or rankable values with a length equal 
        to the number of ranges in the collection.
        
        NOTE: When a lower-ranked range totally eclipses a higher-ranked range, 
        it will be cut to the side of the higher-ranked range on which its 
        center value falls relative to the higher-ranked range's center.
        
        Parameters
        ----------
        scores : array-like, optional
            An array-like of values associated with each range in the 
            collection which will be used to rank the ranges before performing 
            the prioritized separation. If the 'by' argument is provided, 
            the 'scores' argument should be None.
        by : {'lengths','centers', 'begs', 'ends', 'true_centers'}, optional
            The range collection property to be used to rank the ranges before 
            performing the prioritized separation. This is equivalent to 
            providing one of these properties to the 'score' argument manually. 
            The 'by' argument is superceded by the 'scores' argument if it is 
            provided.
        ascending : boolean, default True
            Whether rankings should be computed based on the provided scores or 
            range properties in ascending order (True) or descending order 
            (False).
        drop_short : boolean, default False
            Whether to automatically drop ranges which have a length of zero or 
            which have otherwise been eliminated.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        
        Returns
        -------
        res : RangeCollection
            The resulting RangeCollection object with all overlapping ranges 
            separated based on the provided parameters. Ranges which have been 
            eliminated (i.e., zero-ed) can be removed by calling the drop_short 
            method.            
        """
        #----------------#
        # VALIDATE INPUT #
        #----------------#
        # Scores provided via 'by' parameter
        if scores is None:
            if by is None:
                raise ValueError("Must provide either 'scores' or 'by' "
                    "arguments to perform prioritized separation.")
            try:
                # Attempt to retrieve the selected property
                scores = getattr(self, by)
            except AttributeError:
                raise AttributeError("Invalid range collection property "
                    "provided to the 'by' argument. Must be 'lengths', "
                    "'centers', 'begs', 'ends', or similar.")

        # Validate scores information
        try:
            scores = np.asarray(scores, dtype=float).flatten()
            assert scores.size == self.num_ranges
        except:
            raise TypeError("Provided scores data must be numerical, with a "
                "number of scores equal to the number of ranges in the "
                "collection.")
        # Select ascending or descending
        if not ascending:
            scores = scores * -1
        # Prepare rankings
        ranks = np.argsort(scores)

        #------------------------#
        # PERFORM PRIORITIZATION #
        #------------------------#
        # Iterate over ranges by rank and separate
        rc = self.copy(deep=True)
        for i in ranks:
            # Get information for indexed range
            beg = rc.begs[i]
            end = rc.ends[i]
            center = rc.centers[i]
            # Check for eliminated ranges
            if beg == end:
                continue
            # Cut ranges on left
            on_left = rc.is_on_left(beg=beg, end=end)
            rc._ends[on_left] = beg
            # Cut ranges on right
            on_right = rc.is_on_right(beg=beg, end=end)
            rc._begs[on_right] = end
            # Cut ranges outside
            outside = rc.is_outside(beg=beg, end=end)
            left_center = rc.centers <= center
            rc._ends[outside &  left_center] = beg
            rc._begs[outside & ~left_center] = end
            # Eliminate ranges inside or the same
            inside = rc.is_inside(beg=beg, end=end)
            same = rc.is_same(beg=beg, end=end)
            select = inside | same
            select[i] = False
            rc = rc.eliminate_index(select)
        
        if inplace:
            self._begs = rc._begs
            self._ends = rc._ends
            self.reset_centers(inplace=True)
            # Drop short if requested
            if drop_short:
                self.drop_short(length=0, inplace=True)
            return
        else:
            # Drop short if requested
            if drop_short:
                rc.drop_short(length=0, inplace=True)
            return rc

    def dissolve(self, inplace=False, **kwargs):
        """
        Merge consecutive ranges.
        """
        # Validate input
        if self.num_ranges == 1:
            if inplace:
                return
            else:
                return self.copy(deep=True)
        elif self.num_ranges == 0:
            raise ValueError("No ranges to be dissolved.")
        
        # Sort
        rc = self.sortranges(by=['begs','ends'], ascending=[True, False])
        
        nonconsecutive = ~rc.are_consecutive(all_=False)
        unique, iunique = np.unique(np.cumsum(nonconsecutive),
                                    return_index=True)
        
        begs = [rc.begs[0]]
        ends = []
        
        # Check for first range separation
        if not unique[0] == 0:
            ends.append(rc.ends[0])
            begs.append(rc.begs[1])
        
        # Get min and max bounds of consecutive ranges
        for i in range(1, len(unique)):
            j = iunique[i]
            ends.append(rc.ends[j])
            begs.append(rc.begs[j+1])
        
        # Add final end point
        ends.append(rc.ends[-1])
        
        if inplace:
            self._begs = np.asarray(begs)
            self._ends = np.asarray(ends)
            self.reset_centers(inplace=True)
            self.reset_keys()
            return
        else:
            return self.__class__(begs=begs, ends=ends, closed=self.closed)
        
    def simplify(self, inplace=False, **kwargs):
        """
        Sort and separate ranges, remove point ranges, and merge consecutive
        ranges from the results.
        """
        rc = self.separate(inplace=False, **kwargs)\
                 .drop_short(length=0, inplace=False, **kwargs)\
                 .dissolve(inplace=False, **kwargs)
        
        if inplace:
            self._begs = rc._begs
            self._ends = rc._ends
            self.reset_centers(inplace=True)
            self.reset_keys()
            return
        else:
            return rc
        
    def pixelate(self, length=1, reference=0, extend='right', 
                 return_keys=False, inplace=False, **kwargs):
        """
        Break down ranges and fit into a grid defined by the length and 
        reference parameters.

        Parameters
        ----------
        length : numerical, default 1
            The size of the grid and the max length to enforce on new ranges.
        reference : numerical, default 0
            The defined origin of the grid against which all ranges will be 
            compared when enforcing the length parameter.
        extend : {'right','left','both','neither}, default 'right'
            The direction(s) in which to extend each new range to align each 
            edge with the nearest value on the defined grid.
        return_keys : bool, default False
            Whether to return an array of keys which point to the original 
            ranges from which each new pixelated range was derived.
        inplace : bool, default False
            Whether to perform the operation in place, replacing the original 
            collection with the new, pixelated collection.
        """
        # Initialize new range begin and end points
        begs_all = np.array([])
        ends_all = np.array([])
        ref_keys = np.array([], dtype=int)
        
        # Compute new ranges
        for beg, end, key in zip(self.begs, self.ends, self.keys):
            
            # Calculate begin point delta
            beg_delta = (beg - reference) % length
            end_delta = (end - reference) % length
            
            # Validate delta computations
            beg_delta = 0 if np.isclose(beg_delta, length) else beg_delta
            end_delta = 0 if np.isclose(end_delta, length) else end_delta
            
            # Determine new begin point
            if beg_delta == 0:
                beg_new = beg
            else:
                if extend in ['left', 'both']:
                    beg_new = beg - beg_delta
                else:
                    beg_new = beg - beg_delta + length
                
            # Determine new end point
            if end_delta == 0:
                end_new = end
            else:
                if extend in ['right', 'both']:
                    end_new = end - end_delta + length
                else:
                    end_new = end - end_delta
                
            # Validate and append resulting range options
            if beg_new >= end_new:
                continue
            else:
                num_ranges = int(round((end_new-beg_new)/length))
                begs = np.array([beg_new+length*i \
                                 for i in range(num_ranges)])
                ends = np.array([beg_new+length*(i+1) \
                                 for i in range(num_ranges)])
                begs_all = np.concatenate((begs_all, begs))
                ends_all = np.concatenate((ends_all, ends))
                ref_keys = np.concatenate((ref_keys,
                                           np.full(num_ranges, key)))
                
        # Return results as requested
        if inplace:
            self._begs = begs_all
            self._ends = ends_all
            self.reset_centers(inplace=True)
            self.reset_keys(inplace=True)
        else:
            # Create newly pixelated range collection
            rc = self.__class__(begs=begs_all, ends=ends_all, centers=None, 
                                closed=self.closed, **kwargs)
            if return_keys:
                return rc, ref_keys
            else:
                return rc
            return rc
    
    def cut(self, beg=None, end=None, inplace=False):
        """
        Cut all ranges to a minimum begin point and a maximum end point to 
        enforce a domain of range values.

        Parameters
        ----------
        beg, end : scalar
            The minimum begin point and a maximum end point values to define 
            the domain of range values. If one is not pro
        """
        self._validate_monotonic()
        # Validate input
        beg = -np.inf if beg is None else beg
        end =  np.inf if end is None else end
        if end < beg:
            raise ValueError("End point must be greater than or equal to "
                "begin point.")
        
        # Eliminate non-intersecting ranges
        before = self.set_closed('both').is_before(loc=beg)
        behind = self.set_closed('both').is_behind(loc=end)
        
        # Clip ranges
        lefts  = np.max([self.begs, np.full(self.begs.shape, beg)], axis=0)
        rights = np.min([self.ends, np.full(self.ends.shape, end)], axis=0)
        lefts[behind]  = end
        rights[before] = beg
        
        if inplace:
            self.reset_centers(inplace=True)
            self._begs = lefts
            self._ends = rights
        else:
            rc = self.copy(deep=True)
            rc.reset_centers(inplace=True)
            rc._begs = lefts
            rc._ends = rights
            return rc

    def clip(self, length=1, anchor='begs', inplace=False):
        """
        Clip each range to a maximum length relative to an anchor of begin, 
        end center, or true center points, or relative to an input array of
        anchor values.

        Parameters
        ----------
        length : scalar
            Maximum length to clip each range to.
        anchor : np.ndarray or {begs, ends, centers, true_centers}
            Anchor points within each range to clip relative to.

            Options
            -------
            begs or left : Fix the left end of each range and clip to the right
            ends or right : Fix the right end of each range and clip to the 
                left
            centers : Fix the defined center of each range and clip equally to 
                the left and right
            true_centers or mid : Fix the true center of each range and clip 
                equally to the left and right

        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        self._validate_monotonic()
        # Validate input
        if not np.issubdtype(type(length), np.number):
            raise TypeError("Input clip length must be numeric.")

        # Compute clipped range bounds
        #  - Anchor by custom input anchor values
        if not isinstance(anchor, str):
            try:
                anchor = np.asarray(anchor, float).flatten()
            except:
                raise ValueError("Anchor data must be array-like of scalar "
                                 "values.")
            if not anchor.size == self.num_ranges:
                raise ValueError("Array of anchor values must have length "
                                 "equal to the number of ranges in the "
                                 "collection.")
            if not (anchor >= self.begs) & (anchor <= self.ends):
                raise ValueError("All anchor values must fall within the "
                                 "begin and end points of each respective "
                                 "range.")
            lefts  = np.max([self.begs, anchor-(length/2)], axis=0)
            rights = np.min([self.ends, anchor+(length/2)], axis=0)

        #  - Anchor by begin points
        elif anchor in ('begs', 'left'):
            lefts  = self.begs
            rights = np.min([self.ends, lefts+length], axis=0)

        #  - Anchor by end points
        elif anchor in ('ends', 'right'):
            rights = self.ends
            lefts  = np.max([self.begs, rights-length], axis=0)

        #  - Anchor by center points
        elif anchor in ('centers'):
            lefts  = np.max([self.begs, self.centers-(length/2)], axis=0)
            rights = np.min([self.ends, self.centers+(length/2)], axis=0)

        #  - Anchor by true center points
        elif anchor in ('true_centers', 'mid'):
            lefts  = np.max([self.begs, self.true_centers-(length/2)], axis=0)
            rights = np.min([self.ends, self.true_centers+(length/2)], axis=0)

        #  - Invalid anchor inputs
        else:
            raise ValueError("Anchor must be begs (left), ends (right), "
                             "centers, or true_centers (mid), or an array of "
                             "anchor values.")
        
        if inplace:
            self.reset_centers(inplace=True)
            self._begs = lefts
            self._ends = rights
        else:
            rc = self.copy(deep=True)
            rc.reset_centers(inplace=True)
            rc._begs = lefts
            rc._ends = rights
            return rc
        
    def compensate(self, beg=0.0, end=1.0, by=None, inplace=False, **kwargs):
        """
        (DRAFT)
        """
#        # Validate range
#        if end-beg < self.lengths.max():
#            raise ValueError("Compensation range must be greater than or \
#equal to the maximum length of the range collection.")
        
        # Validate selected reference point information
        if not by in [None, 'centers', 'begs', 'ends', 'true_centers']:
            raise ValueError("Separate 'by' must be either 'centers', \
'begs', 'ends', or 'true_centers' or None.")

        # Prepare sorted ranges for processing
        rc = self.copy(deep=True)
        if not by is None:
            rc.set_centers(centers=by, inplace=True)
        rc, inv = rc.sortranges(by=['left_lengths','right_lengths'],
                                ascending=[True, True],
                                inplace=False,
                                return_inverse=True)
        
        # Define compensation domain
        left  = beg + rc[0].left_lengths
        right = end - rc[-1].right_lengths
        
        # Compute default new range placements and shift
        place = np.linspace(left,right,rc.num_ranges)
        shift = (place - rc.centers)[inv]
        
        if inplace:
            self.shift(shift, inplace=True)
        else:
            rc = self.copy(deep=True)
            rc.shift(shift, inplace=True)
            return rc

    def shift(self, shift, inplace=False, **kwargs):
        """
        Shift all ranges in the collection by a single numeric value amount or 
        based on a provided 1D numpy array whose length is equal to the number 
        of ranges in the collection.

        Parameters
        ----------
        shift : scalar or np.ndarray of scalars
            Numeric distance to shift all ranges in the collection by. Can be 
            provided as a single numeric value or a 1D numpy array whose 
            length is equal to RangeCollection.num_ranges.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        
        Returns
        -------
        res : RangeCollection
            The resulting RangeCollection object with all ranges shifted by 
            the input amount.
        """
        # Validate input shift variable
        error = ValueError("Shift value must be provided as a single \
numeric value or a 1D array of numeric values with a number of elements \
equal to the number of ranges in the collection.")
        try:
            shift = np.asarray(shift).flatten()
        except:
            raise error
        if not shift.size in [1, self.num_ranges]:
            raise error
        
        # Apply shift
        if inplace:
            self._begs += shift
            self._ends += shift
            if self.center_type=='data':
                self._centers += shift
            return
        else:
            rc = self.copy(deep=True)
            rc.shift(shift=shift, inplace=True)
            return rc

    def extend(self, length, direction='right', inplace=False, **kwargs):
        """
        Extend all ranges in the collection by a single numeric value amount or 
        based on a provided 1D numpy array whose length is equal to the number 
        of ranges in the collection. Extension can be made to the left, right, 
        or both directions using the direction parameter.

        Parameters
        ----------
        length : scalar or np.ndarray of scalars
            Numeric distance to extend all ranges in the collection by. Can be 
            provided as a single numeric value or a 1D numpy array whose 
            length is equal to RangeCollection.num_ranges.
        direction : {'right', 'left', 'both'}, default 'right'
            The direction in which to extend existing ranges by the input 
            length parameter. If both, ranges will be extended by the whole  
            input length in both directions.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        
        Returns
        -------
        res : RangeCollection
            The resulting RangeCollection object with all ranges extended by 
            the input amount.
        """
        # Validate input shift variable
        error = ValueError("Extend value must be provided as a single "
                    "numeric value or a 1D array of numeric values with a "
                    "number of elements equal to the number of ranges in the "
                    "collection.")
        try:
            length = np.asarray(length).flatten()
        except:
            raise error
        if not length.size in [1, self.num_ranges]:
            raise error
        direction_ops = {'left','right','both'}
        if not direction in direction_ops:
            raise ValueError("Direction parameter must be one of "
                f"{direction_ops}")
        
        # Apply extension
        if inplace:
            if direction in ['left','both']:
                self._begs -= length
            if direction in ['right','both']:
                self._ends += length
            return
        else:
            rc = self.copy(deep=True)
            rc.extend(length=length, direction=direction, inplace=True)
            return rc

    # Address deprecation and naming
    # - name changed for consistency and due to improved scope
    is_intersecting = intersecting

    def fill(self, direction='both', bounds=None, reset_centers=True, 
             inplace=False, **kwargs):
        """
        """
        # Validate parameters
        direction_ops = {'right','left','both'}
        if not direction in direction_ops:
            raise ValueError(f"Fill direction must be one of {direction_ops}.")
        if not bounds is None:
            try:
                bound_left = bounds[0]
                bound_right = bounds[1]
            except:
                raise ValueError("If provided, fill bounds must be a two-value "
                                 "tuple of scalars indicating minimum and "
                                 "maximum values to extend ranges to.")
        else:
            bound_left = bound_right = None

        # Prepare sorted ranges for processing
        rc = self.copy(deep=True)
        rc, inv = rc.sortranges(
            by=['centers', 'lengths'],
            ascending=[True, False],
            inplace=False,
            return_inverse=True
        )
        # Compute extensions
        begs_r = rc.begs[1:]
        ends_l = rc.ends[:-1]
        gap_locs = begs_r > ends_l
        if direction in ['left']:
            begs_r[gap_locs] = ends_l[gap_locs]
        elif direction in ['right']:
            ends_l[gap_locs] = begs_r[gap_locs]
        elif direction in ['both']:
            begs_r[gap_locs] = \
                ends_l[gap_locs] = ((begs_r + ends_l) / 2)[gap_locs]
        
        # Compute bounds
        if not bound_left is None:
            bound_left = min(bound_left, rc.begs[0])
        else:
            bound_left = rc.begs[0]
        if not bound_right is None:
            bound_right = max(bound_right, rc.ends[-1])
        else:
            bound_right = rc.ends[-1]
        begs = np.append(bound_left, begs_r)
        ends = np.append(ends_l, bound_right)

        # Assign the new begin and end points to the processed ranges
        rc._begs = begs
        rc._ends = ends
        rc = rc[inv]
        if reset_centers:
            rc.reset_centers(inplace=True)

        if inplace:
            self._begs = rc._begs
            self._ends = rc._ends
            self.set_centers(rc.centers, inplace=True)
            return
        else:
            return rc

    def cluster(self, buffer=None, **kwargs):
        """
        Identify clusters of ranges which overlap, returning an array of 
        cluster indices. Returned indices will include a single value for each 
        range, associating it with a unique cluster whose ID reflects the 
        numerical index of the first range within the cluster. The process 
        allows for chaining, grouping ranges which share an overlapped range 
        but which do not themselves overlap.

        Parameters
        ----------
        buffer : scalar, optional
            A numerical buffer to be applied to ranges before checking for 
            overlaps.
        """
        # Buffer self if requested
        if not buffer is None:
            rc = self.extend(buffer, direction='both', inplace=False)
        else:
            rc = self
        # Initialize cluster indices data
        default_clusters = np.arange(rc.num_ranges)
        clusters = default_clusters.copy()
        # Identify intersecting ranges
        intersecting = rc.intersecting(
            rc.begs, rc.ends, validate=False, squeeze=False)
        # Cull duplicates
        intersecting[np.tril_indices(rc.num_ranges, -1)] = False
        # Identify intersecting clusters
        for i, adjacent in enumerate(intersecting):
            # Find points near the indexed point
            adjacent_idx = clusters[adjacent]
            # Find the lowest cluster number among points
            min_idx  = adjacent_idx.min()
            # Find relatives
            clusters_sub = clusters[i:]
            select = np.any(
                clusters_sub.reshape(-1,1) == \
                adjacent_idx.reshape(1,-1), axis=1)
            # Assign this cluster number to the points
            clusters[i:] = np.where(select, min_idx, clusters_sub)
        
        # Return clusters
        return clusters

    def distribute(
        self,
        data,
        values=None,
        blur_size=0,
        blur_style='linear',
        length_normalize=True,
        **kwargs
    ):
        """
        Intersect and distribute events over the range collection, scaling 
        their values relative to their indexed distance from their intersecting 
        range location.
        
        Parameters
        ----------
        data : RangeCollection or 1-2D array-like
            The ranges of the locations of target events being analyzed. All 
            ranges should fall within the defined bounds (if provided) to be 
            considered in the analysis.
        values : numeric or 1d array-like, optional
            The value(s) associated with each event being analyzed. If not 
            provided, all values will default to be 1.
        blur_size : int, default 0
            The number of pixels to blur events across based on the blur style.
        blur_style : str or callable, default 'linear'
            The scaling function to be called at each blurring step to scale 
            original values. If a callable is provided, it must accept a single 
            integer input for the zero-indexed pixel number, returning a single 
            float scaling value. Predefined blurring functions can be called 
            using the following labels:
            
            Options
            -------
            linear : linearly scale down values from the original value to zero 
                at the first index outside the blurred pixel range
            norm_static : Tk
            norm_scale : Tk
            none : do not scale down original values

        length_normalize : bool, default True
            Normalize the intersection scores by the length of the range to 
            account for differing range lengths.
                
        Created:  2022-10-04
        """
        ##################
        # VALIDATE INPUT #
        ##################

        # Check for valid data dimensions; if no valid ranges to operate over, 
        # return an appropriately dimensioned zero array
        if (data.num_ranges == 0) or (self.num_ranges == 0):
            return np.zeros((self.num_ranges, data.num_ranges))
        
        # Validate input range collection
        if not isinstance(data, RangeCollection):
            try:
                data = RangeCollection.from_array(data, sort=False)
            except:
                raise ValueError(
                    "Input data must be range collection type or compatible "
                    "with RangeCollection.from_array() class method.")
        
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
                "Input blur_style must be callable or label of valid "
                "predefined scaling function.")
        
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
                        "Could not convert input values to np.ndarray of "
                        "dtype float.")
                # Check for appropriate size
                if not len(values) == data.num_ranges:
                    raise ValueError(
                        "Number of event values must be equal to the number "
                        f"of events provided. Provided: {data.num_ranges} "
                        f"events, {len(values)} values.")
        
        ####################
        # PERFORM ANALYSIS #
        ####################
        
        # Intersect events data with analysis range collection to determine 
        # initial intersection scores
        intx = self.intersecting(
            beg=data.begs, end=data.ends, squeeze=False) * 1 # coerce integer
        
        # Blur events data by stepping through the blur range and applying the 
        # blur function to scale the initial pixel scores and add them
        scale = blur_function(0)
        result = intx * scale
        for step in range(1, blur_size + 1):
            # Create and blur buffered result data
            scale = blur_function(step)
            buff = np.pad(intx, ((step, step), (0, 0)), mode='constant') * scale
            # Apply buffered result data
            forw = buff[:-step * 2, :]
            back = buff[step * 2:, :]
            result += forw + back
            
        # Normalize result data scores by target range lengths
        if length_normalize:
            term = self.lengths.reshape(-1,1)
            result = np.multiply(result, term)
        
        # Normalize data scores to a single unit per record
        denom = result.sum(axis=0)
        result = np.divide(
            result, denom, out=np.zeros_like(result), where=denom!=0)
        
        # Apply values if used
        if not values is None:
            result = np.multiply(result, values)
        
        # Return results
        return result