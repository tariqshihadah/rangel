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
numpy, matplotlib, copy, warnings

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
        Whether intervals are closed on the left-side, right-side, both or 
        neither.

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
            
    sort : boolean, default True
        Whether to automatically sort the ranges by begin points or to leave
        them in the order they are input.
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
    _ops_centers = {'true_centers','begs','ends'}
    
    def __init__(self, begs=None, ends=None, centers=None, 
                 closed='right', sort=True, copy=True,
                 force_monotonic=True, keys=None):

        # Establish closed parameters
        self.set_closed(closed, inplace=True)
        
        # Process ranges
        if begs is None:
            begs = []
        if ends is None:
            ends = begs
        begs = np.array(begs, dtype=float, copy=copy)
        ends = np.array(ends, dtype=float, copy=copy)
        self._begs = begs.copy()
        self._ends = ends.copy()
        self.set_centers(centers, inplace=True, copy=copy)
    
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
        # Determine numbers of left and right digits to display
        ld = len(str(int(self.arr.max())))
        rd = 3
        # Determine number of records to show
        if self.num_ranges <= 10:
            it = zip(self.begs, self.ends, self._mod_locs)
        else:
            it = zip(np.append(self.begs[:5], self.begs[-5:]), 
                     np.append(self.ends[:5], self.ends[-5:]),
                     np.append(self._mod_locs[:5], self._mod_locs[-5:]))
        # Create formatter
        records = []
        closed = self.closed
        for beg, end, mod in zip(self.begs, self.ends, self._mod_locs):
            # Determine left and right brackets
            lb = '[' if (closed in ['left','left_mod','both']) or mod else '('
            rb = ']' if (closed in ['right','right_mod','both']) or mod else ')'
            # Format string
            record = f'{lb}{beg: >{ld+rd+1}.{rd}f}, {end: >{ld+rd+1}.{rd}f}{rb}'
            records.append(record)
        # Address shown records
        if self.num_ranges > 10:
            spacer = '~' * (ld*2 + rd*2 + 6)
            records = records[:5] + [spacer] + records[-5:]
        text = '\n'.join(records) + '\n' + str(self)
        return text

        # Display a portion of the range
        if self.num_ranges <= 10:
            text = str(self.arr.T)
        else:
            rows = str(self.arr.T).split('\n')
            tbc = ' .' * (len(rows[-1]) // 2)
            text = '\n'.join(rows[:5]+[tbc] + rows[-5:])
        return text + '\n' + self.__str__()
    
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
        return np.stack((self.begs, self.ends, self.centers), axis=0)\
            .reshape((3,-1))
    
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
    def monotonic(self):
        return self._monotonic
    
    @classmethod
    def _breaks_to_ranges(cls, breaks):
        """
        Take an array of break points or a list of arrays of break points and 
        convert them to begin and end points to define unique ranges.
        """
        # Array of numerical break points
        if np.issubdtype(type(breaks[0]), np.number):
            breaks = np.asarray(breaks, dtype=float)
            begs = breaks[:-1]
            ends = breaks[1:]
            
        # List of arrays of numerical break points
        else:
            # Initialize result arrays
            begs = np.asarray([], dtype=float)
            ends = np.asarray([], dtype=float)
            for breaks_i in breaks:
                # Array of numerical break points
                if np.issubdtype(type(breaks_i[0]), np.number):
                    breaks_i = np.asarray(breaks_i, dtype=float)
                    begs = np.concatenate((begs, breaks_i[:-1]), axis=None)
                    ends = np.concatenate((ends, breaks_i[1:]), axis=None)
                else:
                    raise TypeError("Multiple sets of break points must be \
provided as a list of arrays or array-like.")
        
        # Return results as a tuple
        return begs, ends        
    
    def _select_self(self, index):
        """
        Use a static index to set new object attributes. Equivalent to 
        __getitem__ operating inplace.
        """
        # Flatten to avoid dimension loss
        index = np.asarray(index).flatten()
        # Select from data arrays
        self._begs = self._begs[index]
        self._ends = self._ends[index]
        self._keys = self._keys[index]
        # Determine center
        if self.center_type=='data':
            self._centers = self._centers[index]
        
    def _validate_centers(self, centers=None, copy=False):
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
                raise ValueError(f"Centers as string must be one of \
{self._ops_centers}.")
            return centers

        # Validate array input
        # Enforce array class
        try:
            centers = np.array(centers, copy=copy)
        except:
            raise ValueError("Cannot convert centers data to array.")

        # Test for array length
        if len(centers) != self.num_ranges:
            raise ValueError("Centers information must have an equal \
number of elements to the number of ranges in the collection.")
        
        # Test for numeric dtype
        if not np.issubdtype(centers.dtype, np.number):
            raise TypeError("Input center values must be numeric.")

        # Test for NaN
        if np.isnan(centers).sum() > 0:
            raise ValueError("Input center values cannot contain NaN.")

        # Test the centers against the bounds of the ranges
        if not np.all((centers >= self.begs) & (centers <= self.ends)):
            raise ValueError("Range centers must fall within the begin and \
end points of the input ranges.")

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
            raise ValueError("Invalid centers input values. Must be provided \
as an array-like of scalar values.")
        try:
            lengths = np.asarray(lengths, dtype=float).flatten()
        except:
            raise ValueError("Invalid lengths input values. Must be provided \
as a single scalar value or an array-like of scalar values.")
        if lengths.size == 1:
            lengths = np.tile(lengths, centers.size)
        elif lengths.size != centers.size:
            raise ValueError(f"If provided as an array, lengths \
(size={lengths.size:,.0f}) must be equal in size to centers array \
(size={centers.size:,.0f}).")

        # Define range begin and end points and create collection
        delta = lengths / 2
        begs = centers - delta
        ends = centers + delta
        rc = cls(begs=begs, ends=ends, centers=centers, **kwargs)
        return rc

    @classmethod
    def from_array(cls, arr, **kwargs):
        """
        Create a range collection from an array defining the begin and end
        bounds of each range, as well as (optionally) the centers.
        
        Parameters
        ----------
        arr : array-like
            A 2D or 3D array-like of numerical values representing the begin 
            and end bounds of each range, and (optionally) the center values.
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        
        Returns
        -------
        rc : RangeCollection
            RangeCollection object instance created based on the provided 
            constructor parameters.
        """
        if np.shape(arr)[0] == 2:
            begs = arr[0]
            ends = arr[1]
            centers = None
        elif np.shape(arr)[0] == 3:
            begs = arr[0]
            ends = arr[1]
            centers = arr[2]
        else:
            raise ValueError("Input array must have shape of (2,x) or (3,x)")
        rc = cls(begs=begs, ends=ends, centers=centers, **kwargs)
        return rc
            
    @classmethod
    def from_tuples(cls, tuples, **kwargs):
        """
        Create a range collection from a list or array of tuples defining the
        begin and end bound of each range.
        
        Parameters
        ----------
        tuples : list-like
            A list or 1D array of tuples of numerical values representing the 
            begin and end points of each range within the collection.
        **kwargs
            Keyword arguments used for initialization of the RangeCollection 
            class instance.
        
        Returns
        -------
        rc : RangeCollection
            RangeCollection object instance created based on the provided 
            constructor parameters.
        """
        # Parse the input tuples
        arr = np.array(tuples, dtype=float)
        begs = arr[:,0]
        ends = arr[:,1]
        rc = cls(begs=begs, ends=ends, **kwargs)
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
    def from_steps(cls, beg, end, length=1, steps=10, fill=None, **kwargs):
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
        steps : int, default 10
            A number of steps per range length. The resulting step length will 
            be equal to length / steps. For non-overlapped ranges, use a steps 
            value of 1.
        fill : {'none','cut','left','right'}, default 'cut'
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
        fill_options = {'none','cut','left','right'}
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
        else:
            begs = np.append(begs,last_beg+step)
            ends = np.append(ends,end)
        
        # Generate RangeCollection
        rc = cls(begs=begs, ends=ends, **kwargs)
        return rc
    
    @classmethod
    def concatenate(cls, objs=[], reset_centers=True, sort=True, **kwargs):
        """
        Combine multiple RangeCollection instances into a single instance.
        
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
                raise TypeError("Input objects must be list-like of \
RangeCollection instances.")
        
        # Determine range parameters
        arr = np.concatenate([obj.arr for obj in objs], axis=1)
        keys = np.concatenate([obj.keys for obj in objs])
        # Combine collections
        rc = cls.from_array(arr, keys=keys, sort=sort, **kwargs)
        # Modify new collection
        if reset_centers:
            rc.reset_centers(inplace=True)
        return rc
    
    @classmethod
    def random_int(cls, beg_bounds=(0,10), end_bounds=(0,10), size=10,
                   random_center=False, **kwargs):
        """
        Create a randomly-generated collection of integer ranges based on the 
        defined parameters.

        Parameters
        ----------
        beg_bounds, end_bounds : tuple, default (0,10)
            Two-element tuples of min and max begin and end values to use when 
            generating the collection.
        size : int, default 10
            The number of ranges to be generated for the collection.
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
    def random_float(cls, beg_bounds=(0,10), end_bounds=(0,10), size=10,
                     random_center=False, **kwargs):
        """
        Create a randomly-generated collection of float ranges based on the 
        defined parameters.

        Parameters
        ----------
        beg_bounds, end_bounds : tuple, default (0,10)
            Two-element tuples of min and max begin and end values to use when 
            generating the collection.
        size : int, default 10
            The number of ranges to be generated for the collection.
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
    def random_consecutive(cls, beg=0, end=10, size=10, **kwargs):
        """
        Create a randomly-generated collection of consecutive float ranges 
        based on the defined parameters.

        Parameters
        ----------
        beg, end : scalar, default 0, 10
            Begin and end points to define the collection of consecutive ranges 
            being generated.
        size : int, default 10
            The number of ranges to be generated for the collection.
        """
        # Define breaks
        breaks = np.random.random(size=size)
        breaks = breaks.cumsum() / breaks.sum() * (end-beg) + beg
        breaks = np.concatenate([[beg], breaks])
        
        # Generate and return the range collection
        return cls.from_breaks(breaks=breaks, **kwargs)
    
    def reset_keys(self, keys=None, inplace=False):
        """
        Reset key values to enumerate ranges within the collection.
        """
        if keys is None:
            keys = np.arange(0,self.num_ranges,1,dtype=int)
        else:
            if len(keys) != self.num_ranges:
                raise ValueError("Keys array must have a length equal to the \
number of ranges in the collection.")
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
            Whether intervals are closed on the left-side, right-side, both or 
            neither.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        if closed in self._ops_closed:
            if inplace:
                self._closed = closed
            else:
                rc = self.copy()
                rc._closed = closed
                return rc
        else:
            raise ValueError(f"Closed parameter must be one of \
{self._ops_closed}.")
    
    def set_centers(self, centers=None, inplace=False, copy=True):
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
        """
        # Validate input
        centers = self._validate_centers(centers=centers, copy=copy)
        
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
    
    def plot(self, ax=None, figsize=None, centers=True, one_line=False, 
             *args, **kwargs):
        """
        Create a simple plot of the range collection using matplotlib's pyplot 
        interface. All provided **kwargs are passed directly to the ax.plot() 
        axis method call.
        """
        # Create figure and subplot
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
        for i, rng in enumerate(self.iterranges()):
            ax.plot([rng.begs, rng.ends], ([0, 0] if one_line else [-i, -i]),
                    **kwargs)
            if centers:
                ax.scatter(rng.centers, -i, **kwargs)
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
        
    def locate(self, loc, choose='first', closed=None, snap=None, 
               return_dist=False, *args, **kwargs):
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
        closed : str {None, 'left', 'right', 'both', 'neither'}, default None
            Whether intervals are closed on the left side, right side, both or 
            neither. If None, default to the collection's current setting.
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
        index = np.where(self.is_intersecting(beg=loc, closed=closed))[0]
        
        # Check for at least one range
        if len(index) == 0:
            if snap is None:
                raise ValueError(f"Location ({loc}) does not fall within any \
valid ranges.")
            
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
                raise ValueError("Choose parameter must be 'first', 'last', \
or 'all'.")
    
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
    
    def is_before(self, loc=None, closed=None, **kwargs):
        """
        Get boolean mask for ranges which fall entirely before the given range 
        location.
        """
        self._validate_monotonic()
        
        # Validate input points
        if loc is None:
            raise ValueError("No input point provided.")
        
        # Validate closed
        if closed is None:
            closed = self.closed
        
        # Test for intersecting
        if closed in ['right', 'both']:
            t1 = self.ends  < loc
        elif closed in ['left', 'neither']:
            t1 = self.ends <= loc
        
        return t1
    
    def is_behind(self, loc=None, closed=None, **kwargs):
        """
        Get boolean mask for ranges which fall entirely behind the given range 
        location.
        """
        self._validate_monotonic()
        
        # Validate input points
        if loc is None:
            raise ValueError("No input point provided.")
        
        # Validate closed
        if closed is None:
            closed = self.closed
        
        # Test for intersecting
        if closed in ['left', 'both']:
            t1 = self.begs  > loc
        elif closed in ['right', 'neither']:
            t1 = self.ends >= loc
        
        return t1
    
    def intersecting(self, beg=None, end=None, closed=None, validate=True, 
                     **kwargs):
        """
        Get boolean mask for ranges which intersect the given range values. If 
        multiple ranges given, return a mask array with a second dimension 
        equal to the number of ranges provided.

        Parameters
        ----------
        beg, end : numerical or array-like
            The begin and end locations of the range or ranges to be tested. If 
            a single range is to be tested, provide a numeric value. If 
            multiple, provide an array-like with a single begin and end value 
            for each range. If no end parameter provided, point locations will 
            be assumed and end will be set equal to beg.
        closed : str {'left', 'left_mod', 'right', 'right_mod', 'both', 
                'neither'}, optional
            Whether intervals are closed on the left-side, right-side, both or 
            neither. If not provided, default to collection property.
        validate : boolean, default True
            Whether to validate the input begin and end location information. 
            Unless externally validated, always use True.

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
        if validate:
            beg, end = self._validate_beg_end(beg, end)
        
        # Validate closed
        if closed is None:
            closed = self.closed
        
        # Test for intersecting
        rbegs = self.begs.reshape(-1,1)
        rends = self.ends.reshape(-1,1)
        if closed in ['left','left_mod','both']:
            t1 = np.less_equal(rbegs, end.reshape(1,-1))
        elif closed in ['right','right_mod','neither']:
            t1 = np.less(rbegs, end.reshape(1,-1))
        if closed in ['right','right_mod','both']:
            t2 = np.greater_equal(rends, beg.reshape(1,-1))
        elif closed in ['left','left_mod','neither']:
            t2 = np.greater(rends, beg.reshape(1,-1))

        # Modify test for specific closed cases
        if closed in ['left_mod']:
            # Perform tests on modified edge locations
            mod_locs = self._mod_locs
            t2[mod_locs,:] = \
                np.greater_equal(rends[mod_locs,:], beg.reshape(1,-1))
        elif closed in ['right_mod']:
            # Perform tests on modified edge locations
            mod_locs = self._mod_locs
            t1[mod_locs,:] = \
                np.less_equal(rbegs[mod_locs,:], end.reshape(1,-1))
        
        # Combine test results
        res = t1 & t2
        if res.shape[1] == 1:
            res = res.flatten()

        # Return final test results
        return res

    @property
    def _mod_locs(self):
        """
        Get indexes of ranges with modified edges. Only applicable when 
        self.closed in ['left_mod','right_mod'].
        """
        # Modify test for specific closed cases
        if self.closed in ['left_mod']:
            # Identify ends of group ranges which will be modified
            mod_locs = self.are_overlapping(all_=False, when_one=np.array([], dtype=bool), enforce_edges=True)
            mod_locs = np.append(~mod_locs, True)
        elif self.closed in ['right_mod']:
            # Identify ends of group ranges which will be modified
            mod_locs = self.are_overlapping(all_=False, when_one=np.array([], dtype=bool), enforce_edges=True)
            mod_locs = np.append(True, ~mod_locs)
        else:
            mod_locs = np.zeros(self.begs.shape, dtype=bool)
        return mod_locs

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
            return when_one
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
    
    def get_before(self, loc=None, closed=None):
        """
        Return only those ranges which fall entirely before the given range 
        location.

        Parameters
        ----------
        loc : float
            Range location of the overlaid point.
        """
        select = self.is_before(loc=loc, closed=closed)
        return self[select]
    
    def get_behind(self, loc=None, closed=None):
        """
        Return only those ranges which fall entirely behind the given range 
        location.

        Parameters
        ----------
        loc : float
            Range location of the overlaid point.
        """
        select = self.is_behind(loc=loc, closed=closed)
        return self[select]
    
    def get_inside(self, beg=None, end=None):
        """
        Return only those ranges which fall entirely within the given range 
        values.

        Parameters
        ----------
        beg : float
            Begin location of the overlaid range.
        end : float
            End location of the overlaid range.
        """
        select = self.is_inside(beg, end)
        return self[select]
    
    def get_intersecting(self, beg=None, end=None):
        """
        Return only those ranges which intersect the given range values.

        Parameters
        ----------
        beg : float
            Begin location of the overlaid range.
        end : float
            End location of the overlaid range.
        """
        select = self.is_intersecting(beg, end)
        return self[select]
    
    def get_nonintersecting(self, beg=None, end=None):
        """
        Return only those ranges which do not intersect the given range 
        values.

        Parameters
        ----------
        beg : float
            Begin location of the overlaid range.
        end : float
            End location of the overlaid range.
        """
        select = ~self.is_intersecting(beg, end)
        return self[select]
    
    def overlay_old(self, beg=None, end=None, normalize=True, by_sum=True, 
                squeeze=True, **kwargs):
        """
        Compute overlap of the input bounds with respect to collection ranges.
        
        Parameters
        ----------
        beg, end : scalar or array of scalar
            Begin and end locations of the overlaid range(s).
        normalize : boolean, default True
            Whether overlapping lengths should be normalized by the overlaid 
            range length to give a proportional result.
        by_sum : boolean, default True
            Whether to normalize overlapping lengths by the sum total length 
            covered by valid ranges. If True, overlay lengths will be 
            normalized by the sum of all overlapping lengths. If False, overlay 
            lengths will be normalized by the length of the overlaid range. 
            This parameter is only considered when normalize==True.
        squeeze : boolean, default True
            Whether to reduce the dimensions of the output array to 1D if only 
            a single begin/end location was provided. If False, output array 
            will always be 2D.

        NOTE: If no errors created, remove after 6/21/2020
        """
        self._validate_monotonic()
        # Validate input events
        if beg is None:
            raise ValueError("No input range(s) provided.")
        elif end is None:
            end = beg
        try:
            beg = np.asarray(beg, dtype=float)
            end = np.asarray(end, dtype=float)
        except ValueError:
            raise ValueError("Input begin and end locations must be "
                "provided as single numerical values or array-likes of "
                "numerical values.")
        # Validate parameters
        if 'minus_gaps' in kwargs:
            warnings.warn("Use of 'minus_gaps' is deprecated, instead use "
                "synonymous 'by_sum'", DeprecationWarning)
            by_sum = kwargs['minus_gaps']
        
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
            if by_sum:
                denom = overlap.sum(axis=0)
            else:
                denom = end-beg
            # Tile denominator, address zero division, and apply to normalize
            denom = np.tile(denom, (self.num_ranges,1))
            denom = np.where(denom==0, np.inf, denom)
            overlap = overlap / denom
        
        # Shape array based on input and return
        if squeeze and beg.size==1:
            return overlap[:,0]
        else:
            return overlap
    
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
    
    def intersect(self, other=None):
        """
        NOT YET IMPLEMENTED

        Intersect another range collection with this one, producing a single 
        range collection with no overlaps. If no other range collection is 
        provided, the collection will intersect with itself. This may be 
        useful for collections with some overlapping ranges.
        
        Parameters
        ----------
        other : RangeCollection, optional
            Another range collection to intersect with the current instance. 
            If none is provided, the current range collection will intersect 
            with itself.
        """
        raise Exception("Not yet implemented")
    
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
            self.arr, axis=1, 
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
        
        Parameters
        ----------
        closed : boolean, default False
            Whether to define a range as inside the comparison range
            even if at least one of its edges coincides with the 
            comparison range's edge.
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
        before = self.is_before(loc=beg, closed='both')
        behind = self.is_behind(loc=end, closed='both')
        
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
                raise ValueError("Anchor data must be array-like of scalar \
values.")
            if not anchor.size == self.num_ranges:
                raise ValueError("Array of anchor values must have length \
equal to the number of ranges in the collection.")
            if not (anchor >= self.begs) & (anchor <= self.ends):
                raise ValueError("All anchor values must fall within the \
begin and end points of each respective range.")
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
            raise ValueError("Anchor must be begs (left), ends (right), \
centers, or true_centers (mid), or an array of anchor values.")
        
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

