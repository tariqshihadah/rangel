from __future__ import annotations
import numpy as np
import copy, hashlib

# Import helper modules
import utility, relate, modify


class Rangel:
    """
    """

    # Class standard attributes
    _class_options = dict(
        display_max = 10,
        default_closed = 'right',
        keys_all={'index', 'groups', 'locs', 'begs', 'ends', 'centers'},
        anchors_all={'locs', 'begs', 'ends', 'centers'},
        anchors_locs={'begs', 'ends', 'centers'},
        closed = {'left','left_mod','right','right_mod','both','neither'},
        closed_base = {'left','right','both','neither'},
    )

    def __init__(
            self, 
            index=None, 
            groups=None, 
            locs=None, 
            begs=None, 
            ends=None, 
            closed=None, 
            dtype=float, 
            copy=None,
            force_monotonic=True,
        ):
        # Validate inputs
        self._validate_data(index, groups, locs, begs, ends, dtype=dtype, copy=copy)
        self.set_closed(closed, inplace=True)
        self._dtype = dtype
        # Prepare data
        if force_monotonic and self.is_linear:
            self.set_monotonic(inplace=True)

    def from_similar(self, index=None, groups=None, locs=None, begs=None, ends=None, **kwargs):
        """
        Create a new instance of the class with similar properties to the 
        current instance.
        """
        # Populate kwargs
        kwargs = {
            'closed': self.closed,
            'dtype': self._dtype,
            **kwargs
        }
        # Create new instance
        return self.__class__(
            index=index, groups=groups, locs=locs, begs=begs, ends=ends, **kwargs)

    def __str__(self):
        return utility._stringify_instance(self)
    
    def __repr__(self):
        return utility._represent_records(self)
    
    def __getitem__(self, index):
        return self.select_index(index, ignore=False, inplace=False)

    @property
    def index(self):
        """
        Event index.
        """
        return self._index

    @property
    def groups(self):
        """
        Event reference groups.
        """
        return self._groups
    
    @property
    def groups_hashed(self):
        """
        Event reference groups hashed.
        """
        if self.is_grouped:
            return np.array([hashlib.sha256(x.encode()).hexdigest() for x in self.groups])
        else:
            return None

    @property
    def locs(self):
        """
        Event reference positions.
        """
        return self._locs
    
    @property
    def begs(self):
        """
        Event begin positions.
        """
        return self._begs
    
    @property
    def ends(self):
        """
        Event end positions.
        """
        return self._ends
    
    @property
    def lengths(self):
        """
        Event lengths. If the events are points, lengths are zero.
        """
        if self.is_point:
            return None
        else:
            return self._ends - self._begs
        
    @property
    def centers(self):
        """
        Event centers. If the events are points, centers are the locations.
        """
        if self.is_point:
            return self._locs
        else:
            return (self._begs + self._ends) / 2
        
    @property
    def num_events(self):
        """
        Number of events.
        """
        if self.is_located:
            return len(self._locs)
        else:
            return len(self._begs)
        
    @property
    def closed(self):
        """
        Whether the ranges are closed on left, right, both, or neither side.
        """
        return self._closed
    
    @property
    def closed_base(self):
        """
        Base closed parameter without the 'mod' suffix.
        """
        return self._closed.replace('_mod','')
    
    @property
    def closed_mod(self):
        """
        Whether the closed parameter has modified edges.
        """
        return self._closed in ['left_mod','right_mod']
    
    @property
    def arr(self):
        if self.is_point:
            return self._locs.reshape(-1, 1)
        elif self.is_linear and not self.is_located:
            return np.stack((self.begs, self.ends), axis=1)
        elif self.is_linear and self.is_located:
            return np.stack((self.begs, self.ends, self.locs), axis=1)

    @property
    def is_linear(self):
        """
        Whether the events are linear.
        """
        # If begs and ends are defined, the events are linear.
        return self._begs is not None and self._ends is not None
    
    @property
    def is_point(self):
        """
        Whether the events are points.
        """
        # If begs and ends are not defined, the events are points.
        return self._begs is None and self._ends is None
    
    @property
    def is_located(self):
        """
        Whether the events are located.
        """
        # If locs are defined, the events are located.
        return self._locs is not None
    
    @property
    def is_grouped(self):
        """
        Whether the events are grouped.
        """
        # If groups are defined, the events are grouped.
        return self._groups is not None
    
    @property
    def is_monotonic(self):
        """
        Whether the events are all monotonic, increasing from begin to end position. 
        If the events are points, they are always monotonic.
        """
        # If the events are linear, check if they are monotonic.
        if self.is_linear:
            return np.all(self._begs <= self._ends)
        # If the events are points, they are always monotonic.
        else:
            return True
        
    @property
    def is_empty(self):
        """
        Whether the collection is empty.
        """
        return self.num_events == 0
        
    @property
    def modified_edges(self):
        """
        Get indexes of ranges with modified edges. Only applicable when 
        self.closed in {'left_mod','right_mod'}.
        """
        # Check for event type
        if self.is_point:
            edges = np.zeros(self.locs.shape, dtype=bool)
        else:
            # Require minimum ranges
            if self.num_events == 0:
                edges = np.zeros(self.begs.shape, dtype=bool)
            else:
                # Modify test for specific closed cases
                when_one = np.array([], dtype=bool)
                if self.closed in ['left_mod']:
                    # Identify ends of group ranges which will be modified
                    edges = self.next_overlapping(
                        all_=False, when_one=when_one, enforce_edges=True)
                    edges = np.append(~edges, True)
                elif self.closed in ['right_mod']:
                    # Identify ends of group ranges which will be modified
                    edges = self.next_overlapping(
                        all_=False, when_one=when_one, enforce_edges=True)
                    edges = np.append(True, ~edges)
                else:
                    edges = np.zeros(self.begs.shape, dtype=bool)
        return edges

    @property
    def unique_groups(self):
        """
        Get unique group values.
        """
        if self.is_grouped:
            return np.unique(self.groups)
        else:
            return None

    def _validate_index(self, index):
        """
        Validate input index as a 1D scalar np.array.
        """
        if index is None:
            # Create a generic zero-based integer index
            index = np.arange(self.num_events, dtype=int)
        else:
            index = utility._prepare_data_array(index, 'index')
            # Check that all indices are unique
            if len(np.unique(index)) < len(index):
                raise ValueError(
                    "All input indices must be unique.")
        return index
    
    def _validate_groups(self, groups):
        """
        Validate input groups as a 1D scalar np.array.
        """
        if groups is None:
            pass
        else:
            groups = utility._prepare_data_array(groups, 'groups')
        return groups
    
    def _validate_data(self, index, groups, locs, begs, ends, dtype=None, copy=None):
        """
        Validate input data based on the requirements of the class.
        """
        # Check possible valid cases of data input combinations
        data_input_case = (locs is not None, begs is not None, ends is not None)
        data_arrays = {}

        # - Located point events
        if data_input_case == (True, False, False):
            # Check that locs are not passed as an anchor reference
            if isinstance(locs, str):
                raise ValueError(
                    "For located point events, `locs` must be a 1D scalar array-like object."
                )
            # Convert locs to a numpy array
            locs = utility._prepare_data_array(locs, 'locs')
            data_arrays['locs'] = locs

        # - Located linear events
        elif data_input_case == (True, True, True):
            # If locs are passed as an anchor reference, validate
            if isinstance(locs, str):
                if not locs in self._class_options['anchors_locs']:
                    raise ValueError(
                        f"Invalid anchor reference for `locs`. Must be one of: {self._class_options['anchors_locs']}."
                    )
            # Convert data to numpy arrays
            else:
                locs = utility._prepare_data_array(locs, 'locs', dtype=dtype, copy=copy)
                data_arrays['locs'] = locs
            begs = utility._prepare_data_array(begs, 'begs', dtype=dtype, copy=copy)
            ends = utility._prepare_data_array(ends, 'ends', dtype=dtype, copy=copy)
            data_arrays['begs'] = begs; data_arrays['ends'] = ends

        # - Unlocated linear events
        elif data_input_case == (False, True, True):
            begs = utility._prepare_data_array(begs, 'begs', dtype=dtype, copy=copy)
            ends = utility._prepare_data_array(ends, 'ends', dtype=dtype, copy=copy)
            data_arrays['begs'] = begs; data_arrays['ends'] = ends
        
        # - Invalid input data case
        else:
            raise ValueError(
                "Invalid input data. Must provide either `locs`, `begs` and `ends`, or both. "
                f"Received: locs={locs is not None}, begs={begs is not None}, ends={ends is not None}."
            )

        # Validate index and groups
        if not index is None:
            index = self._validate_index(index)
            data_arrays['index'] = index
        if not groups is None:
            groups = self._validate_groups(groups)
            data_arrays['groups'] = groups

        # Validate equal lengths of data arrays
        data_array_lengths = {k: len(v) for k, v in data_arrays.items()}
        if len(set(data_array_lengths.values())) > 1:
            raise ValueError(
                "Input data arrays must have the same length. "
                f"Data array lengths: {data_array_lengths}."
            )
        
        # Assign validated data to class attributes
        self._groups = groups
        self._locs = locs
        self._begs = begs
        self._ends = ends
        if not index is None:
            self._index = index
        else:
            self.reset_index(inplace=True)
    
    def copy(self, deep=False):
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def reset_index(self, inplace=False):
        """
        Reset the index to a generic 0-based index.
        """
        # Create new index
        index = np.arange(self.num_events, dtype=int)
        # Apply changes
        rc = self if inplace else self.copy()
        rc._index = index
        return None if inplace else rc
    
    def select_index(self, index, ignore=False, inplace=False):
        """
        Select events by index or slice. Use ignore=True to use a generic 
        0-based index, ignoring the current index values.
        """
        # Validate input index
        if not isinstance(index, slice):
            # Enforce array type and check dimension
            try:
                index = np.asarray(index)
                assert index.ndim == 1
            except:
                raise ValueError(
                    "Input index must be a 1D array-like object or a slice object.")
            # Address the index selection
            if not ignore:
                sorter = np.argsort(self._index)
                index = np.searchsorted(self._index[sorter], index)

        # Apply selection
        rc = self if inplace else self.copy()
        try:
            rc._index = self._index[index]
            rc._groups = self._groups[index] if self._groups is not None else None
            rc._locs = self._locs[index] if self._locs is not None else None
            rc._begs = self._begs[index] if self._begs is not None else None
            rc._ends = self._ends[index] if self._ends is not None else None
        except:
            raise ValueError(
                "Invalid index selection. Check that the index values are within bounds.")
        return None if inplace else rc

    def select_group(self, group, ungroup=None, inplace=False):
        """
        Select events by group.

        Parameters
        ----------
        group : label
            The label of the group to select or array-like of the same.
        ungroup : bool, default None
            Whether to ungroup the selection, returning the selected events 
            without their group labels. If None and a single group is selected,
            the result will be ungrouped otherwise the group labels will be
            retained.
        """
        # Validate input group
        if not self.is_grouped:
            raise ValueError("No groups in collection.")
        if isinstance(group, (list, np.ndarray)):
            # Multiple group selection
            select_multiple = True
            ungroup = False if ungroup is None else ungroup
            if not all([x in self.unique_groups for x in group]):
                raise KeyError("Input groups not found in collection.")
        else:
            # Single group selection
            select_multiple = False
            ungroup = True if ungroup is None else ungroup
            if not group in self.unique_groups:
                raise KeyError("Input group not found in collection.")
        
        # Identify group indices
        if select_multiple:
            index = np.isin(self.groups, group)
        else:
            index = np.equal(self.groups, group)

        # Apply selection
        rc = self if inplace else self.copy()
        rc = rc.select_index(index, ignore=True, inplace=False)

        # Ungroup selection if necessary
        if ungroup:
            rc._groups = None

        return None if inplace else rc
    
    def set_closed(self, closed=None, inplace=False):
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
        # Check for events type
        if self.is_point and not closed is None:
            raise ValueError("Point events do not have closed parameters.")
        # Validate input closed parameter
        if closed is None:
            closed = self._class_options['default_closed']
        elif not closed in self._class_options['closed']:
            raise ValueError(
                "Collection's closed parameter must be one of "
                f"{self._class_options['closed']}.")
        # Apply changes
        rc = self if inplace else self.copy()
        rc._closed = closed
        rc._closed_base = closed.replace('_mod','')
        return None if inplace else rc

    def ungroup(self, inplace=False):
        """
        Remove group labels from the collection.
        """
        # Apply changes
        rc = self if inplace else self.copy()
        rc._groups = None
        return None if inplace else rc
    
    @utility._method_require(is_linear=True)
    def set_monotonic(self, inplace=False, **kwargs):
        """
        Arrange begin and end positions so that all ranges are increasing.

        Parameters
        ----------
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        # Sort ranges to enforce monotony
        begs, ends = np.sort(np.stack((self.begs, self.ends), axis=0), axis=0)

        # Apply changes
        rc = self if inplace else self.copy()
        rc._begs, rc._ends = begs, ends
        return None if inplace else rc

    def sort(self, by, ascending=True, inplace=False):
        f"""
        Sort the events by a selected event data anchor.
        
        Parameters
        ----------
        by : {self._class_options['keys_all']}
            The event data property or list of properties by which all events 
            should be sorted.
        ascending : boolean, default True
            Whether sorting should be done in ascending order. When False, 
            events will be sorted in descending order.
        inplace : boolean, default False
            Whether to perform the operation in place on the parent range
            collection, returning None.
        """
        # Determine sorting parameters
        if not type(by) is list:
            by = [by]
        if not set(by).issubset(self._class_options['keys_all']):
            raise ValueError(
                "Input 'by' parameter must be one or more of "
                f"{self._class_options['keys_all']}.")
        if self.is_point and ('begs' in by or 'ends' in by):
            raise ValueError(
                "Sorting by 'begs' or 'ends' is not available for point events.")
        if not self.is_located and 'locs' in by:
            raise ValueError(
                "Sorting by 'locs' is not available for unlocated events.")
        if type(ascending) is bool:
            ascending = [ascending for x in range(len(by))]
        elif type(ascending) is list and not len(ascending) == len(by):
            raise ValueError(
                "Input 'ascending' parameter must be single boolean "
                "value or must be list of same length as 'by'.")
        
        # Get the arrays for lexsort
        ascending = [1 if x else -1 for x in ascending[::-1]] # Reverse order per numpy lexsort
        by = [ascending[i] * getattr(self, x) for i, x in enumerate(by[::-1])]
        # Apply sorting
        index = np.lexsort(by)
        
        # Apply changes
        rc = self if inplace else self.copy()
        rc = rc.select_index(index, ignore=True, inplace=False)
        return None if inplace else rc
    
    def sort_standard(self, inplace=False):
        """
        Sort the events by their positional information in the standard order
        of 'groups', 'begs', 'ends' for linear events and 'groups', 'locs' 
        for point events.
        """
        # Determine sorting parameters
        if self.is_point:
            by = ['groups', 'locs']
        else:
            by = ['groups', 'begs', 'ends']
        ascending = [True for x in by]
        
        # Apply sorting
        return self.sort(by, ascending=ascending, inplace=inplace)
            
    def next_overlapping(self, all_=True, when_one=True, enforce_edges=False):
        """
        Whether all or any ranges are overlapping the next range in the 
        collection.

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of overlapping ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            overlapping, False if any adjacent ranges are not overlapping. If 
            False, will return an array of shape num_events - 1 of boolean 
            values indicating whether each range is overlapping the next.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        enforce_edges : bool, default False
            Whether to consider ranges which have a common vertex as 
            overlapping. This is independent of the collection's closed state.
        """
        # Validate input
        if self.num_events == 1:
            return when_one
        elif self.num_events == 0:
            raise ValueError("No ranges in collection.")

        # Check for overlapping
        cond1 = self.groups[1:] == self.groups[:-1] if self.groups is not None else True
        if enforce_edges:
            cond2 = self.begs[1:] <= self.ends[:-1]
        else:
            cond2 = self.begs[1:] < self.ends[:-1]
        res = cond1 & cond2
        if all_:
            return res.all()
        else:
            return res

    def next_consecutive(self, all_=True, when_one=True):
        """
        Whether all or any ranges are consecutive with the next range in the 
        collection, i.e. the end of one range is the beginning of the next.

        Parameters
        ----------
        all_ : bool, default True
            Whether to aggregate all tests of consecutive ranges, returning a 
            single boolean value. If True, will return True if all ranges are 
            consecutive, False if any adjacent ranges are not consecutive. If 
            False, will return an array of shape num_events - 1 of boolean 
            values indicating whether each range is consecutive to the next.
        when_one : bool, default True
            The default boolean value to return when only one range is included 
            in the collection.
        """
        # Validate input
        if self.num_events == 1:
            return np.array([when_one])
        elif self.num_events == 0:
            raise ValueError("No ranges in collection.")
        
        # Check for consecutive ranges
        res = (
            (self.begs[1:] == self.ends[:-1]) & 
            (self.groups[1:] == self.groups[:-1]) if self.groups is not None else True
        )
        if all_:
            return res.all()
        else:
            return res
        
    def consecutive_strings(self):
        """
        Identify strings of consecutive events in the collection, returning 
        an array of integers indicating which consecutive string each event 
        belongs to.
        """
        # Validate input
        if self.num_events == 0:
            raise ValueError("No ranges in collection.")
        elif self.num_events == 1:
            return np.array([0])
        
        # Identify consecutive strings
        res = np.zeros(self.num_events, dtype=int)
        res[1:] = np.cumsum(~self.next_consecutive(all_=False))
        return res

    def iter_groups(self, ungroup=True):
        """
        Iterate over the groups in the collection.
        """
        # Validate input
        if not self.is_grouped:
            raise ValueError("No groups in collection.")

        # Get group indices with groupby routine
        sorted_rng = self.sort_standard(inplace=False)
        unique_groups, splitter_i = np.unique(sorted_rng.groups, return_index=True)
        splitter_j = np.append(splitter_i[1:], len(sorted_rng.groups))
        
        # Iterate over groups
        for group, i, j in zip(unique_groups, splitter_i, splitter_j):
            rng = sorted_rng.select_index(slice(i, j), inplace=False)
            # Ungroup if necessary
            if ungroup:
                rng = rng.ungroup()
            yield group, rng
        
    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def separate(self):
        pass

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def dissolve(self, keep_index=False, return_index=False):
        """
        Dissolve consecutive linear events into single events. For best 
        results, input events should be sorted.

        Parameters
        ----------
        keep_index : bool or {'first', 'last'}, default False
            Whether to keep the index of the first or last event in each  
            dissolved event. If False, a new index will be created.
        return_index : bool, default False
            Whether to return a list of arrays indicating the indices of the 
            original events which were dissolved into each new event.
        """
        return modify.dissolve(self, return_index=return_index)

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def resegment(self):
        pass

    @utility._method_require(is_linear=True, is_monotonic=True, is_empty=False)
    def overlay(self, other: Rangel, normalize=True, norm_by='right'):
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
        # Validate input events
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Input events must be {self.__class__.__name__} class instance.")
        if not other.is_linear or not other.is_monotonic:
            raise ValueError(
                "Input events must be linear and monotonic.")
        
        # Perform overlay
        return relate.overlay(self, other, normalize=normalize, norm_by=norm_by)

    @utility._method_require(is_empty=False)
    def intersecting(self, other: Rangel, enforce_edges=True):
        """
        Identify intersections between two collections of events.

        Parameters
        ----------
        other : Rangel
            The other collection of events to test for intersections.
        enforce_edges : bool, default True
            Whether to consider cases of coincident begin and end points, 
            according to each collection's closed state. For instances where 
            these cases are not relevant, set enforce_edges=False for improved 
            performance.
        """
        # Validate input events
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Input events must be {self.__class__.__name__} class instance.")
        
        # Select intersection testing routine
        if self.is_point and other.is_point:
            return relate.intersection_point_point(
                self, other, enforce_edges=enforce_edges)
        elif self.is_point and other.is_linear:
            return relate.intersection_point_linear(
                self, other, enforce_edges=enforce_edges)
        elif self.is_linear and other.is_point:
            return relate.intersection_point_linear(
                other, self, enforce_edges=enforce_edges).T
        elif self.is_linear and other.is_linear:
            return relate.intersection_linear_linear(
                self, other, enforce_edges=enforce_edges)
        else:
            raise ValueError("Invalid event types for intersection testing.")
    