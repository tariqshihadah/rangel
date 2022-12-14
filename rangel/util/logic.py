import numpy as np

class _ArrayLogicManager(object):
    """
    Class for managing logical comparisons between two arrays which may be 
    referenced in some range operations. Using this class will ensure that 
    costly logical computations are only performed when required, and not more 
    than once.
    """
    
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self, obj):
        try:
            assert isinstance(obj, np.ndarray)
            self._left = obj
        except:
            raise IndexError("Input objects must be np.ndarrays.")
    
    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, obj):
        try:
            assert isinstance(obj, np.ndarray)
            self._right = obj
        except:
            raise IndexError("Input objects must be np.ndarrays.")
    
    @property
    def equal(self):
        try:
            return self._equal
        except AttributeError:
            self._equal = np.equal(self._left, self._right)
            return self._equal
    
    @property
    def greater(self):
        try:
            return self._greater
        except AttributeError:
            self._greater = np.greater(self._left, self._right)
            return self._greater
    
    @property
    def less(self):
        try:
            return self._less
        except AttributeError:
            self._less = np.less(self._left, self._right)
            return self._less
    
    @property
    def greater_equal(self):
        try:
            return self._greater_equal
        except AttributeError:
            self._greater_equal = np.greater_equal(self._left, self._right)
            return self._greater_equal
    
    @property
    def less_equal(self):
        try:
            return self._less_equal
        except AttributeError:
            self._less_equal = np.less_equal(self._left, self._right)
            return self._less_equal
        
    @property
    def not_equal(self):
        try:
            return self._not_equal
        except AttributeError:
            self._not_equal = np.not_equal(self._left, self._right)
            return self._not_equal
        