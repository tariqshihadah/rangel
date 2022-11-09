import unittest
from unittest import TestCase

from rangel import RangeCollection
import numpy as np


class TestConstructors(TestCase):
    """
    Test various constructor methods.
    """
    def test_from_steps(self):
        # from_steps
        # steps = 1
        rc1 = RangeCollection([0,1,2,3,4],[1,2,3,4,5])
        rc2 = RangeCollection.from_steps(0, 5, length=1, steps=1)
        self.assertTrue((rc1.rng==rc2.rng).all())
        # steps != 1
        rc1 = RangeCollection([0,1,2,3,4],[2,3,4,5,6])
        rc2 = RangeCollection.from_steps(0, 6, length=2, steps=2)
        self.assertTrue((rc1.rng==rc2.rng).all())
        # fill == 'cut'
        rc1 = RangeCollection([0,2,4,6],[2,4,6,7])
        rc2 = RangeCollection.from_steps(0, 7, length=2, steps=1, fill='cut')
        self.assertTrue((rc1.rng==rc2.rng).all())
        # fill == 'none'
        rc1 = RangeCollection([0,2,4],[2,4,6])
        rc2 = RangeCollection.from_steps(0, 7, length=2, steps=1, fill='none')
        self.assertTrue((rc1.rng==rc2.rng).all())
        # fill == 'left'
        rc1 = RangeCollection([0,2,4,5],[2,4,6,7])
        rc2 = RangeCollection.from_steps(0, 7, length=2, steps=1, fill='left')
        self.assertTrue((rc1.rng==rc2.rng).all())
        # fill == 'right'
        rc1 = RangeCollection([0,2,4,6],[2,4,6,8])
        rc2 = RangeCollection.from_steps(0, 7, length=2, steps=1, fill='right')
        self.assertTrue((rc1.rng==rc2.rng).all())
        # fill == 'extend'
        rc1 = RangeCollection([0,2,4],[2,4,7])
        rc2 = RangeCollection.from_steps(0, 7, length=2, steps=1, fill='extend')
        self.assertTrue((rc1.rng==rc2.rng).all())

    def test_intersecting(self):
        # Create collections
        
        # single range, middle cases
        rc1 = RangeCollection.from_steps(
            0, 10, length=2, steps=1, closed='left')
        x = rc1.intersecting(1,5)
        y = [True,True,True,False,False]
        self.assertTrue(np.all(x==y))

        # single range, edge cases, left
        rc1 = RangeCollection.from_steps(
            0, 10, length=2, steps=1, closed='left')
        x = rc1.intersecting(2,6)
        y = [False,  True,  True,  True, False]
        self.assertTrue(np.all(x==y))

        # single range, edge cases, right
        rc1 = RangeCollection.from_steps(
            0, 10, length=2, steps=1, closed='right')
        x = rc1.intersecting(2,6)
        y = [True,  True,  True,  False, False]
        self.assertTrue(np.all(x==y))

        # multiple ranges, all cases, various
        rc1 = RangeCollection.from_steps(
            0, 10, length=2, steps=1, closed='right')
        x = rc1.intersecting([0,2,5,10],[1,2,8,12])
        y = [
            [ True,  True, False, False],
            [False, False, False, False],
            [False, False,  True, False],
            [False, False,  True, False],
            [False, False, False,  True]
        ]
        self.assertTrue(np.all(x==y))

        # range collection, all cases, right-both
        rc1 = RangeCollection.from_steps(
            0, 10, length=2, steps=1, closed='right')
        rc2 = RangeCollection([0,2,5,10],[1,2,8,12], closed='both')
        x = rc1.intersecting(other=rc2)
        y = [
            [ True,  True, False, False],
            [False, False, False, False],
            [False, False,  True, False],
            [False, False,  True, False],
            [False, False, False,  True]
        ]
        self.assertTrue(np.all(x==y))

        # range collection, all cases, right-right
        rc1 = RangeCollection.from_steps(
            0, 10, length=2, steps=1, closed='right')
        rc2 = RangeCollection([0,2,5,10],[1,2,8,12], closed='right')
        x = rc1.intersecting(other=rc2)
        y = [
            [ True, False, False, False],
            [False, False, False, False],
            [False, False,  True, False],
            [False, False,  True, False],
            [False, False, False, False]
        ]
        self.assertTrue(np.all(x==y))


if __name__ == '__main__':
    unittest.main()