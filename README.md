# Overview
Module featuring RangeCollection object class for the management of range data and optimized performance of various simple and complex range operations, including range and point overlays, equitable and score-based separation of overlapping ranges, generation of random range data, and range data comparisons among others.

# Release Notes
## 0.0.6 (2022-10-14)
- Modified intersecting method to consider both collection closed parameter and input ranges closed parameter. Now, the closed parameter in the method refers to the edges of the input ranges, not as a call-time modifier to the class instance's closed parameter. Instead, if needed, this should be done prior to the intersection method call with the set_closed method.
- Removed unused get_... methods based on the is_... methods for automatic selecting to clean namespace.
- Change RangeCollection default sort value to False.
- Added closed parameter visualization to .plot() method on RangeCollection class.
- Various minor feature additions
- Various bug fixes, performance improvements

## 0.0.5 (2022-09-02)
- Added balance option to from_steps method
- Various minor feature additions
- Various bug fixes, performance improvements

## 0.0.4 (2022-04-11)
- Added snapping to set_centers and __init__ methods
- Added squeeze parameter to intersecting method
- Various bug fixes, performance improvements

## 0.0.3 (2021-11-08)
- Initial updated merging features
- Various bug fixes, performance improvements

## 0.0.2 (2021-09-27)
- Various bug fixes, performance improvements

## 0.0.1 (2021-09-13)
- Initial release
