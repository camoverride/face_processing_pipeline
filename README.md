# face_processing_pipeline

Utilities and pipeline for processing faces.

This is designed to grab forward-looking faces and align them to standardized
pupil positions. This is very useful for tasks such as face averaging and morphing.

There are several utilities in the pipeline:
- Head orientation detection.
- Gaze direction detection.
- Blurriness quantification.
- Brightness quantification.
- Uniqueness detection (seen before).


## Setup

`python3 -m venv .venv`
`source .venv/bin/activate`
`pip install -r requirements.txt`


## Test

`python pipeline_test.py`


## TODO

- merge `get_no_margin_face` and `get_small_margin_face`
- rlu cache
- The landmark handling could be encapsulated in a dedicated class
- The repeated image copying for debug displays could be optimized
- The blur assessment could be memoized if called multiple times
- The margin overflow check happens after face mesh detection which may be wasteful
- No explicit resource cleanup for OpenCV windows
- The rotation step could introduce artifacts at image edges
- The black padding in pupil cropping might not be ideal for all use cases
- Some docstrings could be more specific about units (e.g., pixels vs normalized coordinates)
- A few type hints could be more precise (e.g., using Optional where None is possible)
- The debug parameter is repeated everywhere - could be part of a config object

- turn into a proper python package that can be imported
- implement "assess head direction"
