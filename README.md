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



- [X] wrong crop when face is at border of image. Should FAIL if face margin overflows or is too close to the edge of the image.
- [X] final cropping step should not distort aspect ratio of image -- instead crop from larger area of image!
- [ ] one more cropping step to crop from image, to satisfy monitor size!
- [ ] Fail if over-cropping (maybe!)
- [ ] add `face_mesh_margin` empirically
- [ ] add a variety of tests with images of different sizes, qualities, number of faces, occlusions, lighting conditions, etc.
- [ ] turn into object so that `FACE_MESH_MIN_CONFIDENCE` from `_face_pipeline_utils.py` can be added
- [ ] turn into a proper python package that can be imported
- [ ] implement "assess head direction"
