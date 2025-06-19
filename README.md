# face_processing_pipeline

Utilities and pipeline for processing faces for various downstream projects.


## Setup

`python3 -m venv .venv`
`source .venv/bin/activate`
`pip install -r requirements.txt`


## Test

`python pipeline_test.py`


## TODO

- [ ] wrong crop when face is at border of image. Should FAIL if face margin overflows or is too close to the edge of the image.
- [ ] add `face_mesh_margin` empirically
- [ ] add a variety of tests with images of different sizes, qualities, number of faces, occlusions, lighting conditions, etc.
- [ ] turn into object so that `FACE_MESH_MIN_CONFIDENCE` from `_face_pipeline_utils.py` can be added
- [ ] turn into a proper python package that can be imported
- [ ] final cropping step should not distort aspect ration of image -- instead crop from larger area of image!
