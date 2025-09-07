# face_processing_pipeline

Utilities and pipeline for processing faces.


## Overview

This repo performs two main tasks:

- (1) it exposes `FaceProcessor` which can be used for identifying high-quality forward-looking faces, demonstrated in `pipeline_test.py`
- (2) it exposes various utilities for face morphing, demonstrated in `morph_test.py`


## Setup

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`


## Test

Test the forward face pipeline:
- `python pipeline_test.py`

Test the morphing function:
- `python morph_test.py`
