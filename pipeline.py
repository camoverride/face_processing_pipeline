import logging
import numpy as np
import os
import sys
from typing import List

from _face_pipeline_utils import detect_faces, get_no_margin_face, \
    get_small_margin_face, does_new_crop_overflow_image, get_face_mesh, \
        reproject_landmarks, rotate_face, pupil_crop_image, get_additional_landmarks, \
            quantify_blur, assess_head_direction, combine_landmarks



# Set up basic logging.
os.environ["QT_LOGGING_RULES"] = "*=false"

logging.basicConfig(level=logging.DEBUG,
                    stream=sys.stdout,
                    force=True,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def face_processing_pipeline(image : np.ndarray,
                             l : float,
                             r : float,
                             t : float,
                             desired_width : int,
                             desired_height : int,
                             detector : str,
                             face_mesh_margin : float,
                             debug : bool) -> List[dict] | None:
    """
    This accepts a picture that may or may not contain faces and
    extracts and processes all the faces so that they are in a
    standardized format. This format means that the eyeballs are in
    the same relative position in each image, and each image is scaled
    to be the same shape. This function returns a dict that contains
    this processed image along with some additional metadata about
    the image:
        {
            "face_image" : np.ndarray,
            "face_landmarks" : List[list[int]],
            "face_landmarks_extra": List[list[int]],
            "blur" : float,
            "head_forward" : bool,
            "original_face_width" : int,
            "original_face_height" : int
        }

    If no faces are detected, the function returns None.

    Parameters
    ----------
    image : np.ndarray
        The input image that might contain faces.
    l : float
        The left margin, as a fraction of K, the distance between pupils.
    r : float
        The right margin, as a fraction of K, the distance between pupils.
    t : float
        The top margin, as a fraction of K, the distance between pupils.
    desired_width : int
        The output width of the face image in pixels.
    desired_height : int
        The output width of the face image in pixels.
    detector : str
        Which face detection model to use. Current options:
            - "mtcnn"
    face_mesh_margin : float
        Margin added around image for `face_mesh` analysis.
        This is calculated in  "units of face width", e.g. if the
        face is 200 pixels wide and `face_mesh_margin` is 0.25, then
        the margin on all sides is 200*0.25 = 50 pixels.

    debug : bool
        Show the intermediate processing steps as images.

    Returns
    -------
    List[dict]
        A list of dicts with the following information:
            {
                "face_image" : np.ndarray,
                "prob" : float,
                "face_landmarks" : List[list[int]],
                "face_landmarks_extra": List[list[int]],
                "blur" : float,
                "head_forward" : bool,
                "original_face_width" : int,
                "original_face_height" : int
            }
    None
        If no faces are detected in the image.
    """
    # Collect all the processed face data here.
    processed_face_data = []

    # Detect faces in the image and get bounding boxes.
    boxes, probs = detect_faces(image=image,
                                detector=detector,
                                debug=debug)

    # If there are no faces, return None.
    if (boxes == None) or (len(boxes) == 0) or (probs == None):
        logging.debug("No faces detected!")
        return None

    # If faces are detected, iterate through all of them.
    # If any of these steps fails, `continue` to the next image.
    for box, prob in zip(boxes, probs):

        # Get a copy of this face with no margin.
        no_margin_face = get_no_margin_face(image=image,
                                            box=box,
                                            debug=debug)

        if no_margin_face is None:
            logging.info("Could not crop face [no margin]!")
            continue

        # Get a copy of this face with a small margin
        small_margin_face, new_bb, margin = get_small_margin_face(image=image,
                                                                  box=box,
                                                                  face_mesh_margin=face_mesh_margin,
                                                                  debug=debug)

        if small_margin_face is None:
            logging.info("Could not crop face [small margin]!")
            continue

        # Check if the newly cropped face overflows the image
        overflow = does_new_crop_overflow_image(image=image,
                                                bb=new_bb)

        if overflow:
            logging.info("Newly cropped face overflows original image. Skipping!")
            continue

        # Get the face_mesh landmarks from the small_margin_face
        face_mesh_landmarks = get_face_mesh(face_image=small_margin_face,
                                            debug=debug)

        if face_mesh_landmarks is None:
            logging.info("No face mesh landmarks!")
            continue

        # Reproject the face_mesh from the `small_margin_face` onto the entire image.
        reprojected_landmarks = reproject_landmarks(image=image,
                                                    cropped_face=small_margin_face,
                                                    box=box,
                                                    face_mesh_landmarks=face_mesh_landmarks,
                                                    margin=margin,
                                                    debug=debug)

        # Use this mesh to rotate the image.
        rotated_image, rotated_landmarks = rotate_face(image=image,
                                                       landmarks=reprojected_landmarks,
                                                       debug=debug)

        if rotated_image is None:
            logging.info("Could not rotate image!")
            continue

        # Use the rotated image and rotated mesh to crop to eyeballs.
        pupil_cropped_face, pupil_cropped_landmarks = \
            pupil_crop_image(image=rotated_image,
                             landmarks=rotated_landmarks,
                             l=l,
                             r=r,
                             t=t,
                             desired_width=desired_width,
                             desired_height=desired_height,
                             debug=debug)

        if (pupil_cropped_face is None) or (pupil_cropped_landmarks is None):
            logging.info("Could not pupil crop image")
            continue

        # Get the all the landmarks required for later processing functions.
        h, w = pupil_cropped_face.shape[:2]
        additional_landmarks = get_additional_landmarks(h, w)

        # NOTE: Not used. Instead, these landmarks are returned separately. Used for debugging.
        _all_landmarks = combine_landmarks(pupil_cropped_face=pupil_cropped_face,
                                           face_landmarks=pupil_cropped_landmarks,
                                           additional_landmarks=additional_landmarks,
                                           debug=debug)

        # Assess the blur using the no-margin face.
        blur = quantify_blur(face_image=no_margin_face)

        # Assess the head direction using the scaled image landmarks.
        head_direction = assess_head_direction(face_landmarks=pupil_cropped_landmarks)        

        # Collect all the face info.
        face_info = {
                "face_image" : pupil_cropped_face,
                "prob" : prob,
                "face_landmarks" : pupil_cropped_landmarks,
                "face_landmarks_extra": additional_landmarks,
                "blur" : blur,
                "head_forward" : head_direction,
                "original_face_width" : no_margin_face.shape[1],
                "original_face_height" : no_margin_face.shape[0]
        }

        processed_face_data.append(face_info)

    return processed_face_data
