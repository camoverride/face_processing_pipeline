import logging
import os
import sys

from typing import List
import numpy as np

from _face_pipeline_utils import detect_faces, get_no_margin_face, \
    get_small_margin_face, get_face_mesh, reproject_landmarks, \
        rotate_face, pupil_crop_image, get_additional_landmarks, \
            debug_display_all_landmarks,scale_image_and_landmarks, \
                  quantify_blur, assess_head_direction



# Set up basic logging.
os.environ["QT_LOGGING_RULES"] = "*=false"

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    force=True,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def face_processing_pipeline(image : np.ndarray,
                             l : float,
                             r : float,
                             t : float,
                             b : float,
                             output_width : int,
                             output_height : int,
                             detector : str,
                             face_mesh_margin : int,
                             debug : bool) -> List[dict] :
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

    If no faces are detected, the function returns `False`

    Parameters
    ----------
    image : np.ndarray
        The input image that might contain faces.
    l : float
        The left margin, as a fraction of K, the distance between eyes.
    r : float
        The right margin, as a fraction of K, the distance between eyes.
    t : float
        The top margin, as a fraction of K, the distance between eyes.
    b : float
        The bottom margin, as a fraction of K, the distance between eyes.
    output_width : int
        The output width of the face image.
    output_height : int
        The output width of the face image.
    detector : str
        Which face detection model should be used. Current options:
            - "mtcnn"
    face_mesh_margin : int
        Margin added around image for face_mesh analysis
        NOTE: should be removed and calculated empirically
    debug : bool
        Show the intermediate processing steps as images.

    Returns
    -------
    list or False
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
        or `False` if no faces are detected in the image.
    
    """
    # Collect all the processed face data here.
    processed_face_data = []

    # Detect faces in the image and get bounding boxes.
    boxes, probs = detect_faces(image=image,
                                detector=detector,
                                debug=debug)

    # If no faces are detected, return False
    if boxes is None or len(boxes) == 0:
        logging.debug("No faces detected!")
        return False

    # If faces are detected, iterate through all of them.
    # If any of these steps fails, `continue` to the next image.
    for box, prob in zip(boxes, probs):        
        # Get a copy of this face with no margin
        no_margin_face = get_no_margin_face(image=image,
                                            box=box,
                                            debug=debug)
        
        if no_margin_face is None:
            logging.info("Could not crop face [no margin]!")
            continue


        # Get a copy of this face with a small margin
        small_margin_face = get_small_margin_face(image=image,
                                                  box=box,
                                                  face_mesh_margin=face_mesh_margin,
                                                  debug=debug)
        
        if small_margin_face is None:
            logging.info("Could not crop face [small margin]!")
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
                                                    face_mesh_margin=face_mesh_margin,
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
                            b=b,
                            debug=debug)
        
        if pupil_cropped_face is None:
            logging.info("Could not pupil crop image")
            continue
        

        # Get the all the landmarks required for later processing functions.
        h, w = pupil_cropped_face.shape[:2]
        additional_landmarks = get_additional_landmarks(h, w)
        _all_landmarks = pupil_cropped_landmarks + additional_landmarks

        if debug:
            debug_display_all_landmarks(pupil_cropped_face=pupil_cropped_face,
                                        all_landmarks=_all_landmarks)


        # Scale the image the the landmarks to the desired output size.
        # (1) Scale the face landmarks.
        scaled_image, scaled_landmarks = \
              scale_image_and_landmarks(image=pupil_cropped_face,
                                        landmarks=pupil_cropped_landmarks,
                                        output_width=output_width,
                                        output_height=output_height,
                                        debug=debug)
        
        # (2) Scale the additional landmarks.
        _, scaled_additional_landmarks = \
              scale_image_and_landmarks(image=pupil_cropped_face,
                                        landmarks=additional_landmarks,
                                        output_width=output_width,
                                        output_height=output_height,
                                        debug=debug)
        

        # Assess the blur using the no-margin face.
        blur = quantify_blur(face_image=no_margin_face)

        # Assess the head direction using the scaled image landmarks.
        head_direction = assess_head_direction(face_landmarks=scaled_landmarks)        

        # Collect all the face info.
        face_info = {
                "face_image" : scaled_image,
                "prob" : prob,
                "face_landmarks" : scaled_landmarks,
                "face_landmarks_extra": scaled_additional_landmarks,
                "blur" : blur,
                "head_forward" : head_direction,
                "original_face_width" :  no_margin_face.shape[1],
                "original_face_height" : no_margin_face.shape[0]
        }

        processed_face_data.append(face_info)
