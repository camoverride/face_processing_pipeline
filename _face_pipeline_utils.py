import cv2
import mediapipe as mp
import numpy as np

from typing import List, Tuple



# Globals for testing
FACE_MESH_MIN_CONFIDENCE = 0.9


# MTCNN setup on CPU
from facenet_pytorch import MTCNN
import torch
device = torch.device("cpu")
mtcnn_detector = MTCNN(keep_all=True, device=device)
mtcnn_detector.pnet = mtcnn_detector.pnet.to(device)
mtcnn_detector.rnet = mtcnn_detector.rnet.to(device)
mtcnn_detector.onet = mtcnn_detector.onet.to(device)


# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(# type: ignore
    static_image_mode=False,

    # Because we are only running face_mesh in a single ROI.
    max_num_faces=1,

    # Needed to get faces.
    refine_landmarks=True,

    # NOTE: Should be placed in a config.
    min_detection_confidence=FACE_MESH_MIN_CONFIDENCE)


def crop_face_with_margin(image: np.ndarray,
                          bb: list,
                          margin_frac: float) -> tuple[np.ndarray, list, int]:
    """
    Crop the face from the image using a MTCNN-style
    bounding box and margin.

    TODO: what if it fails by overflowing the image?

    Parameters
    ----------
    image : np.ndarray
        Original image that contains one or more faces. Only the face corresponding
        region of the `bb` (bounding box) will be cropped.
    bb : list or tuple
        Bounding box in the format [x, y, w, h] in pixels.
    margin : float
        Margin added around image calculated in  "units of face width",
        e.g. if the face is 200 pixels wide and `margin_frac` is
        0.25, then the margin on all sides is 200*0.25 = 50 pixels.

    Returns
    -------
    np.ndarray
        A cropped image with a margin.
    """
    # Unpack the bounding box.
    x, y, w, h = bb

    # Extract image dimensions.
    img_h, img_w = image.shape[:2]

    # Calculate margin in pixels based on face width.
    margin_px = int(w * margin_frac)

    # Compute extended bounding box with margin.
    x1 = max(int(x - margin_px), 0)
    y1 = max(int(y - margin_px), 0)
    x2 = min(int(x + w + margin_px), img_w)
    y2 = min(int(y + h + margin_px), img_h)

    # Compute new width and height
    new_w = x2 - x1
    new_h = y2 - y1

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # New bounding box with margin added
    new_bb = [x1, y1, new_w, new_h]

    # Return the cropped image, new bb, and margin (in pixels).
    return cropped_image, new_bb, margin_px


def detect_faces(image : np.ndarray,
                 detector : str,
                 debug : bool) -> Tuple[list, list] | Tuple[None, None]:
    """
    If there are faces in an image, this function
    returns bounding boxes around each face.
    Returns None if the detector is not implemented.

    Parameters
    ----------
    image : np.ndarray
        An image that may or may not contain faces.
    detector : str
        The type of detector. Current options:
            - "mtcnn"
    debug : bool
        Display debug images.

    Returns
    -------
    Tuple[list]
        A tuple of (boxes, probs) where `boxes` is a list of
        bounding boxes for each face, in the format [x, y, w, h],
        and `probs` is the probability that each box correctly
        locates a face.
    None
        The detector is not implemented.
    """
    # Choose the MTCNN detector.
    if detector == "mtcnn":

        # Get the bounding boxes and probabilities.            
        _boxes, probs = mtcnn_detector.detect(image)  # type: ignore

        # TODO: remove this
        # Return `None` if no faces are detected
        if _boxes is None or len(_boxes) == 0:
            return None, None

        # Collect converted box coordinates
        boxes = []

        # Convert to (x, y, w, h) format
        for box in _boxes:
            x1, y1, x2, y2 = box
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)  # width = x2 - x1
            h = int(y2 - y1)  # height = y2 - y1

            boxes.append([x, y, w, h])

        # Display debug
        if debug and boxes:
            _image = image.copy()

            # Draw the bounding box with the correct coordinates
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the image with all bounding boxes.
            cv2.imshow(f"MTCNN bounding boxes", _image)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

        return boxes, probs  # type: ignore

    else:
        print("Detector not implemented!")
        return None, None


def get_no_margin_face(image : np.ndarray,
                       box : list[int],
                       debug : bool) -> np.ndarray:
    """
    Crops an image to a bounding box that contains a face.
    The face will be tightly cropped with no margin.

    Parameters
    ----------
    image : np.ndarray
        An image that contains one or more faces.
    box : list[int]
        A bounding box containing a face. The box is
        formatted as [x, y, w, h].
    debug : bool
        Display debug images.

    Returns
    -------
    np.ndarray
        An image cropped to a face.
    """
    # Get a face with no margin: `margin_frac` must be set to 0.
    no_margin_face, _, _ = crop_face_with_margin(image=image,
                                           bb=box,
                                           margin_frac=0)

    # Display debug image.
    if debug:
        cv2.imshow("Face cropped with no margin", no_margin_face)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return no_margin_face


def get_small_margin_face(image : np.ndarray,
                          box : list[int],
                          face_mesh_margin: float,
                          debug : bool) -> Tuple[np.ndarray, list, int]:
    """
    Crops an image to a bounding box that contains a face with
    a small margin around the box. This is done so that `face_mesh`
    can get the landmarks (with no margin, `face_mesh` fails).

    Parameters
    ----------
    image : np.ndarray
        An image that contains one or more faces.
    box : list[int]
        A bounding box containing a face. The box is
        formatted as [x, y, w, h].
    face_mesh_margin : float
        Margin added around image for `face_mesh` analysis.
        This is calculated in  "units of face width", e.g. if the
        face is 200 pixels wide and `face_mesh_margin` is 0.25, then
        the margin on all sides is 200*0.25 = 50 pixels.
    debug : bool
        Display debug images.

    Returns
    -------
    Tuple[np.ndarray, list]
        np.ndarray
            An image cropped to a face with a small margin.
        list
            The coordinates of the new bb `[x, y, w, h]`.
        int
            The face margin translated into pixels.
    """
    # Get the face with a small margin
    small_margin_face, new_bb, margin = crop_face_with_margin(image=image,
                                                              bb=box,
                                                              margin_frac=face_mesh_margin)

    # TODO: remove this,
    # if small_margin_face is None or small_margin_face.size == 0:
    #     return None, None, None

    # Display debug image.
    if debug:
        cv2.imshow("Face cropped with a small margin", small_margin_face)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return small_margin_face, new_bb, margin


def does_new_crop_overflow_image(image: np.ndarray,
                                 bb: list) -> bool:
    """
    Checks if the bounding box touches or overflows
    the edges of the image.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    bb : list
        A list of [x, y, w, h] representing the bounding box.

    Returns
    -------
    bool
        True if the bbox touches or overflows the image edges.
        False otherwise.
    """
    # Get the image dimensions.
    img_height, img_width = image.shape[:2]

    # Unpack the bounding boxes.
    x, y, w, h = bb

    # Get the edges of the margin added images.
    x_end = x + w
    y_end = y + h

    # Check for touching or overflow.
    if x <= 0 or y <= 0 or x_end >= img_width or y_end >= img_height:
        return True

    return False


def get_face_mesh(face_image : np.ndarray,
                  debug : bool) -> List[mp.solutions.face_mesh.NamedTuple] | None:  # type: ignore
    """
    Gets the landmarks of a face using mediapipe's `face_mesh`
    function. Requires a face with a margin around it, otherwise
    the "crop" will be too tight and `face_mesh` will fail.

    Parameters
    ----------
    face_image : np.ndarray
        An image containing a face which was detected using a bounding box
        and had a margin added around it.
    debug : bool
        Display debug images.

    Returns
    -------
    List[mp.solutions.face_mesh.NamedTuple]
        Mediapipe normalized face landmarks.
    None
        Face mesh can't be applied.
    """
    # Convert to RGB.
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Process with face mesh.
    face_mesh_results = face_mesh.process(rgb_image)

    # First check: did MediaPipe return any result object at all?
    if not face_mesh_results:
        # This means the entire processing failed (e.g., invalid input image).
        return None

    # Second check: are there actually any detected faces in the results?
    if not face_mesh_results.multi_face_landmarks:
        # Processing worked but no faces were found.
        return None

    # NOTE: we are assuming that only one face appears within the bounding box.
    # This is a generally safe assumption if the margin is not too large.
    face_mesh_landmarks = face_mesh_results.multi_face_landmarks[0]

    # Display debug image.
    if debug:
        _face_image = face_image.copy()
        image_h, image_w = face_image.shape[:2]
        for point in face_mesh_landmarks.landmark:
            cv2.circle(_face_image,
                        (int(point.x * image_w), int(point.y * image_h)),
                        2,
                        (255, 0, 0),
                        -1)

        cv2.imshow("face_mesh on a face that has a small margin", _face_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return face_mesh_landmarks


def reproject_landmarks(image : np.ndarray,
                        cropped_face : np.ndarray,
                        box : list[int],
                        face_mesh_landmarks : List[mp.solutions.face_mesh.NamedTuple],  # type: ignore
                        margin : int,
                        debug : bool) -> List[Tuple[int, int]]:
    """
    Translates the landmarks from the smaller cropped image
    back onto the original large image that contained the face.
    This is used for later rotation and re-cropping.

    Parameters
    ----------
    image : np.ndarray
        The original image that contains the face.
    cropped_face : np.ndarray
        A cropped section of `image` containing a face.
    box : list[int]
        A bounding box containing a face. The box is
        formatted as [x, y, w, h].
    face_mesh_landmarks : List[mp.solutions.face_mesh.NamedTuple]
        Mediapipe normalized face landmarks.
    margin : int
        Margin added around image for face_mesh analysis.
        Previously calculated as a fraction of the face width and translate to pixels.
    debug : bool
        Display debug images.

    Returns
    -------
    List[Tuple[int, int]]
        List of (x,y) coordinate pairs in the original image space,
        where each coordinate is represented as a tuple of two integers.
    """
    # Collect reprojected landmarks here
    reprojected_landmarks = []

    # Extract details from the cropped face.
    image_h, image_w = cropped_face.shape[:2]

    # Extract info from the bounding box.
    x, y, w, h = box

    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)


    for point in face_mesh_landmarks.landmark:  # type: ignore
        px = int(point.x * image_w)
        py = int(point.y * image_h)
        reprojected_landmarks.append((px + x1, py + y1))

    # Display debug image
    if debug:
        _image = image.copy()
        for (px, py) in reprojected_landmarks:
            cv2.circle(_image, (int(px), int(py)), 2, (0, 255, 255), -1)

        cv2.imshow("original frame with reprojected face_mesh", _image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return reprojected_landmarks


def rotate_face(image : np.ndarray,
                landmarks : List[Tuple[int, int]],
                debug : bool) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Rotate an image so that the eyes are positioned horizontally.
    This makes it much more straightforward for subsequent cropping.
    This function also returns all the landmarks, appropriately
    rotated.

    Parameters
    ----------
    image : np.ndarray
        Input image containing a face (BGR format).
    landmarks : List[Tuple[int, int]]
        List of facial landmark coordinates in (x,y) format.
        Expected to include at least eye landmarks (indices 33 and 263
        for left and right eyes respectively in MediaPipe's 468-point set).
    debug : bool
        Whether to display intermediate rotation results.

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int, int]]]
        np.ndarray
            The rotated image with eyes horizontally aligned.
        List[Tuple[int, int]]
            The transformed landmark coordinates in the rotated image space.
    """
    # Unpack the image dimensions.
    h, w = image.shape[:2]

    # Calculate rotation angle
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Create rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Rotate the landmarks (using homogeneous coordinates)
    landmarks_array = np.array(landmarks)
    homogeneous_landmarks = np.column_stack([landmarks_array,
                                             np.ones(len(landmarks_array))])
    rotated_landmarks = (rotation_matrix @ homogeneous_landmarks.T).T

    # Extract the rotated (x, y) coordinates
    rotated_landmarks = [(int(x), int(y)) for x, y, in rotated_landmarks]

    # Display debug images.
    if debug:
        _rotated_image = rotated_image.copy()
        cv2.imshow("rotated original image", _rotated_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        for (px, py) in rotated_landmarks:
            cv2.circle(_rotated_image, (int(px), int(py)), 2, (0, 255, 255), -1)

        cv2.imshow("rotated original image with landmarks", _rotated_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return rotated_image, rotated_landmarks


def pupil_crop_image(image: np.ndarray,
                     landmarks: List[Tuple[int, int]],
                     l: float,
                     r: float,
                     t: float,
                     desired_width: int,
                     desired_height: int,
                     debug: bool) -> Tuple[np.ndarray, List[Tuple[int, int]]] | Tuple[None, None]:
    """
    Crops the image based on the relative position of the eyes.
    K is the distance between pupils. Starting from the pupils,
    the value K is used to calculate the margins. For instance,
    if K is 200 pixels, then a left margin of 1.5 will mean that
    the left margin is 1.5 * 200 = 300 pixels, starting from the
    eyeball on the left side of the image.

    Returns None is cropping is not possible.

    TODO: Instead of padding with black, do smart padding (same color).

    Parameters
    ----------
    image : np.ndarray
        An image containing a face, rotated so eyes are horizontal.
    landmarks : List[Tuple[int, int]]
        List of (x, y) landmark coordinates.
    l : float
        Left margin, calculated as a fraction of K.
    r : float
        Right margin, calculated as a fraction of K.
    t : float
        Top margin, calculated as a fraction of K.
    desired_width : int
        The width of the output image.
    desired_height : int
        The height of the output image.
    debug : bool
        Display debug image.

    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int, int]]]
        np.ndarray
            The aligned, cropped, and resized image.
        List[Tuple[int, int]]
            The re-projected landmarks.
    None
        Cropping failed.
    """
    # Check if iris landmarks are available.
    # If not, try setting `refine_landmarks=True`.
    try:
        left_iris = np.mean(landmarks[468:473], axis=0)
        right_iris = np.mean(landmarks[473:478], axis=0)

    except IndexError:
        return None, None

    # Inter-pupil distance in pixels.
    K = right_iris[0] - left_iris[0]

    # Scale factor to make the crop match desired output width.
    crop_width_units = l + 1 + r
    scale = desired_width / (K * crop_width_units)

    # Compute required bottom margin `b` to preserve aspect ratio.
    crop_height_units = desired_height / (K * scale)
    b = crop_height_units - t

    # Bounding box in original image coordinates (float).
    x1 = left_iris[0] - K * l
    x2 = right_iris[0] + K * r
    y1 = left_iris[1] - K * t
    y2 = left_iris[1] + K * b

    # Use floor/ceil to get integer bounding box edges.
    x1_int = int(np.floor(x1))
    y1_int = int(np.floor(y1))
    x2_int = int(np.ceil(x2))
    y2_int = int(np.ceil(y2))

    # Calculate crop width/height exactly from integer edges.
    crop_width = x2_int - x1_int
    crop_height = y2_int - y1_int

    # Calculate overflow beyond image edges
    left_overflow = max(0, - x1_int)
    top_overflow = max(0, - y1_int)
    right_overflow = max(0, x2_int - image.shape[1])
    bottom_overflow = max(0, y2_int - image.shape[0])

    # Clamp crop coordinates inside image
    x1_clamped = max(0, x1_int)
    y1_clamped = max(0, y1_int)
    x2_clamped = min(image.shape[1], x2_int)
    y2_clamped = min(image.shape[0], y2_int)

    # Crop valid region from original image.
    cropped_valid = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Prepare black canvas for full crop size.
    channels = image.shape[2] if image.ndim == 3 else 1
    canvas_shape = (crop_height, crop_width, channels) \
        if channels > 1 else (crop_height, crop_width)
    canvas = np.zeros(canvas_shape, dtype=image.dtype)

    # Calculate paste offsets in canvas where valid crop will be placed
    paste_x = left_overflow
    paste_y = top_overflow

    # Paste the valid crop into the canvas with black padding around
    canvas[paste_y:paste_y + cropped_valid.shape[0],
           paste_x:paste_x + cropped_valid.shape[1]] = cropped_valid

    # Resize padded crop to desired output size
    resized = cv2.resize(canvas,
                         (desired_width, desired_height),
                         interpolation=cv2.INTER_AREA)

    # Adjust landmarks relative to crop + padding, then scale to output size.
    cropped_landmarks = []
    for (x, y) in landmarks:
        if x1 <= x <= x2 and y1 <= y <= y2:
            # Position relative to crop box
            x_rel = (x - x1)
            y_rel = (y - y1)

            # Add padding offsets
            x_rel += left_overflow
            y_rel += top_overflow

            # Scale to output dimensions
            x_adj = x_rel * desired_width / crop_width
            y_adj = y_rel * desired_height / crop_height
            cropped_landmarks.append((int(round(x_adj)), int(round(y_adj))))
        else:
            cropped_landmarks.append((-1, -1))

    # Show the image for debugging.
    if debug:
        dbg = resized.copy()
        for (x, y) in cropped_landmarks:
            if x != -1 and y != -1:
                cv2.circle(dbg, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow("Debug: pupil-cropped", dbg)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return resized, cropped_landmarks


def get_additional_landmarks(image_height : int,
                             image_width : int) -> List[Tuple[int, int]]:
    """
    Adds additional landmarks to an image. These landmarks are
    around the edges of the image. This helps with morphing so
    that the entire image can be tiled with delauney triangles.

    Parameters
    ----------
    image_height : int
        The height of the image in pixels.
    image_width : int
        The width of the image in pixels.

    Returns
    -------
    List[List[int]]
        A list of lists, where each sub-list is an additional landmark.
    """
    # subdiv.insert() cannot handle max values for edges, so add a small offset.
    offset = 0.0001

    # New coordinates to add to the landmarks
    new_coords = [
        # Corners of the image
        [0, 0],
        [image_width - offset, 0],
        [image_width - offset, image_height - offset],
        [0, image_height - offset],

        # Middle of the top, bottom, left, right sides
        [(image_width - offset) / 2, 0],
        [(image_width - offset) / 2, image_height - offset],
        [0, (image_height - offset) / 2],
        [image_width - offset, image_height / 2],
    ]

    int_coords = [(int(x), int(y)) for (x, y) in new_coords]

    return int_coords


def debug_display_all_landmarks(pupil_cropped_face : np.ndarray,
                                all_landmarks : List[Tuple[int, int]]):
    """
    Display the debug for `get_additional_landmarks`.

    Parameters
    ----------
    pupil_cropped_face : np.ndarray
        Processed face image.
    all_landmarks : List[Tuple[int, int]]
        All the landmarks.

    Returns
    -------
    None
        Displays a debug image.
    """
    _pupil_cropped_face = pupil_cropped_face.copy()
    for (px, py) in all_landmarks:
        cv2.circle(_pupil_cropped_face, (int(px), int(py)), 2, (0, 255, 255), -1)

    cv2.imshow("Pupil cropped image with translated landmarks", _pupil_cropped_face)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def quantify_blur(face_image : np.ndarray) -> float:
    """
    Quantifies how blurry a face is by using Laplacian
    variance to assign a value to the blur.

    NOTE: The face image should be tightly cropped so
    that no background is included in the calculation.

    Parameters
    ----------
    face_image : np.ndarray
        An image containing a single face.

    Returns
    -------
    float
        The amount of blur.
    """
    # Convert to gray.
    grayscale_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Compute the Laplacian and its variance
    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance


def assess_head_direction(face_landmarks : List[Tuple[int, int]]) -> float:
    """
    Assess whether the head is facing forward or to the side.

    Returns True is the head is facing forward, otherwise False.

    TODO: this function is NOT YET IMPLEMENTED!

    Parameters
    ----------
    landmarks : List[int]
        A list containing tuples of landmarks (x, y).
    
    Returns
    -------
    float
        A value that is 0 if the face is looking directly forward
        and some other value if it's looking in another direction.
    """
    return True
