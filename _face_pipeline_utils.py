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
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=FACE_MESH_MIN_CONFIDENCE,
)


def crop_face_with_margin(image: np.ndarray,
                          bb: list,
                          margin: int) -> np.ndarray:
    """
    Crop the face from the image using a MTCNN-style
    bounding box and margin.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    bb : list or tuple
        Bounding box in the format [x, y, w, h] in pixels.
    margin : int
        Margin to add (in pixels) around the bounding box.

    Returns
    -------
    np.ndarray
        A cropped image with a margin.
    """
    x, y, w, h = bb
    img_h, img_w = image.shape[:2]

    # Compute extended bounding box with margin
    x1 = max(int(x - margin), 0)
    y1 = max(int(y - margin), 0)
    x2 = min(int(x + w + margin), img_w)
    y2 = min(int(y + h + margin), img_h)

    # Compute new width and height
    new_w = x2 - x1
    new_h = y2 - y1

    # Return the cropped image and the new bb
    return image[y1:y2, x1:x2], [x1, y1, new_w, new_h]


def detect_faces(image : np.ndarray,
                 detector : str,
                 debug : bool) -> Tuple[list]:
    """
    If there are faces in an image, this function
    returns bounding boxes around each face.

    Parameters
    ----------
    image : np.ndarray
        An image that may or may not contain faces.
    detector : str
        The type of detector.
        Current options: "mtcnn"
    debug : bool
        Display debug images.
    
    Returns
    -------
    Tuple[list]
        A tuple of (boxes, probs) where `boxes` is a list of
        bounding boxes for each face, in the format [x, y, w, h],
        and `probs` is the probability that each box correctly
        locates a face.
    """
    # Choose the MTCNN detector.
    if detector == "mtcnn":

        # Get the bounding boxes and probabilities
        _boxes, probs = mtcnn_detector.detect(image)

        # Return `None` if no faces are detected
        if _boxes is None or len(_boxes) == 0:
            return None
        
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

    return boxes, probs


def get_no_margin_face(image : np.ndarray,
                       box : list[int],
                       debug : bool) -> np.ndarray:
    """
    Crops an image to a bounding box that contains a face with no
    margin around the box.

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
    # `margin` must be set to 0.
    no_margin_face, _ = crop_face_with_margin(image=image,
                                           bb=box,
                                           margin=0)

    if no_margin_face is None or no_margin_face.size == 0:
        return None

    # Display debug image
    if debug:
        cv2.imshow("Face cropped with no margin", no_margin_face)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return no_margin_face


def get_small_margin_face(image : np.ndarray,
                          box : list[int],
                          face_mesh_margin: int,
                          debug : bool) -> tuple:
    """
    Crops an image to a bounding box that contains a face with
    a small margin around the box. This is done so that face_mesh
    can get the landmarks (with no margin, face_mesh fails).

    TODO: calculate MARGIN as a fraction of the face size

    Parameters
    ----------
    image : np.ndarray
        An image that contains one or more faces.
    box : list[int]
        A bounding box containing a face. The box is
        formatted as [x, y, w, h].
    face_mesh_margin : int
        Margin added around image for face_mesh analysis
        NOTE: should be removed and calculated empirically
    debug : bool
        Display debug images.
    
    Returns
    -------
    tuple
        np.ndarray
            An image cropped to a face with a small margin.
        list
            The coordinates of the new bb [x, y, w, h]

    """
    # Get the face with a small margin
    small_margin_face, new_bb = crop_face_with_margin(image=image,
                                                      bb=box,
                                                      margin=face_mesh_margin)

    if small_margin_face is None or small_margin_face.size == 0:
        return None

    # Display debug image.
    if debug:
        cv2.imshow("Face cropped with a small margin", small_margin_face)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return small_margin_face, new_bb


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
        True if the bbox touches or overflows the image edges,
        False otherwise.
    """
    img_height, img_width = image.shape[:2]
    x, y, w, h = bb

    x_end = x + w
    y_end = y + h

    # Check for touching or overflow
    if x <= 0 or y <= 0 or x_end >= img_width or y_end >= img_height:
        return True

    return False


def get_face_mesh(face_image : np.ndarray,
                  debug : bool) :
    """
    Gets the landmarks of a face using mediapipe's `face_mesh`
    function. Requires a face with a margin around it.

    Parameters
    ----------
    face_image : np.ndarray
        An image containing a face with a margin around it.
    debug : bool
        Display debug images
    
    Returns
    -------
    object
        TODO: what type is returned?
    """
    # Convert to RGB
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Process with face mesh
    face_mesh_results = face_mesh.process(rgb_image)

    if not face_mesh_results:
        return None

    if not face_mesh_results.multi_face_landmarks:
        return None

    # NOTE: we are assuming that only one face appears within the bounding box.
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
                        face_mesh_landmarks,
                        face_mesh_margin : int,
                        debug : bool) -> List[list[int]]:
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
    face_mesh_landmarks :
        TODO: what type exactly?
    face_mesh_margin : int
        Margin added around image for face_mesh analysis
        NOTE: should be removed and calculated empirically
    debug : bool
        Display debug images.
    
    Returns
    -------
    List[list[int]]
        TODO: check type. List[tuple[int]]? Are we sure it's ints?
    """
    # Collect reprojected landmarks here
    reprojected_landmarks = []

    # Extract details from the cropped face.
    image_h, image_w = cropped_face.shape[:2]

    # Extract info from the bounding box.
    x, y, w, h = box

    x1 = max(x - face_mesh_margin, 0)
    y1 = max(y - face_mesh_margin, 0)

    
    for point in face_mesh_landmarks.landmark:
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
                landmarks : tuple,
                debug : bool) -> tuple:
    """
    Rotate an image so that the eyes are positioned horizontally.
    This makes it much more straightforward for subsequent cropping.
    This function also returns all the landmarks, appropriately
    rotated.

    Parameters
    ----------
    image : np.ndarray
        An image that should contain a face.
    bb : tuple
        A bounding box containing the face we care about.
    
    Returns
    -------
    tuple
        A tuple of two items.
        The first is a np.ndarray rotated image.
        The second is the rotated landmarks from mediapipe.
        TODO: what type exactly are the mediapipe landmarks?
        TODO: type hint Tuple[type[type]]
    """
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


def pupil_crop_image(image : np.ndarray,
                     landmarks : List[tuple],
                     l : float,
                     r : float,
                     t : float,
                     b : float,
                     debug : bool) -> np.ndarray:
    """
    Crops the image based on the relative position of the eyes.
    K is the distance between pupils. Starting from the pupils,
    the value K is used to calculate the margins. For instance,
    if K is 200 pixels, then a left margin of 1.5 will mean that
    the left margin is 1.5 * 200 = 300 pixels, starting from the
    eyeball on the left side of the image.

    Parameters
    ----------
    image : np.ndarray
        An image containing a face that has been rotated so that the
        eyes are on a horizontal plain.
    landmarks : List[tuple]
        Landmarks that are a list of (x, y) coordinates.
    l : float
        The left margin, calculated as a fraction of K.
    r : float
        The right margin, calculated as a fraction of K.
    t : float
        The top margin, calculated as a fraction of K.
    b : float
        The bottom margin, calculated as a fraction of K.
    debug : bool
        Display debug image.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    # Iris landmarks (468-472 for right eye, 473-477 for left eye)
    try:
        left_iris_landmarks = [landmarks[468],
                               landmarks[469],
                               landmarks[470],
                               landmarks[471],
                               landmarks[472]]
        right_iris_landmarks = [landmarks[473],
                                landmarks[474],
                                landmarks[475],
                                landmarks[476],
                                landmarks[477]]
        
    # If this exception gets hit too much, set refine_landmarks=True
    except IndexError:
        raise ValueError("Iris landmarks not available.")
    
    # Calculate the center of the left and right iris.
    left_iris_center = np.mean([(int(lm[0]),   # lm.x == lm[0], lm.y == lm[1]
                                 int(lm[1])) \
                                    for lm in left_iris_landmarks],
                                 axis=0)
    right_iris_center = np.mean([(int(lm[0]),
                                  int(lm[1])) \
                                    for lm in right_iris_landmarks],
                                  axis=0)
    
    # Calculate the distance between the eyes.
    K = right_iris_center[0] - left_iris_center[0]
    
    # Calculate crop coordinates
    x1 = int(left_iris_center[0] - l * K)
    x2 = int(right_iris_center[0] + r * K)
    y1 = int(left_iris_center[1] - t * K)
    y2 = int(left_iris_center[1] + b * K)
    
    # Create a blank image (black) of the desired crop size
    crop_height = y2 - y1
    crop_width = x2 - x1
    cropped_image = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    
    # Calculate the valid region of the original image to copy
    src_x1 = max(x1, 0)
    src_x2 = min(x2, image.shape[1])
    src_y1 = max(y1, 0)
    src_y2 = min(y2, image.shape[0])
    
    # Calculate the destination region in the blank image
    dst_x1 = src_x1 - x1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = src_y1 - y1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    # Copy the valid region from the original image to the blank image
    if src_x2 > src_x1 and src_y2 > src_y1:
        cropped_image[dst_y1:dst_y2, dst_x1:dst_x2] = \
            image[src_y1:src_y2, src_x1:src_x2]

    # Transform the landmarks to the cropped image space
    cropped_image_landmarks = []
    for (x, y) in landmarks:
        # Convert to pixel coordinates
        x_pixel = int(x)
        y_pixel = int(y)
        
        # Adjust the coordinates to the cropped image's space
        x_cropped = int((x_pixel - x1) * crop_width / (x2 - x1)) \
            if x1 < x_pixel < x2 else -1
        y_cropped = int((y_pixel - y1) * crop_height / (y2 - y1)) \
            if y1 < y_pixel < y2 else -1
        
        # Append the new transformed landmark
        cropped_image_landmarks.append((x_cropped, y_cropped))

    # Display debug image.
    if debug:
        cv2.imshow("pupil-cropped/aligned image", cropped_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    return cropped_image, cropped_image_landmarks


def get_additional_landmarks(image_height : int,
                             image_width : int) -> List[List[int]]:
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
    # TODO: why???
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
                                all_landmarks : list[int]):
    """
    Display the debug for `get_additional_landmarks`.

    Parameters
    ----------
    pupil_cropped_face : np.ndarray
        Processed face image.
    all_landmarks : list[int]
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


def scale_image_and_landmarks(image: np.ndarray,
                              landmarks: list[tuple[float, float]],
                              output_width: int,
                              output_height: int,
                              debug : bool) \
                                    -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Scale image and landmarks to exact output dimensions,
    allowing aspect ratio change.
    
    Parameters
    ----------
    image : np.ndarray
        An image containing a face to be re-scaled.
    landmarks : list[tuple]
        List of (x,y) tuples in original image coordinates
    output_width : int
        Target width
    output_height : int
        Target height
    debug : bool
        Display debug image.
    
    Returns
    -------
        tuple
            A tuple of (scaled_image, scaled_landmarks)
    """
    # Get original dimensions
    orig_height, orig_width = image.shape[:2]
    
    # Calculate scaling factors
    width_ratio = output_width / orig_width
    height_ratio = output_height / orig_height
    
    # Resize image (will stretch if aspect ratios differ)
    scaled_image = cv2.resize(image, (output_width, output_height))
    
    # Scale landmarks using the same ratios
    scaled_landmarks = [(int(x * width_ratio), int(y * height_ratio)) for x, y in landmarks]
    
    # Display debug image.
    if debug:
        _scaled_image = scaled_image.copy()
        for (px, py) in scaled_landmarks:
            cv2.circle(_scaled_image, (int(px), int(py)), 2, (0, 255, 255), -1)

        cv2.imshow("Scaled image and landmarks", _scaled_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    return scaled_image, scaled_landmarks


def quantify_blur(face_image : np.ndarray) -> float:
    """
    Quantifies how blurry a face is by using Laplacian
    variance to assign a value to the blur.

    NOTE: the face image should be tightly cropped so
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


def assess_head_direction(face_landmarks : List[int]) -> float:
    """
    Assess whether the head is facing forward or to the side.

    Returns True is the head is facing forward, otherwise False.

    NOTE: this function is NOT YET IMPLEMENTED!

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
