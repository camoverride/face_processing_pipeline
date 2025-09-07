from dataclasses import dataclass
import cv2
from facenet_pytorch import MTCNN
import logging
import mediapipe as mp
from mediapipe.python.solutions import face_detection
import numpy as np
import os
import sys
import torch
from typing import List, Tuple, Optional
import _face_pipeline_utils



# Suppress unwanted logs.
os.environ["QT_LOGGING_RULES"] = "*=false"


# Set up logging.
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s')


# Data class for constructing margins.
@dataclass
class MarginConfig:
    # NOTE: K is the distance between pupils.
    # The margin added to the face_mesh as a fraction of the face's width.
    face_mesh_margin: float
    # The distance, in units of K, between the left pupil and the margin.
    pupil_left: float
    # The distance, in units of K, between the right pupil and the margin.
    pupil_right: float
    # The distance, in units of K, between the pupils and the top margin.
    pupil_top: float


# Data class for collecting face results.
@dataclass
class FaceResult:
    # The resulting cropped and rotated image.
    image: np.ndarray
    # The corresponding face landmarks.
    landmarks: List[Tuple[int, int]]
    # The extra landmarks around the edge of the image.
    landmarks_extra: List[Tuple[int, int]]
    # The blurriness of the face.
    blur: float
    # The yaw (orientation) of the head.
    head_yaw: float
    # The pitch (orientation) of the head.
    head_pitch: float
    # The roll (orientation) of the head.
    head_roll: float
    # The width of the original image.
    original_width: int
    # The height of the original image.
    original_height: int
    # The confidence that this is a face.
    prob: float


# Data class for determining whether a face is adequate.
@dataclass
class FaceCriteria:
    # The maximum bluriness allowed.
    max_blur: float
    # The maximum yaw (orientation) of the head (abs).
    max_head_yaw: float
    # The maximum pitch (orientation) of the head (abs).
    max_head_pitch: float
    # The maximum roll (orientation) of the head (abs).
    max_head_roll: float
    # The minimum confidence that this is a face.
    min_prob: float


class FaceProcessor:
    def __init__(
        self, 
        detector_type: str,
        face_mesh_conf: float,
        margins: MarginConfig,
        desired_width: int,
        desired_height: int,
        debug: bool):
        """
        Initialize face processor with fixed processing parameters.
        
        Parameters
        ----------
        detector_type : str
            Type of face detector. Current options:
                - "mtcnn"
                - "mediapipe"
        face_mesh_conf : float
            Confidence threshold for face mesh detection.
        margins : MarginConfig
            Configuration for face cropping margins.
        desired_width : int
            Output image width in pixels.
        desired_height : int
            Output image height in pixels.
        debug : bool
            Whether to show debug visualizations.
        """
        # Initialize detectors.
        self.detector = self._init_detector(detector_type)
        self.face_mesh = self._init_face_mesh(face_mesh_conf)

        # Store other configs.
        self.detector_type = detector_type
        self.face_mesh_conf = face_mesh_conf
        self.margins = margins
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.debug = debug


    def _init_detector(
            self,
            detector_type: str):
        """
        Initialize face detector.

        Parameters
        ----------
        detector_type : str
            The type of detector.
            NOTE: currently only "mtcnn" is implemented.
        """
        if detector_type == "mtcnn":
            device = torch.device("cpu")
            detector = MTCNN(keep_all=True, device=device)
            # Move subnets to device
            for net in [detector.pnet, detector.rnet, detector.onet]:
                if net is not None:
                    net.to(device)
            return detector
        
        elif detector_type == "mediapipe":
            # Create a MediaPipe detector wrapper.
            return MediaPipeDetector()
    
        raise ValueError(f"Unsupported detector type: {detector_type}")


    def _init_face_mesh(
            self,
            min_detection_confidence : float):
        """
        Initialize MediaPipe FaceMesh.

        Parameters
        ----------
        min_detection_confidence : float
            The mininum confidence to detect a face.
        """
        return mp.solutions.face_mesh.FaceMesh(# type: ignore
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence)
    

    def __del__(self):
        """
        Clean up resources.
        """
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
    

    def process_image(
        self,
        image : np.ndarray) -> Optional[list[FaceResult]]:
        """
        This accepts a picture that may or may not contain faces and
        extracts and processes all the faces so that they are in a
        standardized format. This format means that the eyeballs are in
        the same relative position in each image, and each image is scaled
        to be the same shape. This function returns a FaceResult dataclass
        that contains this processed image along with some additional metadata
        about the image.

        If no faces are detected, the function returns None.

        Parameters
        ----------
        image : np.ndarray
            The input image that might contain faces.

        Returns
        -------
        FaceResult
            A dataclass containing the image and information about the face.
        None
            If no faces are detected in the image.
        """
        # The image might contain multiple faces. Collect them all here.
        processed_face_data = []

        # Validate the image. Throws errors if invalid.
        _face_pipeline_utils.validate_image(image)

        # Detect faces in the image and get bounding boxes.
        boxes, probs = _face_pipeline_utils.detect_faces(
            image=image,
            detector=self.detector,
            debug=self.debug)

        # If there are no faces, return None.
        if (boxes is None) or (len(boxes) == 0) or (probs is None):
            logging.debug("No faces detected!")
            return None

        # If faces are detected, iterate through all of them.
        # If any of these steps fails, `continue` to the next image.
        for box, prob in zip(boxes, probs):

            # Get a copy of this face with no margin.
            no_margin_face = _face_pipeline_utils.get_no_margin_face(
                image=image,
                box=box,
                debug=self.debug)

            if no_margin_face is None:
                logging.info("> Could not crop face [no margin]!")
                continue

            # Get a copy of this face with a small margin.
            small_margin_face, new_bb, margin = _face_pipeline_utils.get_small_margin_face(
                image=image,
                box=box,
                face_mesh_margin=self.margins.face_mesh_margin,
                debug=self.debug)

            if small_margin_face is None:
                logging.info("> Could not crop face [small margin]!")
                continue

            # Check if the newly cropped face overflows the image
            overflow = _face_pipeline_utils.does_new_crop_overflow_image(
                image=image,    
                bb=new_bb)

            if overflow:
                logging.info("> Newly cropped face overflows original image. Skipping!")
                continue

            # Get the face_mesh landmarks from the small_margin_face
            face_mesh_landmarks = _face_pipeline_utils.get_face_mesh(
                face_mesh_instance=self.face_mesh,
                face_image=small_margin_face,
                debug=self.debug)

            if face_mesh_landmarks is None:
                logging.info("> No face mesh landmarks!")
                continue

            # Reproject the face_mesh from the `small_margin_face` onto the entire image.
            reprojected_landmarks = _face_pipeline_utils.reproject_landmarks(
                image=image,
                cropped_face=small_margin_face,
                box=box,
                face_mesh_landmarks=face_mesh_landmarks,
                margin=margin,
                debug=self.debug)

            # Use this mesh to rotate the image.
            rotated_image, rotated_landmarks = _face_pipeline_utils.rotate_face(
                image=image,
                landmarks=reprojected_landmarks,
                debug=self.debug)

            if rotated_image is None:
                logging.info("> Could not rotate image!")
                continue

            # Use the rotated image and rotated mesh to crop to eyeballs.
            pupil_cropped_face, pupil_cropped_landmarks = _face_pipeline_utils.pupil_crop_image(
                image=rotated_image,
                landmarks=rotated_landmarks,
                l=self.margins.pupil_left,
                r=self.margins.pupil_right,
                t=self.margins.pupil_top,
                desired_width=self.desired_width,
                desired_height=self.desired_height,
                debug=self.debug)

            if (pupil_cropped_face is None) or (pupil_cropped_landmarks is None):
                logging.info("> Could not pupil crop image")
                continue

            # Get the all the landmarks required for later processing functions.
            h, w = pupil_cropped_face.shape[:2]
            additional_landmarks = _face_pipeline_utils.get_additional_landmarks(h, w)

            # NOTE: Not used. Instead, these landmarks are returned separately. Used for debugging.
            _all_landmarks = _face_pipeline_utils.combine_landmarks(
                pupil_cropped_face=pupil_cropped_face,
                face_landmarks=pupil_cropped_landmarks,
                additional_landmarks=additional_landmarks,
                debug=self.debug)

            # Assess the blur using the no-margin face.
            blur = _face_pipeline_utils.quantify_blur(face_image=no_margin_face)

            # Assess the head direction using the scaled image landmarks.
            head_direction = _face_pipeline_utils.assess_head_direction(
                face_landmarks=pupil_cropped_landmarks)        

            # Collect all the face info.
            face_data = FaceResult(
                image = pupil_cropped_face,
                landmarks = pupil_cropped_landmarks,
                landmarks_extra = additional_landmarks,
                blur = blur,
                head_yaw = head_direction["yaw"],
                head_pitch = head_direction["pitch"],
                head_roll = head_direction["roll"],
                original_width = no_margin_face.shape[1],
                original_height = no_margin_face.shape[0],
                prob = prob)
            
            processed_face_data.append(face_data)
        
        # After all the faces are collected into a list of `FaceResult` objects, return.
        return processed_face_data


def is_face_acceptable(
    face_info : FaceResult,
    face_criteria : FaceCriteria) -> bool:
    """
    Compares information from a FaceResult object to a 
    FaceCriteria object to see if a face is acceptable for
    processing.

    Parameters
    ----------
    face_info : FaceResult
        A data class produced when a face is successfully processed.
    face_criteria : FaceCriteria
        A data class that sets standards for whether a face should
        be acceptable.
    """
    if face_info.blur < face_criteria.max_blur \
        and abs(face_info.head_yaw) < face_criteria.max_head_yaw \
        and abs(face_info.head_pitch) < face_criteria.max_head_pitch \
        and abs(face_info.head_roll) < face_criteria.max_head_roll \
        and face_info.prob > face_criteria.min_prob:

        return True

    else:
        return False


class MediaPipeDetector:
    """
    Wrapper around mediapipe.solutions.face_detection to provide an API
    compatible with facenet_pytorch.MTCNN.detect(image) -> (boxes, probs).

    - Exposes detect(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]
      where boxes are (N, 4) in pixel coords [x1, y1, x2, y2] and probs is (N,).
    - Accepts OpenCV BGR images (np.ndarray).
    - Provide close() to free resources.
    """

    def __init__(self,
                 model_selection: int = 1,
                 min_detection_confidence: float = 0.5,
                 keep_all: bool = True):
        """
        Parameters
        ----------
        model_selection : int
            0 for short-range, 1 for full-range (same naming as mediapipe).
        min_detection_confidence : float
            Min confidence to report a detection.
        keep_all : bool
            If True, return all detections. If False, return only the highest-scoring detection.
        """
        self.keep_all = keep_all
        # Create mediapipe face detection instance
        self._mp_fd = face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence)  # type: ignore

    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray
            OpenCV BGR image of shape (H, W, 3) or grayscale (H, W).

        Returns
        -------
        boxes : np.ndarray or None
            Array of shape (N, 4) with boxes in pixel coords [x1, y1, x2, y2], dtype=float32.
            None if no detections.
        probs : np.ndarray or None
            Array of shape (N,) with detection scores. None if no detections.
        """
        if image is None:
            return None, None

        # If grayscale, convert to 3-channel
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # MediaPipe expects RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Process image: note MediaPipe expects uint8 or normalized; passing uint8 is fine.
        results = self._mp_fd.process(image_rgb)

        if not results or not results.detections:
            return None, None

        boxes_list = []
        scores_list = []

        for detection in results.detections:
            # score: detection.score is a list (usually a single score)
            score = float(detection.score[0]) if detection.score else 0.0

            # bounding box (relative)
            rbbox = detection.location_data.relative_bounding_box
            if rbbox is None:
                continue

            # Convert relative bbox to absolute pixel coordinates
            xmin = rbbox.xmin * w
            ymin = rbbox.ymin * h
            width = rbbox.width * w
            height = rbbox.height * h

            # Sometimes MediaPipe returns negative xmin/ymin or width/height that go out of bounds.
            x1 = float(xmin)
            y1 = float(ymin)
            x2 = float(xmin + width)
            y2 = float(ymin + height)

            # Clip to image
            x1 = max(0.0, min(x1, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            x2 = max(0.0, min(x2, w - 1.0))
            y2 = max(0.0, min(y2, h - 1.0))

            # If bbox is degenerate, skip
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(score)

        if len(boxes_list) == 0:
            return None, None

        boxes = np.asarray(boxes_list, dtype=np.float32)
        probs = np.asarray(scores_list, dtype=np.float32)

        # If keep_all is False, return only the highest-scoring detection (like some usages of MTCNN)
        if (not self.keep_all) and boxes.shape[0] > 1:
            idx = int(np.argmax(probs))
            boxes = boxes[idx:idx+1]
            probs = probs[idx:idx+1]

        return boxes, probs

    def close(self):
        """Close the mediapipe resources."""
        if hasattr(self, "_mp_fd") and self._mp_fd is not None:
            self._mp_fd.close()
            self._mp_fd = None

    # Provide context-manager compatibility and destructor safety.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
