from collections import deque
import cv2
import logging
import os
import time
import yaml
from _image_capture_utils import get_image
from _pipeline_utils import MarginConfig, FaceProcessor, FaceCriteria, is_face_acceptable
from _embedding_utils import embed_face, check_dataset_for_embedding
from _morph_utils import get_delauney_triangles, compute_intermediate_landmarks, \
    morph_align_face, alpha_blend, shift_vector
from _fragments_utils import fragment_image, create_composites



# Set up logging,
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format='%(levelname)s: %(message)s')

# Load the config file.
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Set the margins.
margins = MarginConfig(
    # 20% margin for face mesh
    face_mesh_margin=config["face_mesh_margin"],
    # Left margin = 1.5 × distance between pupils.
    pupil_left=config["pupil_left"],
    # Right margin = 1.5 × distance between pupils  
    pupil_right=config["pupil_right"],
    # Top margin = 1.0 × distance between pupils
    pupil_top=config["pupil_top"])

# Create a face processor object.
processor = FaceProcessor(
    # Detector type.
    detector_type=config["detector_type"],
    # Min confidence for face detection.
    face_mesh_conf=config["face_mesh_conf"],
    # Margins (see above).
    margins=margins,
    # Output width of final image, in pixels.
    desired_width=config["display_width"],
    # Output widheightth of final image, in pixels.
    desired_height=config["display_height"],
    # Whether to show debug on each step. Testing only!
    debug=False)

# Set the criteria for an acceptable face.
face_criteria = FaceCriteria(
    # The maximum bluriness allowed.
    max_blur = config["max_blur"],
    # The maximum yaw (orientation) of the head (abs).
    max_head_yaw = config["max_head_yaw"],
    # The maximum pitch (orientation) of the head (abs).
    max_head_pitch = config["max_head_pitch"],
    # The maximum roll (orientation) of the head (abs).
    max_head_roll=config["max_head_roll"],
    # The minimum confidence that this is a face.
    min_prob=config["min_prob"],
    # The minimum size of the face
    min_face_width=config["min_face_width"])

# Delauney triangulization indexes.
delauney_triangles = None


if __name__ == "__main__":
    logging.info("Setting up display")

    # Set the display.
    os.environ["DISPLAY"] = ":0"

    # Rotate the screen.
    # NOTE: this works for Pi only.
    os.system(f"wlr-randr --output HDMI-A-1 --transform {config['rotation']}")

    # Hide the cursor.
    os.system("unclutter -idle 0 &")

    # Make the display fullscreen.
    cv2.namedWindow(
        "Running Average",
        cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Running Average",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN)

    # Hold the fragmented faces in memory.
    fragmented_faces = deque(maxlen=config["face_memory"])

    # # Hold the landmarks of the fragmented faces in memory.
    # fragmented_faces_landmarks = deque(maxlen=config["face_memory"])

    # Hold the most recent N face embeddings in memory.
    previous_face_embeddings = deque(maxlen=config["face_memory"])

    # Get a standardized face and landmarks using the same processor object.
    standardized_image = cv2.imread("cam_smith.png")
    standardized_image_info = processor.process_image(standardized_image)
    if standardized_image_info:
        standardized_image_info = standardized_image_info[0]

        standardized_face_landmarks = standardized_image_info.landmarks_extra
    else:
        logging.warning("The standardized face is no good!")
        raise ValueError
    


    # Main event loop.
    while True:
        try:
            # Get a picture from the webcam.
            start = time.time()
            frame = get_image(config=config)
            logging.debug(f"Snapping photo: {time.time() - start:.3f}")

            # Process the image, extracting faces if they exist.
            start = time.time()
            processed_images_info = processor.process_image(frame)
            logging.debug(f"Processing image: {time.time() - start:.3f}")

            # If there are faces, proceed.
            if processed_images_info:
                # Log the start time for all processing
                processing_start_time = time.time()

                # Only consider the first face.
                # TODO: eventually expand to multiple faces.
                face_info = processed_images_info[0]

                # Save indexes to be reused.
                if delauney_triangles is None:
                    all_landmarks_ = face_info.landmarks + face_info.landmarks_extra
                    delauney_triangles = get_delauney_triangles(
                        landmark_coordinates=all_landmarks_,
                        image_height=face_info.image.shape[0],
                        image_width=face_info.image.shape[1])


                # Check if the face is acceptable.
                if is_face_acceptable(
                    face_info=face_info,
                    face_criteria=face_criteria):

                    # Check if we are tracking previous faces.
                    if config["face_memory_enabled"]:
                        # Embed the face as a vector.
                        start = time.time()
                        face_embedding = embed_face(face_info.image)
                        logging.info(f"Embedding face: {time.time() - start:.3f}")

                        # If the face can be encoded, proceed.
                        if face_embedding is not None:

                            # Check if the face is in the dataset. If it is, skip and continue the loop.
                            if not check_dataset_for_embedding(
                                face_embedding=face_embedding,
                                face_embedding_dataset=previous_face_embeddings,
                                tolerance=config["tolerance"]):

                                # Add it to the embeddings list.
                                previous_face_embeddings.append(face_embedding)

                            # Otherwise, continue the while loop.
                            else:
                                logging.info("Face seen before.")
                                continue


                    # Debug face information.
                    face_height = face_info.original_height
                    face_width = face_info.original_width
                    logging.info(f"Face resolution from original image: {face_height}x{face_width}")

                    # Morph align the new face to the new landmarks.
                    start = time.time()

                    try:
                        start = time.time()
                        morphed_face = morph_align_face(
                            source_face=face_info.image,
                            source_face_all_landmarks=face_info.landmarks + face_info.landmarks_extra,
                            target_face_all_landmarks=standardized_face_landmarks,
                            triangulation_indexes=None)
                        logging.info(f"Morph aligning new face: {time.time() - start:.3f}")

                    except Exception as e:
                        logging.exception(e)
                        continue

                    if (morphed_face is not None):
                        # Fragment the face.
                        start = time.time()

                        fragmented_face = fragment_image(
                            image=morphed_face,
                            grid_size=(config["squares_height"], config["squares_width"]))
                        fragmented_faces.append(fragmented_face)
                        # fragmented_faces_landmarks.append(
                        #     face_info.landmarks + face_info.landmarks_extra)

                        logging.info(f"Fragmenting the face: {time.time() - start:.3f}")

                        new_fragmented_frames = create_composites(
                            fragmented_images=fragmented_faces,  # type: ignore
                            num_composites=30)

                        for image in new_fragmented_frames:
                            cv2.imshow("Running Average", image)

                            if cv2.waitKey(100) & 0xFF == ord('q'):
                                logging.info("Shutting down...")
                                break

                        # Shift the standardized face landmarks slightly towards the new face.
                        if config["shift_landmarks"]:
                            standardized_face_landmarks = shift_vector(
                                input_1=standardized_face_landmarks,
                                input_2=face_info.landmarks + face_info.landmarks_extra,
                                alpha=0.1)

                        # Log the total processing time.
                        logging.info(
                            f"Total face processing time: {time.time() - processing_start_time:.3f}")
                        logging.info("----------------------------------------")


            # Exit if "q" is pressed.
            if cv2.waitKey(100) & 0xFF == ord('q'):
                logging.info("Shutting down...")
                break

        # Log all exceptions, but always keep running.
        except Exception as e:
            logging.warning("Exception logged! Continuing anyway")
            logging.exception(e)
            time.sleep(1)
