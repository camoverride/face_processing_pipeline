import cv2
import os
from _pipeline_utils import MarginConfig, FaceProcessor



def static_image_test(
    image_path : str,
    processor : FaceProcessor):
    """
    Tests the `face_processing_pipeline` function on a
    single static image.
    """
    # Read the image.
    image = cv2.imread(image_path)
    print(image_path)

    # Call the pipeline.
    processed_images_info = processor.process_image(image)

    # Check that a face was detected.
    if processed_images_info:

        # Iterate through every face that as detected.
        for processed_image_info in processed_images_info:
        
            # Print out debug.
            print(f"Blur : {processed_image_info.blur}")
            print(f"Head yaw : {processed_image_info.head_yaw}")
            print(f"Head pitch : {processed_image_info.head_pitch}")
            print(f"Head roll : {processed_image_info.head_roll}")
            print(f"Original height : {processed_image_info.original_height}")
            print(f"Original width : {processed_image_info. original_width}")
            print(f"Confidence : {processed_image_info.prob}")
            print("")

            # Show the actual image.
            cv2.imshow("Processed Image", processed_image_info.image)

            # Pause after each frame.
            print("##########")
            if cv2.waitKey(1000) & 0xFF == ord('q'):  # Press 'q' to quit
                break

            # Save a couple of example cropped images.
            if image_path in ["test_images/1.jpg", "test_images/2.jpg"]:
                cv2.imwrite(f"{image_path.split('/')[-1]}", processed_image_info.image)


def camera_stream_test(
    processor : FaceProcessor):
    """
    Tests the `face_processing_pipeline` function on a
    webcam stream.
    """
    # Start video stream.
    cap = cv2.VideoCapture(0)

    # Check if the stream is open.
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Video processing loop.
    while True:
        # Capture frame-by-frame.
        ret, frame = cap.read()

        # If frame reading was not successful, break.
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Call the pipeline.
        processed_images_info = processor.process_image(frame)

        # Check that a face was detected.
        if processed_images_info:

            # Iterate through every face that as detected.
            for processed_image_info in processed_images_info:
            
                # Print out debug.
                print(f"Blur : {processed_image_info.blur}")
                print(f"Head yaw : {processed_image_info.head_yaw}")
                print(f"Head pitch : {processed_image_info.head_pitch}")
                print(f"Head roll : {processed_image_info.head_roll}")
                print(f"Original height : {processed_image_info.original_height}")
                print(f"Original width : {processed_image_info. original_width}")
                print(f"Confidence : {processed_image_info.prob}")
                print("")

                # Show the actual image.
                cv2.imshow("Processed Image", processed_image_info.image)

        # Pause after each frame.
        print("##########")
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break



if __name__ == "__main__":

    # Set the margins
    margins = MarginConfig(
        # 20% margin for face mesh
        face_mesh_margin=0.20,
        # Left margin = 1.5 × distance between pupils.
        pupil_left=1.5,
        # Right margin = 1.5 × distance between pupils  
        pupil_right=1.5,
        # Top margin = 1.0 × distance between pupils
        pupil_top=3.0)


    # Create a face processor object.
    processor = FaceProcessor(
        # Detector type.
        detector_type="mtcnn",
        # Min confidence for face detection.
        face_mesh_conf=0.7,
        # Margins (see above).
        margins=margins,
        # Output width of final image, in pixels.
        desired_width=1080,
        # Output widheightth of final image, in pixels.
        desired_height=1920,
        # Whether to show debug on each step. Testing only!
        debug=False)


    # Get all the test images.
    TEST_IMAGES_DIR = "test_images"
    test_images = [os.path.join(TEST_IMAGES_DIR, image)\
                   for image in os.listdir(TEST_IMAGES_DIR)]


    # First iterate through all the test images.
    for image_path in test_images:
        static_image_test(
            image_path=image_path,
            processor=processor)


    # Then test the camera stream.
    camera_stream_test(processor=processor)
