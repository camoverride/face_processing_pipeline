import os
import time

import cv2

from pipeline import face_processing_pipeline



def static_image_test(image_path : str):
    """
    Tests the `face_processing_pipeline` function on a
    single static image.
    """
    # Read the image.
    image = cv2.imread(image_path)

    # Get the results.
    res = face_processing_pipeline(image=image,
                                   l=1,
                                   r=1,
                                   t=2,
                                   b=2,
                                   detector="mtcnn",
                                   output_width=1200,
                                   output_height=1200,
                                   face_mesh_margin=70,
                                   debug=True)
    
    if res:
        # Show the result data.
        print(res)

        # Display the resulting processed face.
        cv2.imshow("processed face", res["face_image"])
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


def camera_stream_test():
    """
    Tests the `face_processing_pipeline` function on a
    webcam stream.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame reading was not successful, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Call the pipeline.
        res = face_processing_pipeline(image=frame,
                                       l=1,
                                       r=1,
                                       t=2,
                                       b=2,
                                       detector="mtcnn",
                                       output_width=1200,
                                       output_height=1200,
                                       face_mesh_margin=70,
                                       debug=True)
        
        # Print the results
        print(res)

        # Pause
        time.sleep(5)



if __name__ == "__main__":

    # Get all the test images.
    TEST_IMAGES_DIR = "test_images"
    test_images = [os.path.join(TEST_IMAGES_DIR, image)\
                   for image in os.listdir(TEST_IMAGES_DIR)]
    
    # Iterate through all the test images.
    for image in test_images:
        print(f"Testing image: {image}")
        static_image_test(image)
