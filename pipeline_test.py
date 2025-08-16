import cv2
import os
import time

from pipeline import face_processing_pipeline



def static_image_test(image_path : str):
    """
    Tests the `face_processing_pipeline` function on a
    single static image.
    """
    # Read the image.
    image = cv2.imread(image_path)
    print(image_path)

    # Get the results.
    res = face_processing_pipeline(image=image,
                                   l=1,
                                   r=1,
                                   t=2.5,
                                   detector="mtcnn",
                                   desired_width=1080,
                                   desired_height=1920,
                                   face_mesh_margin=0.2,
                                   debug=True)
    
    if res:
        # Show all detected faces
        for i, face_data in enumerate(res):
            # Show the result data (except for the image and bounding box -- too much debug!)
            print(f"  Face {i}:")
            print(f"    prob : {face_data['prob']}")
            print(f"    blur : {face_data['blur']}")
            print(f"    head_forward : {face_data['head_forward']}")
            print(f"    original_face_width : {face_data['original_face_width']}")
            print(f"    original_face_height : {face_data['original_face_height']}")
            print("------")



            # Display the resulting processed face.
            cv2.imshow(f"processed face {i}", face_data["face_image"])
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
                                       t=1.5,
                                       detector="mtcnn",
                                       desired_width=1080,
                                       desired_height=1920,
                                       face_mesh_margin=0.20,
                                       debug=True)

        # Pause
        time.sleep(5)



if __name__ == "__main__":

    # Get all the test images.
    TEST_IMAGES_DIR = "test_images"
    test_images = [os.path.join(TEST_IMAGES_DIR, image)\
                   for image in os.listdir(TEST_IMAGES_DIR)]
    
    # Iterate through all the test images.
    # for image in test_images:
    #     static_image_test(image)

    # Then test the camera stream.
    camera_stream_test()
