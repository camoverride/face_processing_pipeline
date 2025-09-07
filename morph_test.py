import cv2
import numpy as np
from _morph_utils import get_face_landmarks, get_additional_landmarks, \
  morph_align_face, get_delauney_triangles, get_triangulation_indexes_for_landmarks, \
  compute_intermediate_landmarks



if __name__ == "__main__":

    # Read the images.
    image_1 = cv2.imread("1.jpg")
    image_2 = cv2.imread("2.jpg")

    assert isinstance(image_1, np.ndarray), "image_1 was not loaded correctly or is not a NumPy array"
    assert isinstance(image_2, np.ndarray), "image_1 was not loaded correctly or is not a NumPy array"

    # Get the landmarks.
    face_landmarks_1 = get_face_landmarks(image_1)
    face_landmarks_2 = get_face_landmarks(image_2)

    # Get additional landmarks around the edges of the image.
    additional_landmarks_1 = get_additional_landmarks(
        image_height=image_1.shape[0],
        image_width=image_1.shape[1])
    all_landmarks_1 = face_landmarks_1 + additional_landmarks_1

    additional_landmarks_2 = get_additional_landmarks(
        image_height=image_2.shape[0],
        image_width=image_2.shape[1])
    all_landmarks_2 = face_landmarks_2 + additional_landmarks_2

    # Draw landmarks on both images.
    image_1_copy = image_1.copy()
    for landmark in all_landmarks_1:
        x, y = landmark
        cv2.circle(image_1_copy, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    image_2_copy = image_2.copy()
    for landmark in all_landmarks_2:
        x, y = landmark
        cv2.circle(image_2_copy, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Show both images with landmarks.
    cv2.imshow("(1) Landmarks", image_1_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("(2) Landmarks", image_2_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw Delauney triangles on both images
    delauney_triangles_1 = get_delauney_triangles(
        image_height=image_1_copy.shape[0],
        image_width=image_1_copy.shape[1],
        landmark_coordinates=all_landmarks_1)

    for triangle in delauney_triangles_1:
        pts = np.array(triangle, dtype=np.int32).reshape(3, 2)  # Reshape into (3,2) format
        cv2.polylines(image_1_copy, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

    delauney_triangles_2 = get_delauney_triangles(
        image_height=image_2_copy.shape[0],
        image_width=image_2_copy.shape[1],
        landmark_coordinates=all_landmarks_2)

    for triangle in delauney_triangles_1:
        pts = np.array(triangle, dtype=np.int32).reshape(3, 2)  # Reshape into (3,2) format
        cv2.polylines(image_2_copy, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

    # Show both images with Delaunay triangles.
    cv2.imshow("(1) Delaunay Triangulation", image_1_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("(2) Delaunay Triangulation", image_2_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Precompute triangulation indexes
    triangulation_indexes = get_triangulation_indexes_for_landmarks(
        landmarks=all_landmarks_1,
        image_height=image_1.shape[0],
        image_width=image_1.shape[1])

    # Morph face 1 onto face 2.
    morphed_face_1 = morph_align_face(
        source_face=image_1,
        target_face_all_landmarks=all_landmarks_2,
        triangulation_indexes=triangulation_indexes)

    # Morph face 2 onto face 1.
    morphed_face_2 = morph_align_face(
        source_face=image_2,
        target_face_all_landmarks=all_landmarks_1,
        triangulation_indexes=triangulation_indexes)

    if morphed_face_1 is not None:
        # Display the morphs.
        cv2.imshow("(1) Face 1 morphed onto face 2", morphed_face_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError

    if morphed_face_2 is not None:
        cv2.imshow("2) Face 2 morphed onto face 1", morphed_face_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError

    # Create intermediate alpha blends
    for alpha in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        new_landmarks = compute_intermediate_landmarks(
            landmarks_1=all_landmarks_1,
            landmarks_2=all_landmarks_2,
            alpha=alpha)

        # Morph face 1 onto face 2.
        morphed_face_x = morph_align_face(
            source_face=image_1,
            target_face_all_landmarks=new_landmarks,
            triangulation_indexes=triangulation_indexes)

        if morphed_face_x is not None:
            cv2.imshow(f"(1) Face 1 morphed onto face 2 (alpha: {alpha})", morphed_face_x)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            raise ValueError
