from collections import deque
import cv2
import numpy as np
from numpy.typing import NDArray
import face_recognition



def embed_face(face_image: np.ndarray) -> NDArray[np.float64]:
    """
    Compute an embedding for the given face image using
    face_recognition library.

    Parameters
    ----------
    face_image : np.ndarray
        The face image as a NumPy array (as returned by OpenCV)
        in BGR format.

    Returns
    -------
    NDArray[np.float64]
        A 128-dimensional face embedding vector, or empty array
        if no face found.
    """
    # Convert from BGR to RGB (OpenCV to face_recognition format)
    rgb_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB)

    # Get face embeddings
    embeddings = face_recognition.face_encodings(rgb_image)

    if embeddings:
        return embeddings[0]  # Return the first face embedding found
    else:
        return np.array([], dtype=np.float64)  # Return empty array if no face


def check_dataset_for_embedding(
    face_embedding: NDArray[np.float64],
    face_embedding_dataset: deque[NDArray[np.float64]],
    tolerance: float = 0.6) -> bool:
    """
    Check whether a given embedding is present in the dataset
    using face recognition comparison.

    Parameters
    ----------
    face_embedding : NDArray[np.float64]
        The query embedding to check (128-dimensional vector).
    face_embedding_dataset : deque[NDArray[np.float64]]
        A deque of embeddings representing the dataset.
    tolerance : float, optional
        How much distance between faces to consider it a match.
        Lower is more strict.
        NOTE: default is 0.6 -- typical for face recognition.

    Returns
    -------
    bool
        True if a matching embedding is found in the dataset.
        False otherwise.
    """
    if face_embedding.size == 0 or len(face_embedding_dataset) == 0:
        return False

    # Convert deque to list for face_recognition comparison.
    dataset_list = list(face_embedding_dataset)

    # Compare the query embedding against all embeddings in the dataset.
    matches = face_recognition.compare_faces(
        dataset_list, 
        face_embedding, 
        tolerance=tolerance)

    return any(matches)
