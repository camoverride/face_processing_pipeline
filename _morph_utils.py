from typing import List, Tuple, Optional
import random
import cv2
import numpy as np
import mediapipe as mp

from _deprecated_morph_utils import morph



# Initialize MediaPipe Face Detection
# TODO: is refine landmarks necessary? Does it use too much compute?
# Harmonize this with other face mesh calls.
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)
mp_face_detection = mp.solutions.face_detection.FaceDetection(  # type: ignore
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # type: ignore


def get_face_landmarks(image : np.ndarray) -> List[tuple[int, int]]:
    """
    Accepts an image and returns the landmarks for a face in the image.
    The image should contain a face, already cropped with a margin.

    Parameters
    ----------
    image : np.ndarray
        An image containing a face.
    
    Returns
    -------
    List[tuple[int, int]]
        A list of the tuples of all the landmark locations (x, y).
    """
    # Process the image to get the landmarks.
    results = mp_face_mesh.process(image)

    # Extract the facial landmarks.
    height, width, _ = image.shape
    facial_landmarks = []
    res = results.multi_face_landmarks

    # If there are landmarks, collect their coordinates.
    if res:
        for face_landmarks in res:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                facial_landmarks.append((x, y))

    # Else return an empty list
    else:
        return []

    return facial_landmarks


def get_additional_landmarks(
    image_height : int,
    image_width : int) -> List[tuple[int, int]]:
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
    List[tuple[int, int]]
        A list of the tuples of all the extra landmark locations (x, y).
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


def get_delauney_triangles(
    image_width : int,
    image_height : int,
    landmark_coordinates : List[tuple[int, int]]) -> np.ndarray:
    """
    Accepts an image along with landmark coordinates, which are a
    list of tuples. The landmarks can be just the face landmarks or
    all the landmarks, which will include points along the edge of
    the image, not just the face.

    Returns a list of lists, where every element of the list is
    6 long and contains the three coordinate pairs of every
    delauney triangle:
        [ [[x1, x2, y1, y2, z1, z2], ... ]

    NOTE: there will be more delauney triangles than points.

    Parameters
    ----------
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.
    landmark_coordinates : List[tuple[int, int]]
        A list of all the landmark coordiantes.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (N, 6), where each row contains the 
        coordinates of a Delaunay triangle: [x1, y1, x2, y2, x3, y3].
    """
    # Rectangle to be used with Subdiv2D
    rect = (0, 0, image_width, image_height)

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in landmark_coordinates:
        subdiv.insert(p)

    return subdiv.getTriangleList()  # type: ignore


def get_triangulation_indexes_for_landmarks(
    landmarks : List[tuple[int, int]],
    image_height : int,
    image_width : int) -> List:
    """
    Connect together all the landmarks into delauney triangles that
    span the image.

    Parameters
    ----------
    landmarks : List[tuple[int, int]]
        A list of coordinate pairs for every landmark.
    image_height : int
        The height of the image.
    image_width : int
        The width of the image. 

    Returns
    -------
    List[List[int]]
        The triangulation indexes. A list containing triplets:
        [[458, 274, 459], [465, 417, 464], ... ]
    """
    # Get the delauney triangles based off the landmarks.
    delauney_triangles = get_delauney_triangles(
        image_width,
        image_height,
        landmarks)

    # Convert these points into indexes.
    enumerated_rows = {}
    for index, row in enumerate(landmarks):
        enumerated_rows[str(list(row))] = index

    triangulation_indexes = []

    for x1, x2, y1, y2, z1, z2 in delauney_triangles:
        x = str(list([int(x1), int(x2)]))
        y = str(list([int(y1), int(y2)]))
        z = str(list([int(z1), int(z2)]))

        index_x = enumerated_rows[x]
        index_y = enumerated_rows[y]
        index_z = enumerated_rows[z]

        triangulation_indexes.append([index_x, index_y, index_z])

    return triangulation_indexes


def get_average_landmarks(target_landmarks_paths : list) -> List[tuple[int, int]]:
    """
    Accepts a list of image paths and extracts the landmarks from each
    image, averaging them together.

    Parameters
    ----------
    target_landmarks_paths : list
        A list of the paths to every face image. This can simply
        be a list of one path, or some subset of the dataset.

    Returns
    -------
    List[tuple[int, int]]
        The averaged landmarks.
    """
    # Collect all the landmarks here.
    all_landmarks = []

    # Read all the images.
    for face_path in target_landmarks_paths:
        face_image = cv2.imread(face_path)

        # Get all the landmarks.
        landmarks = get_face_landmarks(face_image)
        if landmarks is not None:
            all_landmarks.append(np.array(landmarks, dtype=np.float32))

    # Compute the average of all landmarks.
    average_landmarks = np.mean(all_landmarks, axis=0).astype(int)

    # Convert to a list
    average_landmarks_list = average_landmarks.tolist()

    return average_landmarks_list


def applyAffineTransform(
    src: np.ndarray, 
    srcTri: List[List[int]], 
    dstTri: List[List[int]], 
    size: Tuple[int, int]) -> np.ndarray:
    """
    Applies an affine transformation to an image region based on 
    corresponding triangle vertices.

    Given a source image region and a pair of corresponding triangles (one 
    in the source image and one in the destination image), this function 
    computes the affine transformation matrix and applies it to warp the 
    source patch onto the destination.

    Parameters
    ----------
    src : np.ndarray
        The source image patch that will be transformed.

    srcTri : List[List[int]]
        A list of three coordinate pairs representing the triangle in the 
        source image. Format: [[x1, y1], [x2, y2], [x3, y3]].

    dstTri : List[List[int]]
        A list of three coordinate pairs representing the corresponding 
        triangle in the destination image. Format: [[x1, y1], [x2, y2], [x3, y3]].

    size : Tuple[int, int]
        The dimensions (width, height) of the output image patch.

    Returns
    -------
    np.ndarray
        The warped image patch with the same dimensions as `size`.

    Notes
    -----
    - The transformation is computed using `cv2.getAffineTransform()`, which 
      finds a 2x3 matrix mapping `srcTri` to `dstTri`.
    - The transformation is applied using `cv2.warpAffine()` with bilinear 
      interpolation and border reflection to handle edge pixels.
    """
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))  # type: ignore

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src,
                         warpMat,
                         (size[0], size[1]),
                         None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morph_align_face(
    source_face : np.ndarray,
    source_face_all_landmarks : List[tuple[int, int]],
    target_face_all_landmarks : List[tuple[int, int]],
    triangulation_indexes: Optional[List]) -> Optional[np.ndarray]:
    """
    Accepts two images of the same dimensions containing faces.
    The features of the `source_face` are morphed so that they
    align with the landmarks of the `target_face`. Returns a morphed
    version of the `source_face`.

    This is done by extracting the landmarks from both faces and
    performing many affine transformations to change portions of
    the `source_face` to align with the `target_face`. These affine
    transformations assume a triangulation index, which is a division
    of all the landmarks into triangles, which are easy to mutate.

    Parameters
    ----------
    source_face : np.ndarray
        The face that will be morphed, having its features changed.
        Must be the same dimensions as `target_face`.
    source_face_all_landmarks : List[tuple[int, int]]
        The "foreground" landmarks which will be morphed onto the target.
        Must be the same dimensions as `target_face_all_landmarks`.        
    target_face_all_landmarks : List[tuple[int, int]]
        The "skeleton" landmarks onto which the source face will be mutated.
        Must be the same dimensions as `source_face_all_landmarks`.
    triangulation_indexes : list
        The list of triangles that span the entire image, used for
        morphing. Can be pre-computed, as it is not related to a
        specific face. It must have the same dimensions as `source_face`
        and `target_face`.

    Returns
    -------
    np.ndarray
        The "skin" from `source_face` morphed onto the landarks
        ("skeleton") of `target_face`.
    """
    # These must be equal!
    if len(source_face_all_landmarks) != len(target_face_all_landmarks):
        print(len(source_face_all_landmarks))
        print(len(target_face_all_landmarks))
        raise ValueError

    # Get the triangulation indexes for the target face.
    # NOTE: the image height/width is the same in all images, so it's taken
    # # from the source, even though the landmarks are from the target.
    if not triangulation_indexes:
        triangulation_indexes = \
            get_triangulation_indexes_for_landmarks(
                image_height=source_face.shape[0],
                image_width=source_face.shape[1],
                landmarks=target_face_all_landmarks)

    else:
        # Load the triangulation indexes.
        # TODO: implement this, save file to git
        pass

    # If there are landmarks, proceed:
    if source_face_all_landmarks and target_face_all_landmarks:
        # Leave space for final output
        morphed_face = np.zeros(source_face.shape, dtype=source_face.dtype)

        # Main event loop to morph triangles.
        for line in triangulation_indexes:
            # ID's of the triangulation points
            x = line[0]
            y = line[1]
            z = line[2]

            # Coordinate pairs
            t1 = [target_face_all_landmarks[x],
                  target_face_all_landmarks[y],
                  target_face_all_landmarks[z]]
            t2 = [source_face_all_landmarks[x],
                  source_face_all_landmarks[y],
                  source_face_all_landmarks[z]]

            r1 = cv2.boundingRect(np.float32([t1]))  # type: ignore
            r2 = cv2.boundingRect(np.float32([t2]))  # type: ignore
            r = cv2.boundingRect(np.float32([t1]))  # type: ignore

            # Offset points by left top corner of the respective rectangles
            t1Rect = []
            t2Rect = []
            tRect = []

            for i in range(0, 3):
                tRect.append(((t1[i][0] - r[0]), (t1[i][1] - r[1])))
                t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
                t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

            # Get the mask by filling triangles
            mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)  # type: ignore

            # Apply to small rectangular patches
            img2Rect = source_face[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

            size = (r[2], r[3])
            warped_image = applyAffineTransform(src=img2Rect,
                                                srcTri=t2Rect,
                                                dstTri=tRect,
                                                size=size)

            # Copy triangular region of the rectangular patch to the output image
            morphed_face[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = \
                morphed_face[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * \
                      (1 - mask) + warped_image * mask

        return morphed_face

    else:
        print("No face landmarks detected")


def get_average_face(image_paths : List[str]) -> np.ndarray:
    """
    Accepts a list of paths to some images and returns an average image.

    Parameters
    ----------
    image_paths : List[str]
        A list to image paths that will be averaged.

    Returns
    -------
    np.ndarray
        An averaged image.
    """
    # Initialize variables to store the sum of images and the count
    image_sum = None
    num_images = 0

    # Iterate over each image file
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            continue

        # Convert the image to float32 for accurate summation
        image = image.astype(np.float32)

        # Initialize the sum if this is the first image
        if image_sum is None:
            image_sum = np.zeros_like(image, dtype=np.float32)

        # Add the image to the sum
        image_sum += image
        num_images += 1

    # Check if any images were processed
    if num_images == 0 or image_sum == None:
        raise ValueError("No valid images found in the directory.")

    # Compute the average image
    averaged_image = image_sum / num_images

    # Convert back to uint8 for saving/displaying
    averaged_image = averaged_image.astype(np.uint8)

    return averaged_image


def generate_continuous_morphs(
    image_1 : np.ndarray,
    image_2 : np.ndarray,
    triangulation_indexes : list,
    num_transitional_morphs : int) -> list[np.ndarray]:
    """
    Takes two images containing faces with the same dimensions
    and creates a series of transitional images that morph between
    them. These faces should not be morph-aligned, but if they have
    their pupils aligned that's OK.

    TODO: this function uses a deprecated `morph` function that should
    eventually be integrated with `morph_align_face` by re-adding the
    `alpha` parameter to this function. In `deprecated_morph_utils.py`

    TODO: rewrite this as a generator so the images do not accumulate
    in memory. However, if `num_transitional_morphs` is small then
    we don't need to worry about memory overflow.

    Parameters
    ----------
    image_1 : np.ndarray
        The starting image for the morph series.
    image_2 : np.ndarray
        The ending image for the morph seires.
    triangulation_indexes : list
        The indexes used for morphing triangles.
        TODO: check the type!
    num_transitional_morphs : int
        The number of transitional images that will morph
        between `image_1` and `image_2`.

    Returns
    -------
    None
        Writes N images to a directory, where N = `num_transitional_morphs`
    """
    if num_transitional_morphs <= 0:
        raise ValueError

    # Collect the partial morphs
    partial_morphs_1 = []
    partial_morphs_2 = []

    # Get the landmarks
    landmarks_1 = get_face_landmarks(image_1)
    additional_landmarks_1 = get_additional_landmarks(
        image_height=image_1.shape[0],
        image_width=image_1.shape[1])
    all_landmarks_1 = landmarks_1 + additional_landmarks_1

    landmarks_2 = get_face_landmarks(image_2)
    additional_landmarks_2 = get_additional_landmarks(
        image_height=image_2.shape[0],
        image_width=image_2.shape[1])
    all_landmarks_2 = landmarks_2 + additional_landmarks_2

    # Use various values of alpha
    alphas = np.linspace(0, 1, num_transitional_morphs).tolist()
    for alpha in alphas:
        # Morph-align the face to the target face.
        morphed_face_1 = morph(
            image_2,
            image_1,
            all_landmarks_2,
            all_landmarks_1,
            triangulation_indexes,
            alpha)
        
        morphed_face_2 = morph(
            image_1,
            image_2,
            all_landmarks_1,
            all_landmarks_2,
            triangulation_indexes,
            alpha)

        partial_morphs_1.append(morphed_face_1)
        partial_morphs_2.append(morphed_face_2)

    # Reverse this list for morphing.
    partial_morphs_1 = list(reversed(partial_morphs_1))

    # Collect the blended faces.
    blended_faces = []

    # Reverse the alphas for blending.
    alphas = list(reversed(alphas))

    # Blend together all the faces.
    for i, alpha in enumerate(alphas):
        blended_face = cv2.addWeighted(
            partial_morphs_1[i],
            alpha, partial_morphs_2[i],
            1 - alpha,
            0)
        blended_faces.append(blended_face)
    
    return blended_faces


def create_composite_image(
    image_list : List[np.ndarray],
    num_squares_height : int) -> np.ndarray :
    """
    Accepts a list of images and desired number of squares
    (along the vertical margin) and creates a composite image
    from them.

    Parameters
    ----------
    image_list : List[np.ndarray]
        A list of images encoded as numpy arrays.
    num_squares_height : int,
        The number of squares to tile the vertical of the image.
    
    Returns
    -------
    np.ndarray
        A composite image encoded as a numpy array.
    """
    # Get the height/width from a random image.
    image_height = image_list[0].shape[0]
    image_width = image_list[0].shape[1]

    # Check that the images all have the same shape
    if len(set((img.shape for img in image_list))) != 1:
        raise ValueError("All images must have the same dimensions.")

    # Set the image dimension info.
    crop_width = image_width - (image_width % num_squares_height)
    crop_height = image_height - (image_height % num_squares_height)
    image_list = [img[:crop_height, :crop_width] for img in image_list]

    square_size = crop_height // num_squares_height
    num_squares_width = crop_width // square_size

    # Generate the individual squares.
    # TODO: this should also be a memmap
    squares = [[[] for _ in range(num_squares_width)] for _ in range(num_squares_height)]
    for img in image_list:
        for i in range(num_squares_height):
            for j in range(num_squares_width):
                top = i * square_size
                left = j * square_size
                square = img[top:top + square_size, left:left + square_size]
                squares[i][j].append(square)

    # Combine the squares into an image.
    composite_image = np.zeros_like(image_list[0][:crop_height, :crop_width])
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            selected_square = random.choice(squares[i][j])
            top = i * square_size
            left = j * square_size
            composite_image[top:top + square_size, left:left + square_size] = selected_square

    return composite_image


def compute_intermediate_landmarks(
        landmarks_1: list[tuple[int, int]],
        landmarks_2: list[tuple[int, int]],
        alpha: float
    ) -> list[tuple[int, int]]:
    """
    Compute intermediate landmarks by linearly blending two sets of landmarks.

    Each landmark in the result is calculated as a weighted average of the 
    corresponding landmarks from `landmarks_1` and `landmarks_2` using the
    blending factor `alpha`.

    Formula:
        intermediate = alpha * landmarks_1 + (1 - alpha) * landmarks_2

    Parameters
    ----------
    landmarks_1 : list of tuple of int
        First set of landmarks. Each landmark is a tuple (x, y).
    landmarks_2 : list of tuple of int
        Second set of landmarks. Must have the same length as `landmarks_1`.
    alpha : float
        Blending factor. Must be between 0.0 and 1.0.
        - alpha=1.0 => result is exactly `landmarks_1`
        - alpha=0.0 => result is exactly `landmarks_2`
        - intermediate values blend the two sets proportionally

    Returns
    -------
    list of tuple of int
        List of blended landmarks, same length as the input lists.
    
    Raises
    ------
    ValueError
        If `landmarks_1` and `landmarks_2` are not the same length or if
        alpha is not between 0.0 and 1.0.
    """

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    if len(landmarks_1) != len(landmarks_2):
        raise ValueError("landmarks_1 and landmarks_2 must have the same length")

    # Compute blended landmarks
    intermediate_landmarks = []
    for (x1, y1), (x2, y2) in zip(landmarks_1, landmarks_2):
        x = int(round(alpha * x1 + (1 - alpha) * x2))
        y = int(round(alpha * y1 + (1 - alpha) * y2))
        intermediate_landmarks.append((x, y))

    return intermediate_landmarks


def alpha_blend(
    background_image : np.ndarray,
    foreground_image : np.ndarray,
    alpha : float) -> np.ndarray:
    """
    Simple alpha blending of two images.

    Parameters
    ----------
    image_1 : np.ndarray
        First image (background).
    image_2 : np.ndarray
        Second image (foreground).
    alpha : float
        blending factor (0.0 to 1.0)
        0.0 = only image_1, 1.0 = only image_2

    Returns
    -------
    np.ndarray
        blended image
    """
    # Ensure images have the same dimensions
    if background_image.shape != foreground_image.shape:
        raise ValueError

    # Perform alpha blending.
    blended = cv2.addWeighted(
        background_image,
        1 - alpha,
        foreground_image, alpha,
        0)

    return blended


def shift_vector(
    input_1 : list[tuple[int, int]],
    input_2 : list[tuple[int, int]],
    alpha : float) -> list[tuple[int, int]]:
    """
    Shift input_1 towards input_2 by a factor of alpha.

    Parameters
    ----------
    input_1 : list[tuple[int, int]]
        List of face landmarks.
    input_2 : list[tuple[int, int]]
        List of face landmarks.
    alpha : float
        (0.0 = no change, 1.0 = fully input_2)

    Returns
    -------
    list[tuple[int, int]]
        input_1 shifted towards input_2
    """
    arr1 = np.array(input_1)
    arr2 = np.array(input_2)

    shifted = arr1 + alpha * (arr2 - arr1)

    return [tuple(point) for point in shifted]
