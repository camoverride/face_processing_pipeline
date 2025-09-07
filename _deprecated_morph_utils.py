import cv2
import numpy as np



def compute_average_coordinates(landmarks_1, landmarks_2, alpha=0):
    """
    NOTE: all current applications have alpha=0, meaning this function
    simply returns `landmarks_1`


    Computes the weighted average coordinates of 2 equal-length lists of tuples,
    `landmarks_1`, `landmarks_2`.

    Returns a list of lists, where each sub-list are the average coordinates.

    Alpha determines the amount of blending. For example:
        - alpha = 0 : returns `landmarks_1`
        - alpha = 0.25 : more similar to `landmarks_1`
        - alpha = 0.5 : equally `landmarks_1` and `landmarks_2`
        - alpha = 1 : returns `landmarks_2`

    TODO: this is not necessary for full morph!!!!
    """
    average_points = []

    if alpha == 0:
        return landmarks_1
    elif alpha == 1:
        return landmarks_2

    # Compute weighted average point coordinates
    for i in range(0, len(landmarks_1)):  # edit to index list within list
        x = (1 - alpha) * landmarks_1[i][0] + alpha * landmarks_2[i][0]
        y = (1 - alpha) * landmarks_1[i][1] + alpha * landmarks_2[i][1]
        average_points.append([x, y])

    return average_points


def applyAffineTransform(src, srcTri, dstTri, size):
    """
    Applies an affine transformation.

    Apply affine transform calculated using srcTri and dstTri to src

    TODO: properly document this function.
    """
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( # type: ignore
        np.float32(srcTri),           # type: ignore
        np.float32(dstTri))           # type: ignore

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, margin, alpha=1):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img

    If alpha=1, keep the skin of img2

    TODO: properly document this function.
    """
    alpha = 1
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1])) # type: ignore
    r2 = cv2.boundingRect(np.float32([t2])) # type: ignore
    r = cv2.boundingRect(np.float32([t])) # type: ignore

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0) # type: ignore

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(src=img1Rect, srcTri=t1Rect, dstTri=tRect, size=size)
    warpImage2 = applyAffineTransform(src=img2Rect, srcTri=t2Rect, dstTri=tRect, size=size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = \
        img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


def morph(image_1, image_2, landmarks_1, landmarks_2, triangulation_indexes, alpha):
    """
    Morph the image.

    TODO: properly document this function.
    TODO: same alpha for both warping and blending... dangerous!
    """
    # Allocate space for final output
    imgMorph = np.zeros(image_1.shape, dtype=image_1.dtype)

    # Get the average points between the two faces.
    average_points = compute_average_coordinates(landmarks_1, landmarks_2, alpha)

    # Read the canonical triangulation
    for line in triangulation_indexes:
        # ID's of the triangulation points
        x = line[0]
        y = line[1]
        z = line[2]

        # Coordinate pairs
        t1 = [landmarks_1[x], landmarks_1[y], landmarks_1[z]]
        t2 = [landmarks_2[x], landmarks_2[y], landmarks_2[z]]
        t = [average_points[x], average_points[y], average_points[z]]

        # Morph one triangle at a time.
        morphTriangle(image_1, image_2, imgMorph, t1, t2, t, alpha)

    return imgMorph


