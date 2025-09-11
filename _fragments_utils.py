import cv2
import numpy as np
import random
from typing import Tuple, List



def fragment_image(
    image : np.ndarray,
    grid_size : Tuple[int, int]) -> List[List[np.ndarray]]:
    """
    Splits an image into a grid of fragments.

    Parameters
    ----------
    image : np.ndarray
        Input image (H x W x C).
    grid_size : Tuple[int, int]
        (rows, cols) of the grid.

    Returns
    -------
    List[List[np.ndarray]]
        2D list of image fragments.
    """
    rows, cols = grid_size
    h, w = image.shape[:2]
    tile_h = h // rows
    tile_w = w // cols

    fragments = []
    for i in range(rows):
        row = []
        for j in range(cols):
            fragment = image[i*tile_h:(i+1)*tile_h, 
                             j*tile_w:(j+1)*tile_w].copy()
            row.append(fragment)
        fragments.append(row)

    return fragments


def create_composites(
    fragmented_images: List[List[List[np.ndarray]]],
    num_composites: int) -> List[np.ndarray]:
    """
    Creates composite images by randomly selecting tiles
    from corresponding positions.

    Parameters
    ----------
    fragmented_images : List[List[List[np.ndarray]]])
        List of fragmented images.
    num_composites : int
        Number of composite images to generate.

    Returns:
    List[np.ndarray]
        List of composite images.
    """
    num_rows = len(fragmented_images[0])
    num_cols = len(fragmented_images[0][0])
    tile_h, tile_w = fragmented_images[0][0][0].shape[:2]

    composites = []

    for _ in range(num_composites):
        composite_tiles = []
        for i in range(num_rows):
            row_tiles = []
            for j in range(num_cols):
                # Randomly pick one tile from the corresponding position
                tile = random.choice(fragmented_images)[i][j]
                row_tiles.append(tile)

            # Horizontally stack one row of tiles
            composite_row = np.hstack(row_tiles)
            composite_tiles.append(composite_row)

        # Vertically stack all rows to make final composite image
        composite_image = np.vstack(composite_tiles)
        composites.append(composite_image)

    return composites



if __name__ == "__main__":
    # Load your 10 images
    images = [cv2.imread("1.jpg"), cv2.imread("2.jpg")]

    # Fragment each image into a 10x10 grid
    grid_size = (10, 10)
    fragmented_images = [fragment_image(img, grid_size) for img in images]

    # Create 30 composite images
    composites = create_composites(fragmented_images, num_composites=30)

    # Save results
    for i, comp in enumerate(composites):
        cv2.imwrite(f"composite_{i}.jpg", comp)
