import numpy as np
import cupy as cp
from cucim.skimage.exposure import adjust_gamma
import cv2 as cv
from typing import Optional, Callable
import matplotlib.pyplot as plt

def import_image(path: str, cvt_control: Optional[int] = cv.COLOR_BGR2GRAY) -> np.ndarray:
    """Import image from path.
    
    Args:
        path (str): Path to the image.
        cvt_control (int): cvtColor control number.

    Returns:
        np.ndarray: Imported image.
    
    """
    image = cv.imread(path)
    if cvt_control is not None:
        image = cv.cvtColor(image, cvt_control)
    cupy_image = cp.asarray(image)
    return cupy_image

def get_image_path(path_template: str, 
                   index_of_image: int, 
                   number_of_digit: int, 
                   format: str) -> str:
    """Get paths of images.

    Args:
        path_template (str): Path template of image.
        number_of_images (int): Number of images.
        number_of_digit (int): Number of digits.
        format (str): Format of image.

    Returns:
        str: Path of image.
    
    """
    digit = f'{index_of_image}'.zfill(number_of_digit)
    path_of_image = path_template + digit + '.' + format
    return path_of_image

def import_image_sequence(path_template: str, 
                          number_of_images: int, 
                          number_of_digits: int,
                          format: str,
                          initial_number: Optional[int] = 0,
                          path_for_save: Optional[str] = None,
                          process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                          cvt_control: Optional[int] = None) -> np.ndarray:
    """ the image sequence is imported as volume. The comvection is assumed to be
    [plane, row, column], or with the direction [z, y, x].

    Args:
        sequence_path (str): Path template of image sequence.
        number_of_images (int): Number of images.
        number_of_digits (int): Number of digits.
        format (str): Format of image.
        process (Optional[Callable[[np.ndarray], np.ndarray]]): Process function of image.

    Returns:
        np.ndarray: Imported image sequence.
    
    """
    
    if process is None:
        process = lambda image: image
    
    volume = cp.stack(
        [process(import_image(get_image_path(path_template, i, number_of_digits, format), cvt_control))
         for i in range(initial_number, number_of_images)], axis=0)
    
    if path_for_save is not None:
        cp.save(path_for_save, volume)

    return volume