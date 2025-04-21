from fiberdev import import_image_sequence
from fiberdev import crop_2D
import cv2 as cv

# Path
path_of_images = "sample_images/NTC_"
path_of_save_dir = "outdir/"
start_pxes = (170, 170)
end_pxes = (180, 180)
number_of_images = 10

# Import images as volume (cupy array)
volume = import_image_sequence(path_of_images, number_of_images, 4, "tif", process=lambda x: crop_2D(start_pxes, end_pxes, x), path_for_save=path_of_save_dir + "volume_sub.npy", cvt_control=cv.COLOR_BGR2GRAY)