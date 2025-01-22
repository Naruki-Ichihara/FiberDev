from fiberdev import import_image_sequence
from fiberdev import crop_2D, drop_edges_3D
from fiberdev import compute_orientation_axial, compute_structure_tensor_cucim
import cupy as cp
import cv2 as cv
import pandas as pd

"This sample code is for computing and saving orientation results"

# Path
path_of_images = "image_dir/HA_1/HA_1_20250117_160621_"
path_of_save_dir = "outdir/"
start_pxes = (150, 150)
end_pxes = (650, 650)
number_of_images = 1880
sigma = 10
# Import images as volume (cupy array)
volume = import_image_sequence(path_of_images, number_of_images, 4, "dcm", initial_number=30, process=lambda x: crop_2D(start_pxes, end_pxes, x), path_for_save=path_of_save_dir + "volume.npy")

structure_tensor = compute_structure_tensor_cucim(volume, sigma=sigma)
orientation = compute_orientation_axial(structure_tensor)
dropped = drop_edges_3D(sigma, orientation)
cp.save(path_of_save_dir + "orientation.npy", dropped)

# Histgram
flatten_orientation = dropped.ravel()

# Inplane
hist, bins = cp.histogram(flatten_orientation, bins=1000, density=True)
hist_series = pd.Series(cp.asnumpy(hist), name="Histgram")
bin_series = pd.Series(cp.asnumpy(bins[1:]), name="Bin")
static_df = pd.DataFrame([bin_series, hist_series], index=["Bin", "Histgram"]).transpose()
static_df.to_csv(path_of_save_dir + "histogram.csv")