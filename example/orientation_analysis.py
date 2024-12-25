from fiberdev import import_image_sequence
from fiberdev import crop_2D, drop_edges_3D
from fiberdev import compute_orientation, compute_structure_tensor_cucim
import cupy as cp
import cv2 as cv
import pandas as pd

"This sample code is for computing and saving orientation results"

# Path
path_of_images = "sample_images/NTC_"#Change this to the path of your images
path_of_save_dir = "sample_outdir/"  #Change this to the path of your save directory

start_pxes = (170, 170)
end_pxes = (850, 850)
number_of_images = 1000
sigma = 10 # Noise scale

# Import images as volume (cupy array)
volume = import_image_sequence(path_of_images, number_of_images, 4, "tif", process=lambda x: crop_2D(start_pxes, end_pxes, x), path_for_save=path_of_save_dir + "volume/volume.npy", cvt_control=cv.COLOR_BGR2GRAY)

# Compute structure tensor
structure_tensor = compute_structure_tensor_cucim(volume, sigma=sigma)

# Compute prientation
inplane, outofplane = compute_orientation(structure_tensor)
inplane_dropped = drop_edges_3D(sigma, inplane)
outofplane_dropped = drop_edges_3D(sigma, outofplane)
cp.save(path_of_save_dir + "orientation/inplane_waviness.npy", inplane_dropped)
cp.save(path_of_save_dir + "orientation/outofplane_waviness.npy", outofplane_dropped)

# Histgram
flatten_orientation_inplane = inplane_dropped.ravel()
flatten_orientation_outofplane = outofplane_dropped.ravel()

# Inplane
hist, bins = cp.histogram(flatten_orientation_inplane, bins=1000, density=True)
hist_series = pd.Series(cp.asnumpy(hist), name="Histgram")
bin_series = pd.Series(cp.asnumpy(bins[1:]), name="Bin")
static_df = pd.DataFrame([bin_series, hist_series], index=["Bin", "Histgram"]).transpose()
static_df.to_csv(path_of_save_dir + "orientation/inplane_histogram.csv")
average = cp.asnumpy(cp.average(flatten_orientation_inplane))
standard = cp.asnumpy(cp.std(flatten_orientation_inplane))
cv_value = standard/average
df = pd.DataFrame([average, standard, cv_value], 
                          index=["average", "standard", "cv_value"])
df.to_csv(path_of_save_dir + "orientation/inplane_static.csv")

# Outofplane
hist, bins = cp.histogram(flatten_orientation_outofplane, bins=1000, density=True)
hist_series = pd.Series(cp.asnumpy(hist), name="Histgram")
bin_series = pd.Series(cp.asnumpy(bins[1:]), name="Bin")
static_df = pd.DataFrame([bin_series, hist_series], index=["Bin", "Histgram"]).transpose()
static_df.to_csv(path_of_save_dir + "orientation/outofplane_histogram.csv")
average = cp.asnumpy(cp.average(flatten_orientation_outofplane))
standard = cp.asnumpy(cp.std(flatten_orientation_outofplane))
cv_value = standard/average
df = pd.DataFrame([average, standard, cv_value], 
                          index=["average", "standard", "cv_value"])
df.to_csv(path_of_save_dir + "orientation/outofplane_static.csv")