import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
import cucim.skimage.feature as ski
from typing import Iterable
import numba
import numba_progress
from itertools import combinations_with_replacement

symmetric_components_3D = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

def _multiply_and_filter(gradient_0, gradient_1, sigma):
    return gaussian_filter(cp.multiply(gradient_0, gradient_1), sigma, mode="nearest")

def compute_gradient_3D(volume: cp.ndarray) -> tuple:
    sigma = 1
    squeezed_volume = cp.squeeze(volume)
    Gx = gaussian_filter(squeezed_volume, sigma, order=[0, 0, 1], mode="nearest")
    Gy = gaussian_filter(squeezed_volume, sigma, order=[0, 1, 0], mode="nearest")
    Gz = gaussian_filter(squeezed_volume, sigma, order=[1, 0, 0], mode="nearest")
    return Gz, Gy, Gx

def compute_structure_tensor(gradients: Iterable, sigma: int) -> cp.ndarray:
    structure_tensor = cp.empty((len(symmetric_components_3D), *gradients[0].shape), dtype=float)
    for n, [grad_0, grad_1] in enumerate(combinations_with_replacement(gradients, 2)):
        structure_tensor[n] = _multiply_and_filter(grad_0, grad_1, sigma)

    return structure_tensor

def compute_structure_tensor_cucim(volume, sigma):
    """ Compute structure tensor using cucim.

    Args:
        volume (cp.ndarray): 3D volume.
        sigma (int): Noise scale.

    Returns:
        cp.ndarray: Structure tensor.
    
    """
    tensors_list = ski.structure_tensor(volume, sigma=sigma, mode="nearest")
    tensors = cp.empty((6, *tensors_list[0].shape), dtype=float)
    for n, tensor in enumerate(tensors_list):
        tensors[n] = tensor
    return tensors

@numba.njit(parallel=True, cache=True)
def orientation_function(structureTensor, progressProxy, fibre=True):
    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    theta = np.zeros(structureTensor.shape[1:], dtype="<f4")
    phi = np.zeros(structureTensor.shape[1:], dtype="<f4")

    for z in numba.prange(0, structureTensor.shape[1]):
        for y in range(0, structureTensor.shape[2]):
            for x in range(0, structureTensor.shape[3]):
                structureTensorLocal = np.empty((3, 3), dtype="<f4")
                for n, [i, j] in enumerate(symmetricComponents3d):
                    structureTensorLocal[i, j] = structureTensor[n, z, y, x]
                    if i != j:
                        structureTensorLocal[j, i] = structureTensor[n, z, y, x]

                w, v = np.linalg.eig(structureTensorLocal)
                m = np.argmin(w)

                selectedEigenVector = v[:, m]

                if selectedEigenVector[0] < 0:
                    selectedEigenVector *= -1

                theta[z, y, x] = np.rad2deg(np.arctan2(selectedEigenVector[2], selectedEigenVector[0]))
                phi[z, y, x] = np.rad2deg(np.arctan2(selectedEigenVector[1], selectedEigenVector[0]))

        progressProxy.update(1)

    return theta, phi

def compute_orientation(structure_tensor):
    """ Compute orientation.
    
    Args:
        structure_tensor (cp.ndarray): Structure tensor.
    
    Returns:
        tuple: Orientation.
    """
    numpy_structure_tensor = cp.asnumpy(structure_tensor)

    with numba_progress.ProgressBar(total=numpy_structure_tensor.shape[1]) as progress:
        numpy_theta, numpy_phi = orientation_function(
            numpy_structure_tensor,
            progress)
            
    theta = cp.asarray(numpy_theta)
    phi = cp.asarray(numpy_phi)

    return theta, phi

@numba.njit(parallel=True, cache=True)
def orientation_function_axial(structureTensor, progressProxy, fibre=True):
    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    theta = np.zeros(structureTensor.shape[1:], dtype="<f4")
    axial_vec = np.array([1, 0, 0], dtype="<f4")

    for z in numba.prange(0, structureTensor.shape[1]):
        for y in range(0, structureTensor.shape[2]):
            for x in range(0, structureTensor.shape[3]):
                structureTensorLocal = np.empty((3, 3), dtype="<f4")
                for n, [i, j] in enumerate(symmetricComponents3d):
                    structureTensorLocal[i, j] = structureTensor[n, z, y, x]
                    if i != j:
                        structureTensorLocal[j, i] = structureTensor[n, z, y, x]

                w, v = np.linalg.eig(structureTensorLocal)
                m = np.argmin(w)

                selectedEigenVector = v[:, m]

                if selectedEigenVector[0] < 0:
                    selectedEigenVector *= -1

                theta[z, y, x] = np.rad2deg(np.arccos(np.dot(selectedEigenVector, axial_vec)))

        progressProxy.update(1)

    return theta

def compute_orientation_axial(structure_tensor):
    numpy_structure_tensor = cp.asnumpy(structure_tensor)

    with numba_progress.ProgressBar(total=numpy_structure_tensor.shape[1]) as progress:
        numpy_theta= orientation_function_axial(
            numpy_structure_tensor,
            progress)
            
    theta = cp.asarray(numpy_theta)
    return theta