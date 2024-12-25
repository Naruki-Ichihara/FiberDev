from fiberdev import MaterialParams, estimate_compression_strength
import numpy as np
import matplotlib.pyplot as plt

# Material parameters
material_params = MaterialParams(
    longitudinal_modulus=112700,
    transverse_modulus=10000,
    poisson_ratio=0.29,
    shear_modulus=5450,
    tau_y=46.4,
    K=(3/7)*0.0075,
    n=4.6)

# Estimate compression strength
deviation = 3.356
comp_stress, comp_strain, stress, strain = estimate_compression_strength(
    initial_misalignment=0,
    standard_deviation=deviation,
    material_params=material_params,
    maximum_axial_strain=0.015,
    fiber_misalignment_step_size=0.02,
    maximum_fiber_misalignment=15
    )
print(f"Compression strength: {comp_stress} MPa")
plt.plot(strain, stress)
plt.xlabel("Axial compressive strain [-]")
plt.ylabel("Axial compressive stress [MPa]")
plt.tick_params(direction='in')
plt.xlim(0, 0.015)
plt.ylim(0, 2000)
plt.savefig("sample_outdir/compression_strength.png", dpi=300)