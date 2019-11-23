import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import matplotlib.pyplot as plt
import numpy as np


# project_name = 'cmos_dielectric_single_band_contrast_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_single_band_contrast_lowindex_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_single_band_contrast_all_transmission_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_single_band_low_index_contrast_all_transmission_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_metal_single_band_m10m3_contrast_all_transmission_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_metal_single_band_m18m12p5_contrast_all_transmission_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_single_band_contrast_multi_freq_all_transmission_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_single_band_contrast_multi_freqx30_all_transmission_2d_no_feature_size_strict_layering_rgb_4xtsmc_um'
# project_name = 'cmos_dielectric_single_band_contrast_multi_freqx30_all_transmission_2d_no_feature_size_strict_layering_rgb_1xtsmc_um'
project_name = 'cmos_dielectric_single_band_contrast_multi_freqx30_all_transmission_2d_no_feature_size_strict_layering_rgb_2p5xtsmc_um'

python_src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
projects_directory_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/'))

if not os.path.isdir(projects_directory_location):
    os.mkdir(projects_directory_location)

projects_directory_location += "/" + project_name

if not os.path.isdir(projects_directory_location):
    os.mkdir(projects_directory_location)


# fom_focus = np.load(projects_directory_location + "/figure_of_merit_focus.npy")
fom_transmission = np.load(projects_directory_location + "/figure_of_merit_transmission.npy")
fom_ref_low = np.load(projects_directory_location + "/figure_of_merit_reflect_low.npy")
fom_ref_high = np.load(projects_directory_location + "/figure_of_merit_reflect_high.npy")
fom = np.load(projects_directory_location + "/figure_of_merit.npy")

fom_transmission = fom_transmission[0]
fom_ref_low = fom_ref_low[0]
fom_ref_high = fom_ref_high[0]
fom = fom[0]

plt.plot(fom_transmission, color='r', linewidth=2)
plt.plot(fom_ref_low, color='g', linewidth=2)
plt.plot(fom_ref_high, color='b', linewidth=2)
plt.plot(fom, color='k', linewidth=2, linestyle='--')
plt.legend(['transmission', 'reflection low', 'reflection high', 'mean'])
plt.show()

