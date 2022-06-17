import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import atomic_info
from IPython import embed

def get_resln(grating, slicer):
    """ Return KCWI spectral resolution (km/s) for a given grating and slicer"""
    if slicer == "Small":
        if grating == "BM":
            return 299792.458/8000
        if grating == "BH2":
            return 299792.458/18000
    print("Not implemented...")
    assert(False)

dirc = "/Users/rcooke/Work/Research/BBN/Yp/HIIregions/Software/BohlinStellarModels/metal_-1.50/"
carbon_options = ["carbon_+0.00", "carbon_+0.25", "carbon_+0.50", "carbon_-0.25", "carbon_-0.50", "carbon_-0.75"]
alpha_options = ["alpha_+0.00", "alpha_+0.25", "alpha_+0.50", "alpha_-0.25"]
carbon = carbon_options[0]
alpha = alpha_options[1]

full_path = dirc + carbon + "/" + alpha + "/*"
all_files = glob.glob(full_path)
line, grating, slicer = "HIg", "BH2", "Small"

resln = get_resln(grating, slicer)
# Load some information
atom_prop = atomic_info.GetAtomProp(line)

# Generate a list of all temperature and surface gravities for the models
all_temp, all_grav = np.zeros(len(all_files)), np.zeros(len(all_files))
for ff, fil in enumerate(all_files):
    filn = os.path.basename(fil)
    try:
        all_temp[ff] = int(filn[14:19])  # >= 10,000 K
        all_grav[ff] = int(filn[20:22])
    except ValueError:
        all_temp[ff] = int(filn[14:18])  # < 10,000 K
        all_grav[ff] = int(filn[19:21])

# Extract the unique set of files
unq_temp = np.sort(np.unique(all_temp))
unq_grav = np.sort(np.unique(all_grav))
np.save("Bohlin2017_Tgrid.npy", unq_temp)
np.save("Bohlin2017_Ggrid.npy", unq_grav)

# Make sure this is a square grid
if True:
    print(unq_grav)
    print(unq_temp)
    plt.plot(all_temp, all_grav, 'rx')
    plt.show()

# Load the first file to determine number of spectral elements and relevant indices
wmin, wmax =
wave = np.loadtxt(all_files[0], unpack=True, use_cols=(0,))
ww = np.where((wave>wmin) & (wave<wmax))
# Output grid shape
full_grid = np.ones((unq_temp.size, unq_grav.size, ww[0].size))
# Load each grid, convolve
for ff, fil in enumerate(all_files):
    filn = os.path.basename(fil)
    try:
        this_temp = int(filn[14:19])  # >= 10,000 K
        this_grav = int(filn[20:22])
    except ValueError:
        this_temp = int(filn[14:18])  # < 10,000 K
        this_grav = int(filn[19:21])
    tidx = np.argmin(np.abs(unq_temp - this_temp))
    gidx = np.argmin(np.abs(unq_grav - this_grav))
    # Read in the data
    wave, flux, cont = np.loadtxt(fil, unpack=True)
    fnorm = flux/cont
    # Convolve the data
    cnorm = convolve(, resln)
    # Extract the relevant bits of data for the interpolation
    full_grid[tidx, gidx, :] = cnorm[ww]

np.save("Bohlin2017_FULLgrid.npy", full_grid)
