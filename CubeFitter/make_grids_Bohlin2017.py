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

def convolve(wav, flx, vfwhm):
    """
    Define the functional form of the model
    --------------------------------------------------------
    x  : array of wavelengths
    y  : model flux array
    p  : array of parameters for this model
    --------------------------------------------------------
    """
    sigd = vfwhm / ( 2.99792458E5 * ( 2.0*np.sqrt(2.0*np.log(2.0)) ) )
    ysize=flx.size
    fsigd=6.0*sigd
    dwav = 0.5 * (wav[2:] - wav[:-2]) / wav[1:-1]
    dwav = np.append(np.append(dwav[0],dwav),dwav[-1])
    df= int(np.min([np.int(np.ceil(fsigd/dwav).max()), ysize//2 - 1]))
    yval = np.zeros(2*df+1)
    yval[df:2*df+1] = (wav[df:2 * df + 1] / wav[df] - 1.0) / sigd
    yval[:df] = (wav[:df] / wav[df] - 1.0) / sigd
    gaus = np.exp(-0.5*yval*yval)
    size = ysize + gaus.size - 1
    fsize = 2 ** np.int(np.ceil(np.log2(size))) # Use this size for a more efficient computation
    conv = np.fft.fft(flx, fsize)
    conv *= np.fft.fft(gaus/gaus.sum(), fsize)
    ret = np.fft.ifft(conv).real.copy()
    del conv
    return ret[df:df+ysize]


# Set the path and grab the model files
dirc = "/Users/rcooke/Work/Research/BBN/Yp/HIIregions/Software/BohlinStellarModels/metal_-1.50/"
carbon_options = ["carbon_+0.00", "carbon_+0.25", "carbon_+0.50", "carbon_-0.25", "carbon_-0.50", "carbon_-0.75"]
alpha_options = ["alpha_+0.00", "alpha_+0.25", "alpha_+0.50", "alpha_-0.25"]
carbon = carbon_options[0]
alpha = alpha_options[1]
full_path = dirc + carbon + "/" + alpha + "/*"
all_files = glob.glob(full_path)

# Set the transition parameters
#line, delwave, grating, slicer = "HIg", 100, "BH2", "Small"
line, delwave, grating, slicer = "HId", 100, "BH2", "Small"
#line, delwave, grating, slicer = "HeI4026", 100, "BH2", "Small"

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

# Plot the grid ranges
if True:
    print(unq_grav)
    print(unq_temp)
    plt.plot(all_temp, all_grav, 'rx')
    plt.show()

# Load the first file to determine number of spectral elements and relevant indices
wave = np.loadtxt(all_files[0], unpack=True, usecols=(0,))
# Cut down the wavelength range to speed up the convolution
wmin, wmax = atom_prop['wave']-2*delwave, atom_prop['wave']+2*delwave
wcut = np.where((wave>wmin) & (wave<wmax))
wavecut = wave[wcut]
# Now select the wavelength range of interest to store in the grid
wmin, wmax = atom_prop['wave']-delwave, atom_prop['wave']+delwave
ww = np.where((wavecut>wmin) & (wavecut<wmax))
np.save(f"Bohlin2017_Wgrid_{grating}_{line}.npy", wavecut[ww])
# Output grid shape
full_grid = np.ones((unq_temp.size, unq_grav.size, ww[0].size))
# Load each grid, convolve
for ff, fil in enumerate(all_files):
    print(ff+1, "/", len(all_files))
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
    flux, cont = np.loadtxt(fil, unpack=True, usecols=(1,2))
    fnorm = flux/cont
    # Convolve the data
    cnorm = convolve(wavecut, fnorm[wcut], resln)
    # plt.plot(wavecut,fnorm[wcut], 'k-')
    # plt.plot(wavecut, cnorm, 'r-')
    # plt.show()
    # Extract the relevant bits of data for the interpolation
    full_grid[tidx, gidx, :] = cnorm[ww]

np.save(f"Bohlin2017_Mgrid_{grating}_{line}.npy", full_grid)
