import numpy as np
from matplotlib import pyplot as plt
from linetools.spectra.xspectrum1d import XSpectrum1D
from astropy.wcs import WCS
import astropy.io.fits as fits
import astropy.units as units
from pypeit import utils
import fitting, atomic_info
from cubefit import get_mapname, mask_one
from IPython import embed

# Set the properties of the extraction/fit
line, npoly, xl, xr = "HIg", 3, 100, 0   # lineID, polynomial order to fit to continuum, number of extra pixels on the left (xl) and right (xr) to include in the final fit (default is +/-150 pixels of the line centre minus the emission line)
#line, npoly = "HId", 3
#line, npoly = "HeI4026", 3
linear = True  # Use a linear fit to the continuum regions?
plotit = True  # Plot some QA?

# Load the datacubes
dirc = "../../../IZw18_KCWI/final_cubes/"
filename = "IZw18_BH2_newSensFunc.fits"
hdus = fits.open(dirc+filename)
wcs = WCS(hdus[1].header)
datcube = hdus[1].data
sigcube = np.sqrt(hdus[2].data)
wave = wcs.wcs_pix2world(0.0, 0.0, np.arange(datcube.shape[0]), 0)[2]*1.0E10

# Load the maps
atom_prop = atomic_info.GetAtomProp(line)
mapname = get_mapname(dirc, filename, line)
data = fits.open(mapname)
maps = dict(flux=data[0].data, errs=data[1].data, cont=data[2].data, complete=data[3].data, params=data[4].data)
msk = data[5].data

# Calculate vshift for all spaxels
make_velocity_map = False
if make_velocity_map:
    bstx, bsty, bstf = -1, -1, -1
    vshift = np.zeros_like(maps['flux'])
    for xx in range(maps['flux'].shape[0]):
        print(f"vshift for row {xx+1}/{datcube.shape[1]}")
        for yy in range(maps['flux'].shape[1]):
            if maps['params'][0,xx,yy]==0: continue
            ww = np.where(msk[:,xx,yy]==1)[0]
            wmin, wmax = wave[ww[0]], wave[ww[-1]]
            we = np.where((msk[:,xx,yy]==0) & (wave>wmin) & (wave<wmax))  # These are the pixels where there is an emission line
            flxcsum = np.cumsum(datcube[we[0],xx,yy])
            flxtot = flxcsum[-1]
            vshift[xx,yy] = np.interp(0.5, flxcsum/flxtot, wave[we])
            if flxtot > bstf:
                bstx, bsty = xx, yy

    vmap = (vshift - vshift[bstx, bsty])/vshift[bstx, bsty]  # this is delta lambda / lambda
    vmap[vshift==0] = 0
    np.save(dirc+"maps/IZw18_BH2_newSensFunc_HIg_vmap.npy", vmap)
    # Plot the vmap
    plt.imshow(299792.458*vmap, vmin=0, vmax=100)
    plt.show()
else:
    vmap = np.load(dirc+"maps/IZw18_BH2_newSensFunc_HIg_vmap.npy")

if plotit:
    xx, yy = 30, 30
    plt.plot(wave/(1+vmap[xx,yy]), datcube[:,xx,yy]/np.max(datcube[:,xx,yy]), 'r-')
    xx, yy = 45, 21
    plt.plot(wave/(1+vmap[xx,yy]), datcube[:,xx,yy]/np.max(datcube[:,xx,yy]), 'k-')
    plt.show()

raw_specs = []
for xx in range(maps['flux'].shape[0]):
    print(f"vshift for row {xx+1}/{datcube.shape[1]}")
    for yy in range(maps['flux'].shape[1]):
        if maps['params'][0,xx,yy]==0: continue
        newwave = wave/(1+vmap[xx,yy])
        if linear:
            ww = np.where((msk[:,xx,yy]==1) & (sigcube[:,xx,yy]!=0.0))[0]
            cc = np.polyfit(wave[ww], datcube[ww,xx,yy], 1)
            cont = np.polyval(cc, wave)
        else:
            cont = np.polyval(maps['params'][:npoly,xx,yy], wave)
        if np.any(cont<=0): continue
        raw_specs.append(XSpectrum1D.from_tuple((newwave, datcube[:,xx,yy]/cont, sigcube[:,xx,yy]/cont), verbose=False))

msked = np.where(msk)[0]
out_wave = wave[msked.min():msked.max()]
maskval = -9.99
sigcut = 5
# Combine the spectra while rejecting outliers
npix, nspec = out_wave.size, len(raw_specs)
new_specs = []
out_flux = maskval*np.ones((npix, nspec))
out_flue = maskval*np.ones((npix, nspec))
for sp in range(nspec):
    new_specs.append(raw_specs[sp].rebin(out_wave*units.AA, do_sig=True, grow_bad_sig=True))
    gpm = new_specs[0].sig != 0.0
    out_flux[gpm,sp] = new_specs[sp].flux[gpm]
    out_flue[gpm,sp] = new_specs[sp].sig[gpm]
# Calculate a reference spectrum
flx_ma = np.ma.array(out_flux, mask=out_flux==maskval, fill_value=0.0)
ref_spec = np.ma.median(flx_ma, axis=1)
ref_spec_mad = 1.4826*np.ma.median(np.abs(flx_ma-ref_spec.reshape(ref_spec.size, 1)), axis=1)
# Determine which pixels to reject/include in the final combination
devs = (out_flux-ref_spec.reshape(ref_spec.size, 1))/out_flue
mskdev = np.ma.abs(devs) < sigcut
# Make a new array
new_mask = np.logical_not(mskdev.data & np.logical_not(flx_ma.mask))
final_flux = np.ma.array(flx_ma.data, mask=new_mask, fill_value=0.0)
final_flue = np.ma.array(out_flue, mask=new_mask, fill_value=0.0)
# Compute the final weighted spectrum
ivar = utils.inverse(final_flue**2)
final_spec = np.ma.average(final_flux, weights=ivar, axis=1)
variance = np.ma.average((final_flux-final_spec[:,np.newaxis])**2, weights=ivar, axis=1)
final_spec_err = np.sqrt(variance)
# Plot it
plt.plot(out_wave, final_spec, 'k-', drawstyle='steps')
plt.plot(out_wave, final_spec_err, 'r-', drawstyle='steps')
plt.show()

# Perform a fit to the continuum+absorption of the combined spectrum
embed()
assert(False)
final_mask = mask_one(final_spec, np.argmax(final_spec), pad=4)
final_mask[:np.argmax(final_spec)-150-xl] = 0
final_mask[np.argmax(final_spec)+150+xr:] = 0
if plotit:
    plt.plot(out_wave[final_mask == 0], final_spec[final_mask == 0], 'ro')
    plt.plot(out_wave, final_spec, 'k-', drawstyle='steps-mid')
    plt.show()
p0a = np.array([14.0, out_wave[np.argmax(final_spec)]/atom_prop['wave']-1, 100.0, atom_prop['wave'], atom_prop['fval'], 13])
# Perform the fit
flxsum, errsum, contval, pars = fitting.fit_one_cont(atom_prop, out_wave, final_spec, final_spec_err, mask,
                                                     npoly=npoly, contsample=100, verbose=True,
                                                     p0c=None, p0a=p0a, p0e=None,
                                                     include_em=False, include_ab=True)
index = np.ones(npoly+6, dtype=int)
index[:npoly] = 0
model = fitting.full_model(np.append(pars[:npoly],p0a), out_wave, index)
plt.plot(out_wave[final_mask == 0], final_spec[final_mask == 0], 'ro')
plt.plot(out_wave, final_spec, 'k-', drawstyle='steps-mid')
plt.plot(out_wave, model, 'r-')
plt.show()
print("ERROR - need to make a correction map!!")
