import numpy as np
import mpfit_single as mpfit
from fitting import get_bohlin_spline
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from cubefit import mask_one
from IPython import embed
import copy

# This routine performs a joint fit to Hg, Hd, and HeI 4026 from the BH2 setup

def func_stellarabs(par, wavein, mgrid):
    tval = np.ones(wavein.size) * par[0]
    gval = np.ones(wavein.size) * par[2]
    return mgrid(np.column_stack((tval,gval,wavein/(1+par[1]))))


def full_model(p, wave, pidx, widx, stellar=None):
    # Continuum
    cont = np.ones_like(wave)
    for ii in range(3):
        cont[widx == ii] = np.polyval(p[pidx == ii], wave[widx == ii])
    # Absorption
    abs = np.ones_like(wave)
    subpar = p[pidx==np.max(pidx)]
    for ii in range(3):
        tval = np.ones(wave[widx == ii].size) * subpar[0]
        gval = np.ones(wave[widx == ii].size) * subpar[2]
        abs[widx == ii] = stellar[ii](np.column_stack((tval, gval, wave[widx == ii] / (1 + subpar[1]))))
    return cont*abs


def resid(p, fjac=None, wave=None, flux=None, errs=None, pidx=None, widx=None, stellar=None):
    # Calculate the model
    model = full_model(p, wave, pidx, widx, stellar=stellar)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    resid = (model - flux) / errs
    return [status, resid]

allwave, allflux, allmask, widx = np.array([]), np.array([]), np.array([],dtype=int), np.array([],dtype=int)
# Load the data and setup the fit
dirc = "/Users/rcooke/Work/Research/BBN/Yp/HIIregions/Software/HeH_BCD/CubeFitter/"
fils = ["HIg", "HId", "HeI4026"]
xl = [250, 200, 120]
xr = [150, 150, 120]
pad = [4, 4, 1]
stellar = []
for ff, fil in enumerate(fils):
    wave, flux = np.loadtxt(dirc+fil+"_BH2_stellar_abs.dat", unpack=True, usecols=(0,1))
    final_mask = mask_one(flux, np.argmax(flux), pad=pad[ff])
    final_mask[:np.argmax(flux) - xl[ff]] = 0
    final_mask[np.argmax(flux) + xr[ff]:] = 0
    allwave = np.append(allwave, wave)
    allflux = np.append(allflux, flux)
    allmask = np.append(allmask, final_mask.copy())
    widx = np.append(widx, ff*np.ones(flux.size))
    stellar.append(get_bohlin_spline("BH2", fil)[3])

# Starting params
pinit = np.array([])
pidx = np.array([], dtype=int)
param_info = []
param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'step': 0}

# Set up the continuum parameters
npoly = [3, 3, 3]
cntr = 0
for pp in range(len(npoly)):
    ptmp = np.zeros(npoly[pp])
    ptmp[-1] = 1
    pinit = np.append(pinit, ptmp.copy())
    pidx = np.append(pidx, pp*np.ones(npoly[pp], dtype=int))
    for i in range(npoly[pp]):
        param_info.append(copy.deepcopy(param_base))
        param_info[cntr + i]['value'] = ptmp[i]
    cntr += npoly[pp]
# Set up the absorption parameters
ptmp = np.array([12492.43252, 0.002379467609, 27.00872819])
pinit = np.append(pinit, ptmp.copy())
pidx = np.append(pidx, (1+np.max(pidx))*np.ones(ptmp.size, dtype=int))
for i in range(ptmp.size):
    param_info.append(copy.deepcopy(param_base))
    param_info[cntr + i]['value'] = ptmp[i]
param_info[cntr]['limited'] = [1,1]
param_info[cntr]['limits'] = [3500,30000]
param_info[cntr+2]['limited'] = [1,1]
param_info[cntr+2]['limits'] = [0,50]

# Now tell the fitting program what we called our variables
ww = np.where(allmask==1)
fa = {'wave': allwave[ww], 'flux': allflux[ww], 'errs': np.ones(allflux[ww].size), 'pidx': pidx, 'widx':widx[ww], 'stellar':stellar}

m = mpfit.mpfit(resid, pinit, parinfo=param_info, functkw=fa, quiet=False)

print(m.status)
model = full_model(m.params, allwave, pidx, widx, stellar=stellar)
plt.subplot(131)
plt.plot(allwave[widx==0], allflux[widx==0], 'k-', drawstyle='steps-mid')
plt.plot(allwave[widx==0], model[widx==0], 'r-')
plt.fill_between(allwave[widx==0], 0, 2, where=allmask[widx==0]==1, facecolor='green', alpha=0.5)
plt.ylim([0.5,1.2])
plt.subplot(132)
plt.plot(allwave[widx==1], allflux[widx==1], 'k-', drawstyle='steps-mid')
plt.plot(allwave[widx==1], model[widx==1], 'r-')
plt.fill_between(allwave[widx==1], 0, 2, where=allmask[widx==1]==1, facecolor='green', alpha=0.5)
plt.ylim([0.5,1.2])
plt.subplot(133)
plt.plot(allwave[widx==2], allflux[widx==2], 'k-', drawstyle='steps-mid')
plt.plot(allwave[widx==2], model[widx==2], 'r-')
plt.fill_between(allwave[widx==2], 0, 2, where=allmask[widx==2]==1, facecolor='green', alpha=0.5)
plt.ylim([0.5,1.2])
plt.show()