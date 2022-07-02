import numpy as np
import mpfit_single as mpfit
from fitting import get_bohlin_spline, func_voigt
from atomic_info import GetAtomProp
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from cubefit import mask_one
from IPython import embed
import copy

"""This routine performs a joint fit to Hg, Hd, and HeI 4026 from the BH2 setup"""


def full_model(p, wave, pidx, widx, aprop):
    # Continuum
    cont = np.ones_like(wave)
    for ii in range(3):
        cont[widx == ii] = np.polyval(p[pidx == ii], wave[widx == ii])
    # Absorption
    abs = np.ones_like(wave)
    zpar = p[pidx==3][0]
    for ii in range(3):
        subpar = p[pidx == 4+ii]
        abs[widx == ii] = func_voigt([subpar[0], zpar, 1.0E-10, aprop[ii]['wave'], aprop[ii]['fval'], subpar[1]], wave[widx == ii])
    return cont*abs


def resid(p, fjac=None, wave=None, flux=None, errs=None, pidx=None, widx=None, aprop=None):
    # Calculate the model
    model = full_model(p, wave, pidx, widx, aprop)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    resid = (model - flux) / errs
    return [status, resid]


allwave, allflux, allerrs, allmask, widx = np.array([]), np.array([]), np.array([]), np.array([],dtype=int), np.array([],dtype=int)
# Load the data and setup the fit
dirc = "/Users/rcooke/Work/Research/BBN/Yp/HIIregions/Software/HeH_BCD/CubeFitter/"
fils = ["HIg", "HId", "HeI4026"]
aprop = [GetAtomProp(line) for line in fils]
xl = [250, 200, 120]
xr = [150, 150, 120]
pad = [4, 4, 1]
for ff, fil in enumerate(fils):
    wave, flux, errs = np.loadtxt(dirc+fil+"_BH2_stack.dat", unpack=True)
    final_mask = mask_one(flux, np.argmax(flux), pad=pad[ff])
    final_mask[:np.argmax(flux) - xl[ff]] = 0
    final_mask[np.argmax(flux) + xr[ff]:] = 0
    # Some manual unmasking
    wman = None
    if ff == 0:
        wman = np.where(((wave > 4302.0) & (wave < 4327.0)) | ((wave > 4382.0) & (wave < 4390.0)))
    elif ff == 1:
        wman = np.where(((wave > 4137.0) & (wave < 4149.0)) | ((wave > 4071.0) & (wave < 4078.0)))
    elif ff == 2:
        wman = np.where(((wave > 4010.0) & (wave < 4018.0)) | ((wave > 4044.0) & (wave < 4066.0)))
    if wman is not None:
        final_mask[wman] = 1
    # Store the data
    allwave = np.append(allwave, wave)
    allflux = np.append(allflux, flux)
    allerrs = np.append(allerrs, errs)
    allmask = np.append(allmask, final_mask.copy())
    widx = np.append(widx, ff*np.ones(flux.size))

# Starting params
pinit = np.array([])
pidx = np.array([], dtype=int)
param_info = []
param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'step': 0}

# Set up the continuum parameters
npoly = [4, 4, 4]
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
# Set the redshift of all lines
pinit = np.append(pinit, 0.002362139526)
pidx = np.append(pidx, len(npoly))
param_info.append(copy.deepcopy(param_base))
param_info[cntr]['value'] = pinit[-1]
cntr += 1
# Set up the absorption parameters
ptmp = np.array([13.2, 12.9])
for pp in range(3):
    pinit = np.append(pinit, ptmp.copy())
    pidx = np.append(pidx, (pp + 4) * np.ones(ptmp.size, dtype=int))
    for i in range(ptmp.size):
        param_info.append(copy.deepcopy(param_base))
        param_info[cntr + i]['value'] = ptmp[i]
    cntr += 2

# Now tell the fitting program what we called our variables
ww = np.where(allmask == 1)
fa = {'wave': allwave[ww], 'flux': allflux[ww], 'errs': allerrs[ww], 'pidx': pidx, 'widx':widx[ww], 'aprop':aprop}
# Fit the data
m = mpfit.mpfit(resid, pinit, parinfo=param_info, functkw=fa, quiet=False)

model = full_model(m.params, allwave, pidx, widx, aprop)
# plt.subplot(131)
# plt.plot(allwave[widx==0], allflux[widx==0], 'k-', drawstyle='steps-mid')
# plt.plot(allwave[widx==0], model[widx==0], 'r-')
# plt.fill_between(allwave[widx==0], 0, 2, where=allmask[widx==0]==1, facecolor='green', alpha=0.5)
# plt.ylim([0.5,1.2])
# plt.subplot(132)
# plt.plot(allwave[widx==1], allflux[widx==1], 'k-', drawstyle='steps-mid')
# plt.plot(allwave[widx==1], model[widx==1], 'r-')
# plt.fill_between(allwave[widx==1], 0, 2, where=allmask[widx==1]==1, facecolor='green', alpha=0.5)
# plt.ylim([0.5,1.2])
# plt.subplot(133)
# plt.plot(allwave[widx==2], allflux[widx==2], 'k-', drawstyle='steps-mid')
# plt.plot(allwave[widx==2], model[widx==2], 'r-')
# plt.fill_between(allwave[widx==2], 0, 2, where=allmask[widx==2]==1, facecolor='green', alpha=0.5)
# plt.ylim([0.5,1.2])
# plt.show()

for ff, fil in enumerate(fils):
    this = (widx==ff)
    cont = np.polyval(m.params[pidx == ff], allwave[this])
    np.savetxt(dirc+fil+"_BH2_stack_fit.dat", np.column_stack((allwave[this],allflux[this]/cont,allerrs[this]/cont,model[this]/cont)))
    plt.subplot(1,3,ff+1)
    plt.plot(allwave[widx==ff], allflux[widx==ff]/cont, 'k-', drawstyle='steps-mid')
    plt.plot(allwave[widx==ff], model[widx==ff]/cont, 'r-')
    plt.plot(allwave[widx==ff], cont/cont, 'b--')
    plt.fill_between(allwave[widx==ff], 0, 2, where=allmask[widx==ff]==1, facecolor='green', alpha=0.5)
    plt.ylim([0.5,1.2])
plt.show()
