import numpy as np
import mpfit_single as mpfit
from fitting import get_bohlin_spline, func_voigt, newstart
from atomic_info import GetAtomProp
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from cubefit import mask_one
from IPython import embed
import copy

"""This routine performs a joint fit to Hg, Hd, and HeI 4026 from the BH2 setup"""


def full_model(p, wave, pidx, aprop):
    # Continuum
    wmin, wmax = np.min(wave), np.max(wave)
    wfit = (wave-wmin)/(wmax-wmin)
    cont = np.polyval(p[pidx == 0], wfit)
    # Absorption
    zpar = p[pidx==1][0]
    subpar = p[pidx == 2]
    abs = func_voigt([subpar[0], zpar, 1.0E-10, aprop['wave'], aprop['fval'], subpar[1]], wave)
    return cont*abs


def resid(p, fjac=None, wave=None, flux=None, errs=None, pidx=None, aprop=None):
    # Calculate the model
    model = full_model(p, wave, pidx, aprop)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    resid = (model - flux) / errs
    return [status, resid]


allwave, allflux, allerrs, allmask, widx = np.array([]), np.array([]), np.array([]), np.array([],dtype=int), np.array([],dtype=int)
# Load the data and setup the fit
dirc = "/Users/rcooke/Work/Research/BBN/Yp/HIIregions/Software/HeH_BCD/CubeFitter/"
#line, grat, xl, xr, pad, npoly = "HIg", "BH2", 250, 150, 4, 6
#line, grat, xl, xr, pad, npoly = "HId", "BH2", 200, 150, 4, 6
#line, grat, xl, xr, pad, npoly = "HeI4026", "BH2", 120, 120, 0, 6
line, grat, xl, xr, pad, npoly = "HeI4472", "BM", 130, 130, 0, 6
#line, grat, xl, xr, pad, npoly = "HIg", "BM", 100, 75, 2, 6

aprop = GetAtomProp(line)
wave, flux, errs = np.loadtxt(dirc+line+f"_{grat}_stack.dat", unpack=True)
final_mask = mask_one(flux, np.argmax(flux), pad=pad)
final_mask[:np.argmax(flux) - xl] = 0
final_mask[np.argmax(flux) + xr:] = 0
# Some manual unmasking
wman = None
if line == "HIg":
    wman = np.where(((wave > 4302.0) & (wave < 4327.0)) | ((wave > 4382.0) & (wave < 4390.0)))
elif line == "HId":
    wman = np.where(((wave > 4137.0) & (wave < 4149.0)) | ((wave > 4071.0) & (wave < 4078.0)))
elif line == "HeI4026":
    wman = np.where(((wave > 4010.0) & (wave < 4018.0)) | ((wave > 4044.0) & (wave < 4066.0)) | ((wave > 4034.0) & (wave < 4036.0)))
if wman is not None:
    final_mask[wman] = 1
# Store the data
allwave = np.append(allwave, wave)
allflux = np.append(allflux, flux)
allerrs = np.append(allerrs, errs)
allmask = np.append(allmask, final_mask.copy())

# Starting params
pinit = np.array([])
pidx = np.array([], dtype=int)
param_info = []
param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'step': 0}

# Set up the continuum parameters
cntr = 0
ptmp = np.zeros(npoly)
ptmp[-1] = 1
pinit = np.append(pinit, ptmp.copy())
pidx = np.append(pidx, np.zeros(npoly, dtype=int))
for i in range(npoly):
    param_info.append(copy.deepcopy(param_base))
    param_info[cntr + i]['value'] = ptmp[i]
cntr += npoly
# Set the redshift of all lines
pinit = np.append(pinit, 0.002363156706)
pidx = np.append(pidx, 1)
param_info.append(copy.deepcopy(param_base))
param_info[cntr]['value'] = pinit[-1]
param_info[cntr]['fixed'] = 1
cntr += 1
# Set up the absorption parameters
ptmp = np.array([13.2, 12.9])
pinit = np.append(pinit, ptmp.copy())
pidx = np.append(pidx, 2 * np.ones(ptmp.size, dtype=int))
for i in range(ptmp.size):
    param_info.append(copy.deepcopy(param_base))
    param_info[cntr + i]['value'] = ptmp[i]

# Now tell the fitting program what we called our variables
ww = np.where(allmask == 1)
fa = {'wave': allwave[ww], 'flux': allflux[ww], 'errs': allerrs[ww], 'pidx': pidx, 'aprop':aprop}
# Fit the data
m = mpfit.mpfit(resid, pinit, parinfo=param_info, functkw=fa, quiet=False)

model = full_model(m.params, allwave, pidx, aprop)
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

# Sample the covariance matrix
ptb = np.zeros((m.params.size, 1000))
ptb[np.where(m.perror != 0)[0], :] = newstart(m.covar, 1000)
wmin, wmax = np.min(allwave), np.max(allwave)
wfit = (allwave-wmin)/(wmax-wmin)
cont = np.polyval(m.params[pidx == 0], wfit)
np.savetxt(dirc+line+f"_{grat}_stack_fit.dat", np.column_stack((allwave,allflux/cont,allerrs/cont,model/cont)))
# Now sample the parameters to estimate the error associated with the underlying absorption
this_ptb = ptb[pidx==2, :]
np.save(dirc+line+f"_{grat}_stack_ptb.npy", this_ptb)
contnorm=1
plt.plot(allwave, allflux/contnorm, 'k-', drawstyle='steps-mid')
plt.plot(allwave, model/contnorm, 'r-')
plt.plot(allwave, cont/contnorm, 'b--')
#plt.plot(allwave, 0.8 + 0.1*(allflux-model)/allerrs, 'g-', drawstyle='steps-mid')
plt.fill_between(allwave, 0, 2, where=allmask==1, facecolor='green', alpha=0.5)
plt.ylim([0.5,1.2])
plt.show()

embed()
assert(False)

wmin, wmax = np.min(allwave), np.max(allwave)
wfit = (allwave-wmin)/(wmax-wmin)
cont = np.polyval(m.params[pidx == 0], wfit)
# Now sample the parameters to estimate the error associated with the underlying absorption
this_ptb = ptb[2, :]
plt.plot(allwave, allflux/cont, 'k-', drawstyle='steps-mid')
contpars = (np.outer(m.params, np.ones(1000)) + np.array(ptb))
for pp in range(1000):
    model = full_model(contpars[:,pp], allwave, pidx, aprop)
    plt.plot(allwave, model/cont, 'r-', alpha=0.01)
plt.plot(allwave, cont/cont, 'b--')
plt.fill_between(allwave, 0, 2, where=allmask==1, facecolor='green', alpha=0.5)
plt.ylim([0.5,1.2])
plt.show()
