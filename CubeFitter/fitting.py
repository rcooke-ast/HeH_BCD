import numpy as np
import mpfit_single as mpfit
from scipy.special import wofz
from scipy.interpolate import RegularGridInterpolator
import copy
from IPython import embed


def newstart(covar, num):
    """Sample the covariance matrix (covar), num times."""
    # Find the non-zero elements in the covariance matrix
    cvsize = covar.shape[0]
    cxzero, cyzero = np.where(covar == 0.0)
    bxzero, byzero = np.bincount(cxzero), np.bincount(cyzero)
    wxzero, wyzero = np.where(bxzero == cvsize)[0], np.where(byzero == cvsize)[0]
    zrocol = np.intersect1d(wxzero, wyzero) # This is the list of columns (or rows), where all elements are zero
    # Create a mask for the non-zero elements
    mask = np.zeros_like(covar)
    mask[:, zrocol], mask[zrocol, :] = 1, 1
    cvnz = np.zeros((cvsize-zrocol.size, cvsize-zrocol.size))
    cvnz[np.where(cvnz == 0.0)] = covar[np.where(mask == 0.0)]
    # Generate a new set of starting parameters from the covariance matrix
    X_covar_fit = np.matrix(np.random.standard_normal((cvnz.shape[0], num)))
    C_covar_fit = np.matrix(cvnz)
    U_covar_fit = np.linalg.cholesky(C_covar_fit)
    Y_covar_fit = U_covar_fit * X_covar_fit
    return Y_covar_fit


def prepare_fitting(atom_prop, wave, spec, include_em=False, include_ab=True, npoly=2, p0c=None, p0a=None, p0e=None, stellar=False):
    param_info = []
    param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'step': 0}
    # Get some starting info from the input
    cont = max(np.median(spec), 0.0)
    ampl = max(np.max(spec) - cont, 0.0)
    wcen = wave[np.argmax(spec)]
    zabs = wcen / atom_prop['wave'] - 1
    sigm = 2 * (wave[1] - wave[0])

    # Set up the continuum
    idx = np.zeros(npoly, dtype=int)
    if p0c is None:
        p0c = np.zeros(npoly)
        p0c[-1] = cont
    cntr = 0
    for i in range(npoly):
        param_info.append(copy.deepcopy(param_base))
        param_info[cntr + i]['value'] = p0c[i]
    cntr += len(p0c)
    pinit = np.copy(p0c)

    # Set up the absorption
    if include_ab:
        idx = np.append(idx, 1 * np.ones(6, dtype=int))
        if p0a is None:
            if stellar:
                p0a = np.array([25000, zabs, 30, atom_prop['wave'], atom_prop['fval'], atom_prop['lGamma']])
            else:
                p0a = np.array([13.0, zabs, 400.0, atom_prop['wave'], atom_prop['fval'], atom_prop['lGamma']])
        for i in range(len(p0a)):
            param_info.append(copy.deepcopy(param_base))
            param_info[cntr + i]['value'] = p0a[i]
            if i == 0:
                if stellar:
                    param_info[cntr + i]['limited'] = [1, 1]
                    param_info[cntr + i]['limits'] = [3500, 30000]
                else:
                    param_info[cntr + i]['limited'] = [1, 0]
                    param_info[cntr + i]['limits'] = [0, 0]
                if p0a[0] in [0,14.611626,15.02934536,13.94769654]: param_info[cntr + i]['fixed'] = 1
            elif i == 1:
                if p0a[0] in [0,14.611626,15.02934536,13.94769654]: param_info[cntr + i]['fixed'] = 1
            elif i == 2:
                if stellar:
                    param_info[cntr + i]['limited'] = [1, 1]
                    param_info[cntr + i]['limits'] = [0, 50]
                else:
                    param_info[cntr + i]['limited'] = [1, 0]
                    param_info[cntr + i]['limits'] = [1.0E-11, 0]
                if p0a[0] in [0,14.611626,15.02934536,13.94769654]: param_info[cntr + i]['fixed'] = 1
            elif i == 3:
                param_info[cntr + i]['fixed'] = 1
            elif i == 4:
                param_info[cntr + i]['fixed'] = 1
            elif i == 5:
                param_info[cntr + i]['fixed'] = 1  # Fix the damping constant for now - probably not needed.
                param_info[cntr + i]['limited'] = [1, 0]
                param_info[cntr + i]['limits'] = [0, 0]
        cntr += len(p0a)
        pinit = np.append(pinit, p0a.copy())

    # Set up the emission
    if include_em:
        idx = np.append(idx, 2 * np.ones(3, dtype=int))
        if p0e is None:
            p0e = np.array([ampl, wcen, sigm])
        for i in range(len(p0e)):
            param_info.append(copy.deepcopy(param_base))
            param_info[cntr + i]['value'] = p0e[i]
            if i == 0:
                param_info[cntr + i]['limited'] = [1, 0]
                param_info[cntr + i]['limits'] = [0, 0]
            elif i == 1:
                pass
            elif i == 2:
                param_info[cntr + i]['limited'] = [1, 0]
                param_info[cntr + i]['limits'] = [0.01, 0]
        cntr += len(p0e)
        pinit = np.append(pinit, p0e.copy())
    # Return everything we need
    return pinit, param_info, idx.astype(int)


def func_voigt(par, wavein):
    gama = 10.0**par[5]
    wv = par[3] * 1.0e-8
    cold = 10.0**par[0]
    zp1 = par[1]+1.0
    bl = par[2]*wv/2.99792458E5
    a = gama*wv*wv/(3.76730313461770655E11*bl)
    cns = wv*wv*par[4]/(bl*2.002134602291006E12)
    cne = cold*cns
    ww = (wavein*1.0e-8)/zp1
    v = wv * ((wv / ww) - 1) / bl
    tau = cne*wofz(v + 1j * a).real
    return np.exp(-1.0*tau)


def func_stellarabs(par, wavein, mgrid):
    tval = np.ones(wavein.size) * par[0]
    gval = np.ones(wavein.size) * par[2]
    return mgrid(np.column_stack((tval,gval,wavein/(1+par[1]))))


def func_gauss_oned(p, x):
    amp, cen, sig = p[0], p[1], p[2]
    y = (amp/(sig*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2/(2.0*sig**2))
    return y


def func_cont(p, x):
    # First make the stellar continuum
    cont = np.polyval(p, x)
    return cont


def extract_params(p, idx):
    cp = p[idx == 0]  # continuum
    ap = p[idx == 1]  # absorption
    ep = p[idx == 2]  # emission
    return cp, ap, ep


def full_model(p, wave, idx, stellar=None):
    # Extract the parameters of the different functions
    cp, ap, ep = extract_params(p, idx)
    # First make the stellar continuum
    cont = func_cont(cp, wave)
    # Now model the stellar absorption as a voigt profile
    absp = 1
    if np.any(idx == 1):
        if stellar is None:
            absp = func_voigt(ap, wave)
        else:
            absp = func_stellarabs(ap, wave, stellar)
    # Obtain a model of a Gaussian
    emis = 0.0
    if np.any(idx == 2):
        emis = func_gauss_oned(ep, wave)
    # Combine the full model
    model = emis + cont*absp
    # Return it
    return model


def resid(p, fjac=None, wave=None, flux=None, errs=None, idx=None, stellar=None):
    # Calculate the model
    model = full_model(p, wave, idx, stellar=stellar)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    resid = (model - flux) / errs
    return [status, resid]


def fit_one_cont(atom_prop, wave, spec, errs, mask,
                 npoly=2, contsample=100, verbose=True,
                 p0c=None, p0a=None, p0e=None,
                 include_em=False, include_ab=True):
    # Perform a fit
    ww = np.where((mask == 1) & (errs != 0.0) & (spec != 0.0))
    if ww[0].size <= 5:
        return None, None, None, None
    fitspec = spec[ww]
    fiterrs = errs[ww]
    fitwave = wave[ww]

    # Initialise the fitting
    pinit, param_info, idx = prepare_fitting(atom_prop, fitwave, fitspec,
                                             npoly=npoly, p0c=p0c, p0a=p0a, p0e=p0e,
                                             include_ab=include_ab, include_em=include_em)
    # Now tell the fitting program what we called our variables
    fa = {'wave': fitwave, 'flux': fitspec, 'errs': fiterrs, 'idx': idx}

    if verbose: print("Fitting continuum")
    m = mpfit.mpfit(resid, pinit, parinfo=param_info, functkw=fa, quiet=True)

    # Make the emission mask
    emis_gpm = np.zeros(wave.size)
    emis_gpm[np.where((wave >= np.min(fitwave)) & (wave <= np.max(fitwave)))] = 1
    emis_gpm -= mask

    # Subtract the continuum and then sum everything above zero
    wdiff = np.append(wave[1] - wave[0], np.diff(wave)) * emis_gpm  # Fold in the mask here to save computation in the loop
    tmp_sum = 0.0
    try:
        if verbose: print("Sampling continuum")
        tmp_sum = np.zeros(contsample)
        ptb = newstart(m.covar, contsample)
        contpars = np.outer(m.params, np.ones(contsample)) + ptb
        for ss in range(contsample):
            cont = full_model(contpars[:, ss], wave, idx)
            specnew = spec - cont
            # Integrate over the masked region
            tmp_sum[ss] = np.sum(specnew * wdiff)
        scale = 1
    except:
        # Cont fit errors failed - just double the error, probably it's just noise.
        print("ERROR calculating total flux and error")
        scale = 2
    if verbose: print("Calculating flux and error")
    # Calculate the best values
    contabs = full_model(m.params, wave, idx)
    specnew = spec - contabs
    # Now get just the continuum value
    cpar, _, _ = extract_params(m.params, idx)
    cont = func_cont(cpar, wave)
    cont_val = cont[np.argmax(spec)]  # Pick the continuum where the flux is maximum
    # Integrate over the masked regions (the mask is included in wdiff)
    sum_flux = np.sum(specnew * wdiff)
    # Add in the continuum error
    sum_err = scale * np.sqrt(np.sum((errs * wdiff) ** 2) + np.std(tmp_sum) ** 2)
    return sum_flux, sum_err, cont_val, m.params


def fit_stellar_abs(atom_prop, wave, spec, errs, mask, grating='BH2',
                    npoly=2, contsample=100, verbose=True,
                    p0c=None, p0a=None, p0e=None, quiet=True,
                    include_em=False, include_ab=True):
    # Perform a fit
    ww = np.where((mask == 1) & (errs != 0.0) & (spec != 0.0))
    if ww[0].size <= 5:
        return None, None, None, None
    fitspec = spec[ww]
    fiterrs = errs[ww]
    fitwave = wave[ww]

    # Initialise the fitting
    pinit, param_info, idx = prepare_fitting(atom_prop, fitwave, fitspec, stellar=True,
                                             npoly=npoly, p0c=p0c, p0a=p0a, p0e=p0e,
                                             include_ab=include_ab, include_em=include_em)

    # Make the spline representation of the stellar absorption
    tgrid, ggrid, wgrid, abs_spl = get_bohlin_spline(grating, atom_prop['line'])

    # Now tell the fitting program what we called our variables
    fa = {'wave': fitwave, 'flux': fitspec, 'errs': fiterrs, 'idx': idx, 'stellar':abs_spl}

    if verbose: print("Fitting continuum and stellar absorption")
    m = mpfit.mpfit(resid, pinit, parinfo=param_info, functkw=fa, quiet=quiet)

    wg = np.where((wave>np.min(fitwave)) & (wave<np.max(fitwave)))
    wavefin = wave[wg]
    specfin = spec[wg]
    errsfin = errs[wg]
    # Make the emission mask
    emis_gpm = np.zeros(wavefin.size)
    emis_gpm[np.where((wavefin >= np.min(fitwave)) & (wavefin <= np.max(fitwave)))] = 1
    emis_gpm -= mask[wg]

    # Subtract the continuum and then sum everything above zero
    wdiff = np.append(wavefin[1] - wavefin[0], np.diff(wavefin)) * emis_gpm  # Fold in the mask here to save computation in the loop
    tmp_sum = 0.0
    try:
        if verbose: print("Sampling continuum")
        tmp_sum = np.zeros(contsample)
        ptb = np.append(newstart(m.covar, contsample), np.zeros((6,contsample)), axis=0)
        contpars = (np.outer(m.params, np.ones(contsample)) + ptb)
        for ss in range(contsample):
            part = np.squeeze(np.asarray(contpars[:, ss]))
            if (part[npoly]>np.min(tgrid)) and (part[npoly]<np.max(tgrid)) and \
                (part[npoly+2]>np.min(ggrid)) and (part[npoly+2]<np.max(ggrid)):
                cont = full_model(part, wavefin, idx, stellar=abs_spl)
                specnew = specfin - cont
                # Integrate over the masked region
                tmp_sum[ss] = np.sum(specnew * wdiff)
        scale = 1
    except:
        # Cont fit errors failed - just double the error, probably it's just noise.
        print("ERROR calculating total flux and error")
        scale = 2
    if verbose: print("Calculating flux and error")
    # Calculate the best values
    contabs = full_model(m.params, wavefin, idx, stellar=abs_spl)
    specnew = specfin - contabs
    # Now get just the continuum value
    cpar, _, _ = extract_params(m.params, idx)
    cont = func_cont(cpar, wavefin)
    cont_val = cont[np.argmax(specfin)]  # Pick the continuum where the flux is maximum
    # Integrate over the masked regions (the mask is included in wdiff)
    sum_flux = np.sum(specnew * wdiff)
    # Add in the continuum error
    sum_err = scale * np.sqrt(np.sum((errsfin * wdiff) ** 2) + np.std(tmp_sum[tmp_sum!=0]) ** 2)
    return sum_flux, sum_err, cont_val, m.params


def fit_lorentz_abs(atom_prop, wave, spec, errs, mask, grating='BH2',
                    npoly=2, contsample=100, verbose=True,
                    p0c=None, p0a=None, p0e=None, quiet=True,
                    include_em=False, include_ab=True):
    # Perform a fit
    ww = np.where((mask == 1) & (errs != 0.0) & (spec != 0.0))
    if ww[0].size <= 5:
        return None, None, None, None
    fitspec = spec[ww]
    fiterrs = errs[ww]
    fitwave = wave[ww]

    # Initialise the fitting
    pinit, param_info, idx = prepare_fitting(atom_prop, fitwave, fitspec, stellar=False,
                                             npoly=npoly, p0c=p0c, p0a=p0a, p0e=p0e,
                                             include_ab=include_ab, include_em=include_em)

    # Now tell the fitting program what we called our variables
    fa = {'wave': fitwave, 'flux': fitspec, 'errs': fiterrs, 'idx': idx}

    if verbose: print("Fitting continuum and stellar absorption")
    m = mpfit.mpfit(resid, pinit, parinfo=param_info, functkw=fa, quiet=quiet)

    wg = np.where((wave>np.min(fitwave)) & (wave<np.max(fitwave)))
    wavefin = wave[wg]
    specfin = spec[wg]
    errsfin = errs[wg]
    # Make the emission mask
    emis_gpm = np.zeros(wavefin.size)
    emis_gpm[np.where((wavefin >= np.min(fitwave)) & (wavefin <= np.max(fitwave)))] = 1
    emis_gpm -= mask[wg]

    # Subtract the continuum and then sum everything above zero
    wdiff = np.append(wavefin[1] - wavefin[0], np.diff(wavefin)) * emis_gpm  # Fold in the mask here to save computation in the loop
    tmp_sum = 0.0
    try:
        if verbose: print("Sampling continuum")
        tmp_sum = np.zeros(contsample)
        ptb = np.zeros((m.params.size, contsample))
        ptb[np.where(m.perror != 0)[0], :] = newstart(m.covar, contsample)
        contpars = (np.outer(m.params, np.ones(contsample)) + ptb)
        for ss in range(contsample):
            part = np.squeeze(np.asarray(contpars[:, ss]))
            cont = full_model(part, wavefin, idx)
            specnew = specfin - cont
            # Integrate over the masked region
            tmp_sum[ss] = np.sum(specnew * wdiff)
        scale = 1
    except:
        # Cont fit errors failed - just double the error, probably it's just noise.
        embed()
        print("ERROR calculating total flux and error")
        scale = 2
    if verbose: print("Calculating flux and error")
    # Calculate the best values
    contabs = full_model(m.params, wavefin, idx)
    specnew = specfin - contabs
    # Now get just the continuum value
    cpar, _, _ = extract_params(m.params, idx)
    cont = func_cont(cpar, wavefin)
    cont_val = cont[np.argmax(specfin)]  # Pick the continuum where the flux is maximum
    # Integrate over the masked regions (the mask is included in wdiff)
    sum_flux = np.sum(specnew * wdiff)
    # Add in the continuum error
    sum_err = scale * np.sqrt(np.sum((errsfin * wdiff) ** 2) + np.std(tmp_sum[tmp_sum!=0]) ** 2)
    return sum_flux, sum_err, cont_val, m.params


def get_bohlin_spline(grating, line):
    tgrid = np.load("Bohlin2017_Tgrid.npy")
    ggrid = np.load("Bohlin2017_Ggrid.npy")
    wgrid = np.load(f"Bohlin2017_Wgrid_{grating}_{line}.npy")
    mgrid = np.load(f"Bohlin2017_Mgrid_{grating}_{line}.npy")
    abs_spl = RegularGridInterpolator((tgrid, ggrid, wgrid), mgrid)
    return tgrid, ggrid, wgrid, abs_spl

def robust_stats(arr):
    med = np.median(arr)
    mad = 1.4826 * np.median(np.abs(arr-med))
    return med, mad