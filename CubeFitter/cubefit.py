from datetime import datetime
import os
import copy
import numpy as np
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.image import NonUniformImage
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.transforms as mtransforms
from matplotlib.widgets import Button, Slider

from scipy.ndimage import gaussian_filter1d
import astropy.io.fits as fits
from astropy.wcs import WCS
from atomic_info import GetAtomProp
import fitting

from IPython import embed

operations = dict({'cursor': "Select lines (LMB click)\n" +
                    "         Select regions (LMB drag = add, RMB drag = remove)\n" +
                    "         Navigate (LMB drag = pan, RMB drag = zoom)",
                   'left'  : "Advance the line list slider to the left by one",
                   'right' : "Advance the line list slider to the right by one",
                   'p' : "Toggle pan/zoom with the cursor",
                   'q' : "Close Identify window and continue PypeIt reduction",
                   'c' : "Mark spaxel as complete",
                   'f' : "Perform a fit",
                   'l' : "Reload the current map",
                   'm' : "Set the fit parameters",
                   'n' : "Reset the m cycle to zero (i.e. start a new fit)",
                   'o' : "Print parameters of the current fit to the command line",
                   's' : "Save the current map",
                   'y' : "Toggle the y-axis scale between logarithmic and linear",
                   'z' : "Zap a spaxel - delete its model parameters...",
                   '+/-' : "Raise/Lower the order of the fitting polynomial"
                   })

# Define some axis variables
AXMAIN = 0
AXRESID = 1
AXZOOM = 2
AXINFO = 3
AXFLUXMAP = 4
AXWHITELIGHT = 5
AXCOMPLETE = 6


class CubeFitter:
    """
    GUI to interactively fit emission and absorption lines in a datacube.
    """

    def __init__(self, canvas, axes, specim, wave, datacube, sigcube, mskcube, all_maps, map_name, idx, idy, atomprop, include_ab=True, y_log=True, npoly=2):
        """A hacked script to (re)fit emission lines in a datacube manually, and estimate the total emission line flux.

        The main goal of this routine is to interactively identify arc lines
        to be used for wavelength calibration.

        Parameters
        ----------
        canvas : Matploltib figure canvas
            The canvas on which all axes are contained
        axes : dict
            Dictionary of four Matplotlib axes instances (Main spectrum panel, one for zoom in, one for residuals, one for information, and one for the image)
        specim : dict
            Dictionary of several Matplotlib instances (Line2D and Image) which contains plotting information of the plotted arc spectrum
        y_log : bool, optional
            Scale the Y-axis logarithmically instead of linearly?  (Default: True)
        """
        # Store the axes
        self.axes = axes
        # Initialise the spectrum properties
        self.specim = specim
        self.maps = all_maps
        self.map_name = map_name
        self.spec = specim['spec']#.get_ydata()
        self.speczoom = specim['speczoom']#.get_ydata()
        self.model = specim['model']#.get_ydata()
        self.resid = specim['resid']#.get_ydata()
        self.image = specim['im']#.get_data()
        self.wlimage = specim['imwl']
        self.complete = specim['imcomp']
        self._imxarr = np.arange(self.maps['complete'].shape[1])
        self._imyarr = np.arange(self.maps['complete'].shape[0])
        self.pt_map = specim['pt_map']
        self.pt_wl = specim['pt_wl']
        self.pt_comp = specim['pt_comp']
        self.curr_wave = wave
        self.curr_flux = datacube[:, idx, idy]
        self.curr_err = sigcube[:, idx, idy]
        self.y_log = y_log
        self._atomprop = atomprop
        # datacube
        self.datacube = datacube
        self.sigcube = sigcube
        self.maskcube = mskcube
        # Fitting properties
        self._fitdict = dict(model=None)
        # Unset some of the matplotlib keymaps
        matplotlib.pyplot.rcParams['keymap.fullscreen'] = ''        # toggling fullscreen (Default: f, ctrl+f)
        matplotlib.pyplot.rcParams['keymap.home'] = ''              # home or reset mnemonic (Default: h, r, home)
        matplotlib.pyplot.rcParams['keymap.back'] = ''              # forward / backward keys to enable (Default: left, c, backspace)
        matplotlib.pyplot.rcParams['keymap.forward'] = ''           # left handed quick navigation (Default: right, v)
        #matplotlib.pyplot.rcParams['keymap.pan'] = ''              # pan mnemonic (Default: p)
        matplotlib.pyplot.rcParams['keymap.zoom'] = ''              # zoom mnemonic (Default: o)
        matplotlib.pyplot.rcParams['keymap.save'] = ''              # saving current figure (Default: s)
        matplotlib.pyplot.rcParams['keymap.quit'] = ''              # close the current figure (Default: ctrl+w, cmd+w)
        matplotlib.pyplot.rcParams['keymap.grid'] = ''              # switching on/off a grid in current axes (Default: g)
        matplotlib.pyplot.rcParams['keymap.grid_minor'] = ''        # switching on/off a (minor) grid in current axes (Default: G)
        matplotlib.pyplot.rcParams['keymap.yscale'] = ''            # toggle scaling of y-axes ('log'/'linear') (Default: l)
        matplotlib.pyplot.rcParams['keymap.xscale'] = ''            # toggle scaling of x-axes ('log'/'linear') (Default: L, k)
        #matplotlib.pyplot.rcParams['keymap.all_axes'] = ''          # enable all axes (Default: a)

        # Initialise the main canvas tools
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_event)
        self.canvas = canvas
        self.background = self.canvas.copy_from_bbox(self.axes['main'].bbox)

        # Interaction variables
        self._idx, self._idy = idx, idy
        self._nx, self._ny = self.datacube.shape[1], self.datacube.shape[2]
        self._coord = [idx, idy]
        self._fitr, self._fitrzoom, self._fitrresid = None, None, None  # Matplotlib shaded fit region (for refitting lines)
        self._fitregions = self.maskcube[:, idx, idy]  # Mask of the pixels to be included in a fit
        self._include_ab = include_ab
        self._npoly = npoly
        self._mcycle = 0
        self._modpar = np.zeros(6)
        self._p0c, self._p0a, self._p0e = None, None, None
        self._addsub = 0   # Adding a region (1) or removing (0)
        self._msedown = False  # Is the mouse button being held down (i.e. dragged)
        self._respreq = [False, None]  # Does the user need to provide a response before any other operation will be permitted? Once the user responds, the second element of this array provides the action to be performed.
        self._qconf = False  # Confirm quit message
        self._changes = False

        # Setup slider for the linelist
        self.image_scale_init()

        # Draw the spectrum
        self.replot()

    @classmethod
    def initialise(cls, datcube, sigcube, wave, line, zem, dirc, fname, refit=False, include_ab=False, npoly=2, y_log=False):
        """Initialise the 'CubeFitter' window for manual fitting of continuum near emission lines

        Parameters
        ----------
        y_log : bool, optional
            Scale the Y-axis logarithmically instead of linearly?  (Default: True)

        Returns
        -------
        object : :class:`CubeFitter`
            Returns an instance of the :class:`Identify` class, which contains the results of the fit
        """
        # Get the details of the line to be fitted
        atom_prop = GetAtomProp(line)
        waveobs = atom_prop['wave']*(1+zem)
        delwave = 15.0

        # Make a whitelight image, and the initial
        whitelight = np.sum(datcube, axis=0)/np.sum(sigcube == 0, axis=0)
        whitelight /= np.max(whitelight)
        flx_map = np.zeros_like(whitelight)
        err_map = np.zeros_like(whitelight)
        cnt_map = np.zeros_like(whitelight)
        comp_map = np.ones_like(whitelight)
        par_map = None

        # # TODO :: REMOVE -- Load up the Hd maps as starting parameters for the Hg maps
        # all_maps, mskcube = load_maps(get_mapname(dirc, fname, "HId"))
        # all_maps['params'][6, :, :] = 4341.691
        # all_maps['params'][7, :, :] = 4.4694e-02
        # mskcube = np.roll(mskcube, 2081, axis=0)
        # for xx in range(datcube.shape[1]):
        #     for yy in range(datcube.shape[2]):
        #         amtmp = np.where(mskcube[:,xx,yy]==1)[0]
        #         if amtmp.size != 0:
        #             mskcube[amtmp[-1]-20:,xx,yy] = 0

        # TODO :: REMOVE -- This was a fix to convert the fits from legendre to polyonomial
        # mapname = get_mapname(dirc, fname, line)
        # all_maps, mskcube = load_maps(mapname)
        # for xx in range(datcube.shape[1]):
        #     for yy in range(datcube.shape[2]):
        #         flx, err, msk = datcube[:, xx, yy], sigcube[:, xx, yy], mskcube[:, xx, yy]
        #         ww = np.where((msk == 1) & (err != 0.0) & (flx != 0.0))
        #         if ww[0].size <= 5:
        #             continue
        #         fitwave = wave[ww]
        #         wred = (fitwave - fitwave[0]) / (fitwave[-1] - fitwave[0])
        #         cont = np.polyval(all_maps['params'][:npoly, xx, yy], wred)
        #         newpar = np.polyfit(fitwave, cont, npoly-1)
        #         all_maps['params'][:npoly, xx, yy] = newpar
        # save_maps(mapname, all_maps['flux'], all_maps['errs'], all_maps['cont'], all_maps['complete'], all_maps['params'], mskcube)

        mapname = get_mapname(dirc, fname, line)
        if refit:
            #mskcube = make_mask(wave, datcube, waveobs, delwave)
            for xx in range(datcube.shape[1]):
                print(f"fitting row {xx+1}/{datcube.shape[1]}")
                for yy in range(datcube.shape[2]):
                    p0c, p0a = None, None
                    if False:
                        # Refitting the data
                        if all_maps['params'][0, xx, yy] == 0:
                            continue
                        # Update mskcube
                        #ww = np.where((wave>4315)&(wave<4340))[0]
                        #mskcube[ww, xx, yy] = 1
                        p0c = all_maps['params'][:npoly, xx, yy]
                        p0a = all_maps['params'][npoly:, xx, yy]
                    flx, err, msk = datcube[:, xx, yy], sigcube[:, xx, yy], mskcube[:, xx, yy]
                    flxsum, errsum, contval, pars = fitting.fit_one_cont(atom_prop, wave, flx, err, msk, npoly=npoly, contsample=100, verbose=False, include_ab=include_ab, p0c=p0c, p0a=p0a)
                    if flxsum is None:
                        # Something failed.
                        continue
                    flx_map[xx, yy] = flxsum
                    err_map[xx, yy] = errsum
                    cnt_map[xx, yy] = contval
                    comp_map[xx, yy] = 0
                    if par_map is None:
                        par_map = np.zeros((pars.size, flx_map.shape[0], flx_map.shape[1]))
                    par_map[:, xx, yy] = pars
            save_maps(mapname, flx_map, err_map, cnt_map, comp_map, par_map, mskcube)
        # Load the saved maps to check it worked, and put it in the correct format
        all_maps, mskcube = load_maps(mapname)
        print("Spectra left to fit:", int(np.sum(1-all_maps['complete'])))

        # Set the starting location, and generate the model at this location
        idx, idy = datcube.shape[1]//2, datcube.shape[2]//2
        _, _, index = fitting.prepare_fitting(atom_prop, wave, datcube[:, idx, idy], npoly=npoly, include_ab=include_ab)
        model = fitting.full_model(all_maps['params'][:,idx,idy], wave, index)
        # Create a Line2D instance for the spectrum
        spec = Line2D(wave, datcube[:, idx, idy],
                      linewidth=1, linestyle='solid', color='k',
                      drawstyle='steps-mid', animated=True)

        speczoom = Line2D(wave, datcube[:, idx, idy],
                          linewidth=1, linestyle='solid', color='k',
                          drawstyle='steps-mid', animated=True)

        specfit = Line2D(wave, model,
                         linewidth=1, linestyle='solid', color='r',
                         animated=True)

        resid = Line2D(wave, (datcube[:, idx, idy]-model)*inverse(sigcube[:, idx, idy]),
                      linewidth=1, linestyle='solid', color='k',
                      drawstyle='steps-mid', animated=True)

        # Add the main figure axis
        fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
        plt.subplots_adjust(bottom=0.5, top=0.85, left=0.05, right=0.6)
        ax.add_line(spec)
        if y_log:
            ax.set_yscale('log')
            ax.set_ylim( (max(1., spec.get_ydata().min()), 4.0 * spec.get_ydata().max()))
        else:
            ax.set_yscale('linear')
            ax.set_ylim((0.0, 1.1 * spec.get_ydata().max()))
        ax.set_ylabel('Flux')
        ax.set_xlim((waveobs-delwave, waveobs+delwave))

        # Add an image of the map
        xarr = np.arange(whitelight.shape[1])
        yarr = np.arange(whitelight.shape[0])
        axmap = fig.add_axes([0.65, .5, .2, .2*16/9])
        im = NonUniformImage(axmap, interpolation='nearest', origin='lower', cmap=cm.inferno)
        im.set_data(xarr, yarr, all_maps['flux'])
        im.set_clim(vmin=0, vmax=np.max(all_maps['flux']))
        im.set_extent((0, xarr.size-1, 0, yarr.size-1))
        mappt = axmap.scatter([idx], [idy], marker='x', color='b')

        # Add a whitelight image
        axwl = fig.add_axes([0.65, .1, .2, .2*16/9])
        imwl = NonUniformImage(axwl, interpolation='nearest', origin='lower', cmap=cm.inferno)
        imwl.set_data(xarr, yarr, whitelight)
        imwl.set_clim(vmin=0, vmax=np.max(whitelight))
        imwl.set_extent((0, xarr.size-1, 0, yarr.size-1))
        wlpt = axwl.scatter([idx], [idy], marker='x', color='b')

        # Add a completeness image
        axcomp = fig.add_axes([0.86, .6, .13, .13*16/9])
        imcomp = NonUniformImage(axcomp, interpolation='nearest', origin='lower', cmap=cm.bwr_r)
        imcomp.set_data(xarr, yarr, all_maps['complete'])
        imcomp.set_clim(vmin=0, vmax=1)
        imcomp.set_extent((0, xarr.size-1, 0, yarr.size-1))
        comppt = axcomp.scatter([idx], [idy], marker='x', color='y')

        # Add two residual fitting axes
        axfit = fig.add_axes([0.05, .22, .55, 0.25])
        axfit.sharex(ax)
        axres = fig.add_axes([0.05, .05, .55, 0.17])
        axres.sharex(ax)

        # Residuals
        axres.axhspan(-2, 2, alpha=0.5, color='grey')
        axres.axhspan(-1, 1, alpha=0.5, color='darkgrey')
        axres.axhline(0.0, color='r', linestyle='-')  # Zero level
        axres.add_line(resid)
        axres.set_ylim((-3, 3))
        axres.set_xlabel('Wave')
        axres.set_ylabel('Residual')

        # Zoom in for continuum
        axfit.add_line(speczoom)
        axfit.add_line(specfit)
        axfit.set_ylim((0, 0.5))  # This will get updated as lines are identified
        axfit.set_ylabel('Flux')

        # Add an information GUI axis
        axinfo = fig.add_axes([0.15, .92, .7, 0.07])
        axinfo.get_xaxis().set_visible(False)
        axinfo.get_yaxis().set_visible(False)
        axinfo.text(0.5, 0.5, "Press '?' to list the available options", transform=axinfo.transAxes,
                    horizontalalignment='center', verticalalignment='center')
        axinfo.set_xlim((0, 1))
        axinfo.set_ylim((0, 1))

        axes = dict(main=ax, fit=axfit, resid=axres, info=axinfo, fmap=axmap, fwl=axwl, fcomp=axcomp)
        specim = dict(im=im, imwl=imwl, imcomp=imcomp, spec=spec, speczoom=speczoom, model=specfit, resid=resid, pt_map=mappt, pt_wl=wlpt, pt_comp=comppt)
        # Initialise the identify window and display to screen
        fig.canvas.set_window_title('CubeFitter')
        fitter = CubeFitter(fig.canvas, axes, specim, wave, datcube, sigcube, mskcube, all_maps, mapname, idx, idy, atom_prop, include_ab=include_ab, npoly=npoly, y_log=y_log)

        plt.show()

        # Now return the results
        return fitter

    def image_scale_init(self):
        """Initialise the linelist Slider (used to assign a line to a detection)
        """
        axcolor = 'lightgoldenrodyellow'
        # Slider
        self.axl = plt.axes([0.15, 0.87, 0.7, 0.04], facecolor=axcolor)
        self._slideis = Slider(self.axl, "Image scale", -2, 3, valinit=0, valstep=0.01)
        self._slideis.valtext.set_visible(False)
        self._slideis.on_changed(self.update_image_scale)

    def print_help(self):
        """Print the keys and descriptions that can be used for Identification
        """
        keys = operations.keys()
        print("===============================================================")
        # print(" Colored lines in main panels:")
        # print("   gray   : wavelength has not been assigned to this detection")
        # print("   red    : currently selected line")
        # print("   blue   : user has assigned wavelength to this detection")
        # print("   yellow : detection has been automatically assigned")
        # print(" Colored symbols in residual panels:")
        # print("   gray   : wavelength has not been assigned to this detection")
        # print("   blue   : user has assigned wavelength to this detection")
        # print("   yellow : detection has been automatically assigned")
        # print("   red    : automatically assigned wavelength was rejected")
        print("---------------------------------------------------------------")
        print("       IDENTIFY OPERATIONS")
        for key in keys:
            print("{0:6s} : {1:s}".format(key, operations[key]))
        print("---------------------------------------------------------------")

    def replot(self):
        """Redraw the entire canvas
        """
        # First set the xdata to be shown
        self.canvas.restore_region(self.background)
        self.draw_residuals()
        self.canvas.draw()

    def toggle_yscale(self):
        self.y_log = not self.y_log
        # Update the y-axis scale and axis range
        if self.y_log:
            self.axes['main'].set_yscale('log')
            self.axes['main'].set_ylim((max(1., self.spec.get_ydata().min()),
                                       4.0 * self.spec.get_ydata().max()))
        else:
            self.axes['main'].set_yscale('linear')
            self.axes['main'].set_ylim((0.0, 1.1 * self.spec.get_ydata().max()))

    def draw_residuals(self):
        """Update the subplots that show the residuals
        """
        if self._fitdict["model"] is None:
            msg = "Cannot plot residuals until a fit has been performed"
            self.update_infobox(message=msg, yesno=False)
        else:
            # Extract the fitting info
            model = self._fitdict['model']

            # Pixel vs wavelength
            self.model.set_ydata(model)
            self.resid.set_ydata((self.curr_flux-model)/self.curr_err)

            # Pixel residuals
            self.axes['resid'].set_ylim((-3.0, 3.0))

            # Write some statistics on the plot
            # disptxt = r'$\Delta\lambda$={:.3f}$\AA$ (per pix)'.format(dwv_pix)
            # rmstxt = 'RMS={:.3f} (pixels)'.format(self._fitdict['rms'])
            # self._fitdict["res_stats"].append(self.axes['fit'].text(0.1 * self.specdata.size,
            #                                                         ymin + 0.90 * (ymax - ymin),
            #                                                         disptxt, size='small'))
            # self._fitdict["res_stats"].append(self.axes['fit'].text(0.1 * self.specdata.size,
            #                                                         ymin + 0.80 * (ymax - ymin),
            #                                                         rmstxt, size='small'))

    def draw_callback(self, event):
        """Draw the lines and annotate with their IDs

        Args:
            event (Event): A matplotlib event instance
        """
        # Get the background
        self.background = self.canvas.copy_from_bbox(self.axes['main'].bbox)
        # Set the axis transform
        trans = mtransforms.blended_transform_factory(self.axes['main'].transData, self.axes['main'].transAxes)
        transZoom = mtransforms.blended_transform_factory(self.axes['fit'].transData, self.axes['fit'].transAxes)
        transResid = mtransforms.blended_transform_factory(self.axes['resid'].transData, self.axes['resid'].transAxes)
        self.draw_fitregions(trans, transZoom, transResid)
        self.axes['main'].draw_artist(self.spec)
        self.axes['fit'].draw_artist(self.speczoom)
        self.axes['fit'].draw_artist(self.model)
        self.axes['resid'].draw_artist(self.resid)
        self.axes['fmap'].draw_artist(self.image)
        self.axes['fmap'].draw_artist(self.pt_map)
        self.axes['fwl'].draw_artist(self.wlimage)
        self.axes['fwl'].draw_artist(self.pt_wl)
        self.axes['fcomp'].draw_artist(self.complete)
        self.axes['fcomp'].draw_artist(self.pt_comp)

    def draw_fitregions(self, transMain, transZoom, transResid):
        """Refresh the fit regions

        Args:
            trans (AxisTransform): A matplotlib axis transform from data to axes coordinates
        """
        if self._fitr is not None:
            self._fitr.remove()
        # Find all regions
        regwhr = np.copy(self._fitregions == 1)
        # Fudge to get the leftmost pixel shaded in too
        regwhr[np.where((self._fitregions[:-1] == 0) & (self._fitregions[1:] == 1))] = True
        self._fitr = self.axes['main'].fill_between(self.curr_wave, 0, 1, where=regwhr, facecolor='green',
                                                    alpha=0.5, transform=transMain)
        # Do the zoom in plot
        if self._fitrzoom is not None:
            self._fitrzoom.remove()
        self._fitrzoom = self.axes['fit'].fill_between(self.curr_wave, 0, 1, where=regwhr, facecolor='green',
                                                    alpha=0.5, transform=transZoom)
        # Do the residual plot
        if self._fitrresid is not None:
            self._fitrresid.remove()
        self._fitrresid = self.axes['resid'].fill_between(self.curr_wave, 0, 1, where=regwhr, facecolor='green',
                                                    alpha=0.5, transform=transResid)

    def get_xidx_under_point(self, event):
        """Get the index of the line closest to the cursor

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            ind (int): Index of the spectrum where the event occurred
        """
        ind = np.argmin(np.abs(self.curr_wave - event.xdata))
        return ind

    def get_coord_under_point(self, event):
        """Get the index of the line closest to the cursor

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            list (list): Indices of the image under the cursor
        """
        return [int(round(event.xdata)), int(round(event.ydata))]

    def shift_coord(self, dirn):
        newx, newy = self._coord[0], self._coord[1]
        if dirn == 'left': newx -= 1
        elif dirn == 'right': newx += 1
        elif dirn == 'up': newy += 1
        elif dirn == 'down': newy -= 1
        return [np.clip(newx,0,self._nx-1), np.clip(newy,0,self._ny-1)]

    def get_axisID(self, event):
        """Get the ID of the axis where an event has occurred

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            axisID (int, None): Axis where the event has occurred
        """
        if event.inaxes == self.axes['main']:
            return AXMAIN
        elif event.inaxes == self.axes['resid']:
            return AXRESID
        elif event.inaxes == self.axes['fit']:
            return AXZOOM
        elif event.inaxes == self.axes['info']:
            return AXINFO
        elif event.inaxes == self.axes['fmap']:
            return AXFLUXMAP
        elif event.inaxes == self.axes['fwl']:
            return AXWHITELIGHT
        elif event.inaxes == self.axes['fcomp']:
            return AXCOMPLETE
        return None

    def button_press_callback(self, event):
        """What to do when the mouse button is pressed

        Args:
            event (Event): Matplotlib event instance containing information about the event
        """
        if event.inaxes is None:
            return
        if self.canvas.toolbar.mode != "":
            return
        if event.button == 1:
            self._addsub = 1
        elif event.button == 3:
            self._addsub = 0
        if self.get_axisID(event) in [AXMAIN, AXZOOM]:
            self._msedown = True
        axisID = self.get_axisID(event)
        self._start = self.get_xidx_under_point(event)
        self._startdata = event.xdata

    def motion_notify_event(self, event):
        if event.inaxes is None:
            return
        self._middata = event.xdata

    def button_release_callback(self, event):
        """What to do when the mouse button is released

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            None
        """
        self._msedown = False
        if event.inaxes is None:
            return
        axisID = self.get_axisID(event)
        if axisID == AXINFO:
            if (event.xdata > 0.8) and (event.xdata < 0.9):
                answer = "y"
            elif event.xdata >= 0.9:
                answer = "n"
            else:
                return
            self.operations(answer, -1, None)
            self.update_infobox(default=True)
            return
        elif self._respreq[0]:
            # The user is trying to do something before they have responded to a question
            return
        if self.canvas.toolbar.mode != "":
            return
        # Draw an actor
        if axisID is not None:
            if axisID in [AXMAIN, AXZOOM]:
                self._end = self.get_xidx_under_point(event)
                if self._end == self._start:
                    # The mouse button was pressed (not dragged)
                    self.operations('m', axisID, event)
                elif self._end != self._start:
                    # The mouse button was dragged
                    if self._start > self._end:
                        tmp = self._start
                        self._start = self._end
                        self._end = tmp
                    self.update_regions()
            elif axisID in [AXFLUXMAP, AXWHITELIGHT, AXCOMPLETE]:
                self._coord = self.get_coord_under_point(event)
                # Update the spectrum that is being plotted
                self.update_spectrum()

        # Now plot
        trans = mtransforms.blended_transform_factory(self.axes['main'].transData, self.axes['main'].transAxes)
        transZoom = mtransforms.blended_transform_factory(self.axes['fit'].transData, self.axes['fit'].transAxes)
        transResid = mtransforms.blended_transform_factory(self.axes['resid'].transData, self.axes['resid'].transAxes)
        self.canvas.restore_region(self.background)
        self.draw_fitregions(trans, transZoom, transResid)
        # Now replot everything
        self.replot()

    def key_press_callback(self, event):
        """What to do when a key is pressed

        Args:
            event (Event): Matplotlib event instance containing information about the event

        Returns:
            None
        """
        # Check that the event is in an axis...
        if not event.inaxes:
            return
        # ... but not the information box!
        if event.inaxes == self.axes['info']:
            return
        axisID = self.get_axisID(event)
        self.operations(event.key, axisID, event)

    def operations(self, key, axisID, event):
        """Canvas operations

        Args:
            key (str): Which key has been pressed
            axisID (int): The index of the axis where the key has been pressed (see get_axisID)
        """
        # Check if the user really wants to quit
        if key == 'q' and self._qconf:
            if self._changes:
                self.update_infobox(message="WARNING: There are unsaved changes!!\nPress q again to exit", yesno=False)
                self._qconf = True
            else:
                plt.close()
        elif self._qconf:
            self.update_infobox(default=True)
            self._qconf = False

        # Manage responses from questions posed to the user.
        if self._respreq[0]:
            if key != "y" and key != "n":
                return
            else:
                # Switch off the required response
                self._respreq[0] = False
                # Deal with the response
                if self._respreq[1] == "write":
                    # First remove the old file, and save the new one
                    self.operations('s',-1,None)
                else:
                    return
            # Reset the info box
            self.update_infobox(default=True)
            return

        if key == '?':
            self.print_help()
        elif key in ['left', 'right', 'up', 'down']:
            self._coord = self.shift_coord(key)
            # Update the spectrum that is being plotted
            self.update_spectrum()
        elif key == 'a':
            self.SetBetterStartParams()
        elif key == 'c':
            # Toggle that the current spaxel is complete
            self.maps['complete'][self._idx, self._idy] = 1-self.maps['complete'][self._idx, self._idy]
            self.complete.set_data(self._imxarr, self._imyarr, self.maps['complete'])
            self.replot()
        elif key == 'd':
            self.FitConstant()
            self.update_spectrum()
            self.replot()
        elif key == 'f':
            # First perform the fit
            self.perform_fit()
            # Now update the spectrum being shown, and replot
            self.update_spectrum()
            self.replot()
        elif key == 'l':
            self.maps, self.maskcube = load_maps(self.map_name)
            self.replot()
        elif key == 'm':
            self._modpar[2 * self._mcycle] = event.xdata
            self._modpar[2 * self._mcycle + 1] = event.ydata
            self._mcycle += 1
            self._mcycle = self._mcycle % 3
            # Check if all mod params are set
            if self._mcycle == 0:
                self.update_fitpar()
                self.replot()
        elif key == 'n':
            self._mcycle = 0
        elif key == 'o':
            print(self.maps['params'][:, self._idx, self._idy])
        elif key == 'q':
            if self._changes:
                self.update_infobox(message="WARNING: There are unsaved changes!!\nPress q again to exit", yesno=False)
                self._qconf = True
            else:
                plt.close()
        elif key == 's':
            save_maps(self.map_name, self.maps['flux'], self.maps['errs'], self.maps['cont'], self.maps['complete'], self.maps['params'], self.maskcube)
        elif key == 'y':
            self.toggle_yscale()
            self.replot()
        elif key == 'z':
            self.maps['flux'][self._idx, self._idy] = 0
            self.maps['errs'][self._idx, self._idy] = 0
            self.maps['cont'][self._idx, self._idy] = 0
            self.maps['params'][:, self._idx, self._idy] = 0
            self.maskcube[:, self._idx, self._idy] = 0
            if self.maps['complete'][self._idx, self._idy] == 1:
                self.operations('c', -1, event) # Mark as complete, and this step also replots
        elif key == '+':
            if self._fitdict["polyorder"] < 10:
                self._fitdict["polyorder"] += 1
                self.update_infobox(message="Polynomial order = {0:d}".format(self._fitdict["polyorder"]), yesno=False)
                self.fitsol_fit()
                self.replot()
            else:
                self.update_infobox(message="Polynomial order must be <= 10", yesno=False)
        elif key == '-':
            if self._fitdict["polyorder"] > 1:
                self._fitdict["polyorder"] -= 1
                self.update_infobox(message="Polynomial order = {0:d}".format(self._fitdict["polyorder"]), yesno=False)
                self.fitsol_fit()
                self.replot()
            else:
                self.update_infobox(message="Polynomial order must be >= 1", yesno=False)
        self.canvas.draw()

    def perform_fit(self, include_ab=None):
        """Perform a fit to the current data inside the fit regions
        """
        include_em = False
        if include_ab is None:
            include_ab = self._include_ab
        self._p0c = self.maps['params'][:self._npoly, self._idx, self._idy]
        if include_ab:
            if self._p0a is None:
                self._p0a = self.maps['params'][self._npoly:, self._idx, self._idy]
        else:
            self._p0a = np.array([0.0, 0.0, 300.0, self._atomprop['wave'], self._atomprop['fval'], 0.0])
        # Perform the fit
        flxsum, errsum, contval, pars = fitting.fit_one_cont(self._atomprop, self.curr_wave, self.curr_flux, self.curr_err, self._fitregions,
                                                             contsample=100, verbose=False, p0c=self._p0c, p0a=self._p0a, p0e=self._p0e,
                                                             include_ab=self._include_ab, npoly=self._npoly, include_em=include_em)
        if flxsum is None:
            # Something failed.
            self.update_infobox(message="Fit failed...", yesno=False)
        else:
            self.update_infobox(message="Fit successful...", yesno=False)
            self.maps['flux'][self._idx, self._idy] = flxsum
            self.maps['errs'][self._idx, self._idy] = errsum
            self.maps['cont'][self._idx, self._idy] = contval
            if pars.size == self.maps['params'].shape[0]:
                self.maps['params'][:, self._idx, self._idy] = pars
            elif not include_ab:
                tmp = np.array([0.0, 0.0, 300.0, self._atomprop['wave'], self._atomprop['fval'], 0.0])
                self.maps['params'][:, self._idx, self._idy] = np.append(pars, tmp.copy())
            else:
                self.update_infobox(message="Fit Failed...", yesno=False)
                self.maps['flux'][self._idx, self._idy] = 0
                self.maps['errs'][self._idx, self._idy] = 0
                self.maps['cont'][self._idx, self._idy] = 0
        return

    def update_infobox(self, message="Press '?' to list the available options",
                       yesno=True, default=False):
        """Send a new message to the information window at the top of the canvas

        Args:
            message (str): Message to be displayed
        """

        self.axes['info'].clear()
        if default:
            self.axes['info'].text(0.5, 0.5, "Press '?' to list the available options", transform=self.axes['info'].transAxes, horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return
        # Display the message
        self.axes['info'].text(0.5, 0.5, message, transform=self.axes['info'].transAxes,
                      horizontalalignment='center', verticalalignment='center')
        if yesno:
            self.axes['info'].fill_between([0.8, 0.9], 0, 1, facecolor='green', alpha=0.5, transform=self.axes['info'].transAxes)
            self.axes['info'].fill_between([0.9, 1.0], 0, 1, facecolor='red', alpha=0.5, transform=self.axes['info'].transAxes)
            self.axes['info'].text(0.85, 0.5, "YES", transform=self.axes['info'].transAxes,
                          horizontalalignment='center', verticalalignment='center')
            self.axes['info'].text(0.95, 0.5, "NO", transform=self.axes['info'].transAxes,
                          horizontalalignment='center', verticalalignment='center')
        self.axes['info'].set_xlim((0, 1))
        self.axes['info'].set_ylim((0, 1))
        self.canvas.draw()

    def update_regions(self):
        """Update the regions used to fit Gaussian
        """
        self._fitregions[self._start:self._end] = self._addsub

    def update_fitpar(self):
        """Update the regions used to fit Gaussian
        """
        # Determine the starting parameters to use for the fit
        # Start with the continuum
        xv = np.array([self._modpar[0], self._modpar[2]])
        yv = np.array([self._modpar[1], self._modpar[3]])
        minx, maxx = np.min(xv), np.max(xv)
        xr = (xv-minx)/(maxx-minx)
        if self._npoly==0:
            p0c = np.polyfit(xr, yv, 0)
        else:
            p0c = np.polyfit(xr, yv, 0)
            if self._npoly == 1:
                self._p0c = p0c
            else:
                self._p0c = np.append(np.zeros(self._npoly-1), p0c)
        # Now set up the absorption
        zabs = (self._modpar[4]/self._atomprop['wave']) - 1
        bval = 366.0
        xr = (self._modpar[4]-minx)/(maxx-minx)
        contval = np.polyval(self._p0c, xr)
        depth = contval-self._modpar[5]
        if depth <= 0:
            logcold = 0
        else:
            wv = self._atomprop['wave'] * 1.0e-8
            bl = bval * wv / 2.99792458E5
            cns = wv * wv * self._atomprop['fval'] / (bl * 2.002134602291006E12)
            logcold = np.log10(-np.log(self._modpar[5]/contval)/cns)
        self._p0a = np.array([logcold, zabs, bval, self._atomprop['wave'], self._atomprop['fval'], 0.0])
        # No emission parameters
        self._p0e = None
        # Update the model in the plotting window
        pinit, _, index = fitting.prepare_fitting(self._atomprop, self.curr_wave, self.datacube[:, self._idx, self._idy],
                                                  p0c=self._p0c, p0a=self._p0a, p0e=self._p0e, npoly=self._npoly, include_ab=self._include_ab)
        model = fitting.full_model(pinit, self.curr_wave, index)
        self.model.set_ydata(model)
        self.resid.set_ydata((self.curr_flux-model)*inverse(self.curr_err))
        self.replot()

    def update_spectrum(self):
        """Update the regions used to fit Gaussian
        """
        self._idx = self._coord[1]#self._nx-1-self._coord[0]
        self._idy = self._coord[0]#self._ny-1-self._coord[1]
        # Update the current flux and error
        self.curr_flux = self.datacube[:, self._idx, self._idy]
        self.curr_err = self.sigcube[:, self._idx, self._idy]
        # Update the plots
        # if self.maps['complete'][self._idx, self._idy] == 0:
        #     self.SetBetterStartParams()
        self._fitregions = self.maskcube[:, self._idx, self._idy]
        _, _, index = fitting.prepare_fitting(self._atomprop, self.curr_wave, self.datacube[:, self._idx, self._idy], npoly=self._npoly, include_ab=self._include_ab)
        model = fitting.full_model(self.maps['params'][:, self._idx, self._idy], wave, index)
        self.model.set_ydata(model)
        self.resid.set_ydata((self.curr_flux-model)*inverse(self.curr_err))
        self.spec.set_ydata(self.curr_flux)
        self.speczoom.set_ydata(self.curr_flux)
        # Update the limits
        xmin, xmax = self.axes['main'].get_xlim()
        ww = np.where((self.curr_wave >= xmin) & (self.curr_wave <= xmax))
        fmax = np.max(self.curr_flux[ww])*1.1
        fmed = np.median(self.curr_flux[ww])
        fmad = 1.4826*np.median(np.abs(self.curr_flux[ww]-fmed))
        self.axes['main'].set_ylim(0, fmax)
        self.axes['fit'].set_ylim(0, min(fmed+3*fmad, fmax))
        # Update the images
        self.image.set_data(self._imxarr, self._imyarr, self.maps['flux'])
        # Update the location of the scatter points in the images
        self.pt_map.set_offsets(np.c_[self._coord[0], self._coord[1]])
        self.pt_wl.set_offsets(np.c_[self._coord[0], self._coord[1]])
        self.pt_comp.set_offsets(np.c_[self._coord[0], self._coord[1]])
        # Reset the fit parameters
        self._p0c, self._p0a, self._p0e = None, None, None
        self._mcycle = 0
        # Replot the data
        self.replot()

    def update_image_scale(self, value):
        # Update the image scales
        self.image.set_clim(vmin=0, vmax=np.max(self.maps['flux']) * 10.0**value)
        self.wlimage.set_clim(vmin=0, vmax=10.0**value)
        # Replot the data
        self.replot()

    def FitConstant(self):
        # Find the left and right regions
        ww = np.where(self.maskcube[:, self._idx, self._idy]==1)
        wi = ww[0]
        ll = np.min(self.curr_wave[ww])
        rr = np.max(self.curr_wave[ww])
        mnflx = np.median(self.curr_flux[ww])
        self._modpar[0] = ll
        self._modpar[1] = mnflx
        self._modpar[2] = rr
        self._modpar[3] = mnflx
        self._modpar[4] = 0.5*(ll+rr)
        self._modpar[5] = mnflx + 100
        self.update_fitpar()
        # Now fit it
        self.perform_fit(include_ab=False)

    def SetBetterStartParams(self):
        # Find the left and right regions
        ww = np.where(self.maskcube[:, self._idx, self._idy]==1)
        wi = ww[0]
        ll = np.min(self.curr_wave[ww])
        rr = np.max(self.curr_wave[ww])
        mnflx = np.median(self.curr_flux[ww])
        self._modpar[0] = ll
        self._modpar[1] = mnflx
        self._modpar[2] = rr
        self._modpar[3] = mnflx
        self._modpar[4] = 0.5*(ll+rr)
        self._modpar[5] = mnflx*0.8
        self.update_fitpar()
        # Update the fitted regions
        newmask = self.maskcube[:, self._idx, self._idy].copy()
        xtra = int(0.3*(wi[-1]-wi[0]))
        newmask[wi[0]-xtra:wi[0]+1] = 1
        newmask[wi[-1]:wi[-1]+xtra] = 1
        self.maskcube[:, self._idx, self._idy] = newmask.copy()
        # Now fit it
        self.perform_fit()


def make_mask(wave, datcube, waveobs, delwave):
    print("Generating mask")
    mskcube = np.zeros_like(datcube)
    ww = np.where((wave >= waveobs-delwave) & (wave <= waveobs+delwave))[0]
    extcube = datcube[ww,:,:]
    npix, nxx, nyy = extcube.shape
    idx = np.unravel_index(np.argmax(extcube), extcube.shape)
    # Calculate the mask for this set of pixels
    spec = gaussian_filter1d(datcube[ww, idx[1], idx[2]], 1)
    refidx = idx[0]
    refmsk = mask_one(spec, refidx)
    refsum = np.sum(refmsk)
    mskcube[ww, idx[1], idx[2]] = refmsk.copy()
    # Loop through all pixels and produce a mask
    for xx in range(nxx):
        for yy in range(nyy):
            spec = gaussian_filter1d(datcube[ww, xx, yy], 1)
            if np.all(spec==0.0): continue
            smx = np.argmax(spec)
            # Check if it's a confident detection
            med = np.median(spec)
            mad = 1.4826*np.median(np.abs(med-spec))
            if spec[smx] < med + 5*mad:
                mskcube[ww, idx[1], idx[2]] = refmsk.copy()
                continue
            # Generate a new mask
            thismsk = mask_one(spec, smx)
            if thismsk is None:
                mskcube[ww, xx, yy] = refmsk.copy()
            elif np.sum(thismsk) < refsum:
                # Just use the reference mask to be conservative
                mskcube[ww+smx-refidx, xx, yy] = refmsk.copy()
            else:
                mskcube[ww, xx, yy] = thismsk.copy()
    return mskcube


def mask_one(spec, idx, pad=1):
    # Starting with spec[idx], mask all pixels deemed to continue flux from an emission line
    try:
        diff = spec[1:]-spec[:-1]
        # Look for pixels to the red of an emission line
        wgd = np.where(diff>0)[0]
        wup = wgd[np.where(wgd>idx)][0]
        # Look for pixels to the blue of an emission line
        wgd = np.where(diff<0)[0]
        wlo = wgd[np.where(wgd<idx)][-1]
        msk = np.ones_like(spec)
        msk[wlo-pad:wup+pad] = 0
    except:
        msk = None
    return msk

def grow_mask(msk):
    newmsk = msk.copy()
    ww = np.where(msk==1)
    # Go left
    ii = (np.clip(ww[0]-1, 0, msk.shape[0]-1), ww[1])
    newmsk[ii] = 1
    # Go Right
    ii = (np.clip(ww[0]+1, 0, msk.shape[0]-1), ww[1])
    newmsk[ii] = 1
    # Go Up
    ii = (ww[0], np.clip(ww[1]+1, 0, msk.shape[1]-1))
    newmsk[ii] = 1
    # Go Down
    ii = (ww[0], np.clip(ww[1]-1, 0, msk.shape[1]-1))
    newmsk[ii] = 1
    # Return new mask
    return newmsk

def load_maps(mapname):
    print(f"Loading {mapname}")
    data = fits.open(mapname)
    maps = dict(flux=data[0].data, errs=data[1].data, cont=data[2].data, complete=data[3].data, params=data[4].data)
    return maps, data[5].data


def save_maps(mapname, map_flux, map_errs, map_cont, map_comp, map_params, maskcube):
    print(f"Saving {mapname}")
    pri_hdu = fits.PrimaryHDU(map_flux)
    img_hdu1 = fits.ImageHDU(map_errs)
    img_hdu2 = fits.ImageHDU(map_cont)
    img_hdu3 = fits.ImageHDU(map_comp)
    img_hdu4 = fits.ImageHDU(map_params)
    img_hdu5 = fits.ImageHDU(maskcube)
    hdu = fits.HDUList([pri_hdu, img_hdu1, img_hdu2, img_hdu3, img_hdu4, img_hdu5])
    hdu.writeto(mapname, overwrite=True)
    print("Spectra left to fit:", int(np.sum(1 - map_comp)))
    return


def get_mapname(dirc, name, line):
    mapname = dirc + "maps/" + name.replace(".fits","") + f"_{line}.fits"
    return mapname


def inverse(arr):
    return (arr!=0)/(arr + (arr==0))


if __name__ == "__main__":
    # Datacube
    dirc = "../../../IZw18_KCWI/final_cubes/"
    filename = "IZw18_BH2_newSensFunc.fits"
    #filename = "IZw18_B.fits"
    refit = False
    #line, include_ab, npoly = "HIg", True, 3
    #line, include_ab, npoly = "HId", True, 3
    line, include_ab, npoly = "HeI4026", False, 3
    zem = (717.0 / 299792.458)
    hdus = fits.open(dirc+filename)
    wcs = WCS(hdus[1].header)
    datcube = hdus[1].data
    sigcube = np.sqrt(hdus[2].data)
    wave = wcs.wcs_pix2world(0.0, 0.0, np.arange(datcube.shape[0]), 0)[2]*1.0E10
    # Open the Cube Fitter
    cubefitr = CubeFitter.initialise(datcube, sigcube, wave, line, zem, dirc, filename, include_ab=include_ab, npoly=npoly, refit=refit)
