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

import mpfit_single as mpfit
from scipy.special import wofz
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
                   'a' : "Automatically identify lines using current solution",
                   'c' : "Clear automatically identified lines",
                   'd' : "Delete all line identifications (start from scratch)",
                   'f' : "Fit the wavelength solution",
                   'i' : "Include an undetected line to the detected line list\n" +
                         "         First select fitting pixels (LMB drag = add, RMB drag = remove)\n" +
                         "         Then press 'i' to perform a fit." +
                         "         NOTE: ghost solution must be turned off to select fit regions.",
                   'm' : "Select a line",
                   'r' : "Refit a line",
                   'y' : "Toggle the y-axis scale between logarithmic and linear",
                   '+/-' : "Raise/Lower the order of the fitting polynomial"
                   })

# Define some axis variables
AXMAIN = 0
AXRESID = 1
AXZOOM = 2
AXINFO = 3
AXFLUXMAP = 4
AXWHITELIGHT = 5

class CubeFitter:
    """
    GUI to interactively fit emission and absorption lines in a datacube.
    """

    def __init__(self, canvas, axes, specim, wave, datacube, sigcube, idx, idy, atomprop, y_log=True):
        """Controls for the Identify task in PypeIt.

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
        self.spec = specim['spec']#.get_ydata()
        self.speczoom = specim['speczoom']#.get_ydata()
        self.model = specim['model']#.get_ydata()
        self.resid = specim['resid']#.get_ydata()
        self.image = specim['im']#.get_data()
        self.wlimage = specim['imwl']
        self.pt_map = specim['pt_map']
        self.pt_wl = specim['pt_wl']
        self.curr_wave = wave
        self.curr_flux = datacube[:, idx, idy]
        self.curr_err = sigcube[:, idx, idy]
        self.y_log = y_log
        self._atomprop = atomprop
        # datacube
        self.datacube = datacube
        self.sigcube = sigcube
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
        self._fitr = None  # Matplotlib shaded fit region (for refitting lines)
        self._fitregions = np.zeros(self.curr_wave.size, dtype=np.int)  # Mask of the pixels to be included in a fit
        self._addsub = 0   # Adding a region (1) or removing (0)
        self._msedown = False  # Is the mouse button being held down (i.e. dragged)
        self._respreq = [False, None]  # Does the user need to provide a response before any other operation will be permitted? Once the user responds, the second element of this array provides the action to be performed.
        self._qconf = False  # Confirm quit message
        self._changes = False

        # Draw the spectrum
        self.replot()

    @classmethod
    def initialise(cls, datcube, sigcube, wave, line, zem, dirc, fname, refit=False, y_log=False):
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
        delwave = 7.0

        # Make a whitelight image, and the initial
        whitelight = np.sum(datcube, axis=0)/np.sum(sigcube == 0, axis=0)
        flx_map = np.zeros_like(whitelight)
        err_map = np.zeros_like(whitelight)

        mapname = get_mapname(dirc, fname, line)
        if refit:
            mskcube = make_mask(wave, datcube, waveobs, delwave)
            for xx in range(datcube.shape[1]):
                print(f"fitting row {xx+1}/{datcube.shape[1]}")
                for yy in range(datcube.shape[2]):
                    flx, err, msk = datcube[:, xx, yy], sigcube[:, xx, yy], mskcube[:, xx, yy]
                    flxsum, errsum = fitting.fit_one_cont(atom_prop, wave, flx, err, msk, contsample=100, verbose=False)
                    flx_map[xx, yy] = flxsum
                    err_map[xx, yy] = errsum
            save_maps(mapname, flx_map, err_map)
        # Load the saved maps to check it worked, and put it in the correct format
        all_maps = load_maps(mapname)

        idx, idy = datcube.shape[1]//2, datcube.shape[2]//2
        # Create a Line2D instance for the spectrum
        spec = Line2D(wave, datcube[:, idx, idy],
                      linewidth=1, linestyle='solid', color='k',
                      drawstyle='steps-mid', animated=True)

        speczoom = Line2D(wave, datcube[:, idx, idy],
                          linewidth=1, linestyle='solid', color='k',
                          drawstyle='steps-mid', animated=True)

        specfit = Line2D(wave, np.ones(wave.size),
                         linewidth=1, linestyle='solid', color='r',
                         animated=True)

        resid = Line2D(wave, np.zeros(wave.size),
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
        im = NonUniformImage(axmap, interpolation='nearest', cmap=cm.inferno)
        im.set_data(xarr, yarr, all_maps['flux'])
        im.set_clim(vmin=0, vmax=30.0)
        im.set_extent((0, xarr.size, 0, yarr.size))
        mappt = axmap.scatter([idx], [idy], marker='x', color='b')

        # Add a whitelight image
        axwl = fig.add_axes([0.65, .1, .2, .2*16/9])
        imwl = NonUniformImage(axwl, interpolation='nearest', cmap=cm.inferno)
        imwl.set_data(xarr, yarr, whitelight)
        imwl.set_clim(vmin=0, vmax=np.max(whitelight))
        imwl.set_extent((0, xarr.size, 0, yarr.size))
        wlpt = axwl.scatter([idx], [idy], marker='x', color='b')

        # Add two residual fitting axes
        axfit = fig.add_axes([0.05, .2, .55, 0.25])
        axfit.sharex(ax)
        axres = fig.add_axes([0.05, .05, .55, 0.15])
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
        axfit.set_ylim((-0.3, 0.3))  # This will get updated as lines are identified
        axfit.set_ylabel('Flux')

        # Add an information GUI axis
        axinfo = fig.add_axes([0.15, .92, .7, 0.07])
        axinfo.get_xaxis().set_visible(False)
        axinfo.get_yaxis().set_visible(False)
        axinfo.text(0.5, 0.5, "Press '?' to list the available options", transform=axinfo.transAxes,
                    horizontalalignment='center', verticalalignment='center')
        axinfo.set_xlim((0, 1))
        axinfo.set_ylim((0, 1))

        axes = dict(main=ax, fit=axfit, resid=axres, info=axinfo, fmap=axmap, fwl=axwl)
        specim = dict(im=im, imwl=imwl, spec=spec, speczoom=speczoom, model=specfit, resid=resid, pt_map=mappt, pt_wl=wlpt)
        # Initialise the identify window and display to screen
        fig.canvas.set_window_title('CubeFitter')
        fitter = CubeFitter(fig.canvas, axes, specim, wave, datcube, sigcube, idx, idy, atom_prop, y_log=y_log)

        plt.show()

        # Now return the results
        return fitter

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
        self.draw_fitregions(trans)
        self.axes['main'].draw_artist(self.spec)
        self.axes['fit'].draw_artist(self.speczoom)
        self.axes['resid'].draw_artist(self.resid)
        self.axes['fmap'].draw_artist(self.image)
        self.axes['fmap'].draw_artist(self.pt_map)
        self.axes['fwl'].draw_artist(self.wlimage)
        self.axes['fwl'].draw_artist(self.pt_wl)

    def draw_fitregions(self, trans):
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
                                                    alpha=0.5, transform=trans)

    def get_ind_under_point(self, event):
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
        return [np.clip(newx,0,self._nx), np.clip(newy,0,self._ny)]

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
        self._start = self.get_ind_under_point(event)
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
            self.operations(answer, -1)
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
                self._end = self.get_ind_under_point(event)
                if self._end == self._start:
                    # The mouse button was pressed (not dragged)
                    self.operations('m', axisID, event)
                elif self._end != self._start:
                    # The mouse button was dragged
                    if axisID == 0:
                        if self._start > self._end:
                            tmp = self._start
                            self._start = self._end
                            self._end = tmp
                        self.update_regions()
            elif axisID in [AXFLUXMAP, AXWHITELIGHT]:
                self._coord = self.get_coord_under_point(event)
                # Update the spectrum that is being plotted
                self.update_spectrum()

        # Now plot
        trans = mtransforms.blended_transform_factory(self.axes['main'].transData, self.axes['main'].transAxes)
        self.canvas.restore_region(self.background)
        self.draw_fitregions(trans)
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
                msgs.bug("Need to change this to kill and return the results to PypeIt")
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
                    msgs.work("Not implemented yet!")
                    self.write()
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
        elif key == 'c':
            wclr = np.where((self._lineflg == 2) | (self._lineflg == 3))
            self._lineflg[wclr] = 0
            self.replot()
        elif key == 'd':
            self._lineflg *= 0
            self._lineids *= 0.0
            self._fitdict['coeff'] = None
            self.replot()
        elif key == 'f':
            self.perform_fit()
            self.replot()
        elif key == 'm':
            self._end = self.get_ind_under_point(event)
            self.replot()
        elif key == 'q':
            if self._changes:
                self.update_infobox(message="WARNING: There are unsaved changes!!\nPress q again to exit", yesno=False)
                self._qconf = True
            else:
                plt.close()
        elif key == 'y':
            self.toggle_yscale()
            self.replot()
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

    # TODO
    def perform_fit(self):
        """Perform a fit to the current data inside the fit regions
        """
        # Define the minimum distance between lines (in pixels)
        mindist = 4
        # Get the selected regions
        ww = np.where(self._fitregions == 1)
        xfit = self.curr_wave.copy()[ww]
        yfit = self.specdata.copy()[ww]
        from scipy.optimize import curve_fit
        # Make sure there are enough pixels for the fit
        npix = len(xfit)
        if npix <= 3:
            return
        # Some starting parameters
        ampl = np.max(yfit)
        mean = sum(xfit * yfit) / sum(yfit)
        sigma = sum(yfit * (xfit - mean) ** 2) / sum(yfit)
        # Perform a gaussian fit
        gaus = lambda x, a, x0, sigma: a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        popt, pcov = curve_fit(gaus, xfit, yfit, p0=[ampl, mean, sigma])
        print(ampl, mean, sigma, popt[1])
        # Get the new detection
        new_detn = popt[1]
        # Check that the detection doesn't already exist
        cls_line = np.min(np.abs(self._detns - new_detn))
        if cls_line > mindist:
            detns = np.append(self._detns, new_detn)
            arsrt = np.argsort(detns)
            self._detns = detns[arsrt]
            self._detnsy = self.get_ann_ypos()  # Get the y locations of the annotations
            self._lineids = np.append(self._lineids, 0)[arsrt]
            self._lineflg = np.append(self._lineflg, 0)[arsrt]  # Flags: 0=no ID, 1=user ID, 2=auto ID, 3=flag reject
        else:
            self.update_infobox("New detection is <{0:d} pixels of a detection - ignoring".format(mindist))
        # Reset the fit regions
        self._fitregions = np.zeros_like(self._fitregions)
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

    def update_spectrum(self):
        """Update the regions used to fit Gaussian
        """
        self._idx = self._nx-1-self._coord[0]
        self._idy = self._ny-1-self._coord[1]
        # Update the current flux and error
        self.curr_flux = self.datacube[:, self._idx, self._idy]
        self.curr_err = self.sigcube[:, self._idx, self._idy]
        # Update the plots
        model = np.zeros(self.curr_wave.size)
        self.model.set_ydata(model)
        self.resid.set_ydata(model)
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
        # Update the location of the scatter points in the images
        self.pt_map.set_offsets(np.c_[self._coord[0], self._coord[1]])
        self.pt_wl.set_offsets(np.c_[self._coord[0], self._coord[1]])
        # Replot the data
        self.replot()


# TODO
def calc_total_flux(atom_prop, wave, flx, err, msk, contsample=100):
    """
    Perform a fit to the continuum, and then calculate the total emission
    line flux, including continuum uncertainties.
    """

    return

def make_mask(wave, datcube, waveobs, delwave):
    print("Generating mask")
    mskcube = np.zeros_like(datcube)
    ww = np.where((wave >= waveobs-delwave) & (wave <= waveobs+delwave))[0]
    extcube = datcube[ww,:,:]
    npix, nxx, nyy = extcube.shape
    idx = np.unravel_index(np.argmax(extcube), extcube.shape)
    # Calculate the mask for this set of pixels
    spec = datcube[ww, idx[1], idx[2]]
    refidx = idx[0]
    refmsk = mask_one(spec, refidx)
    refsum = np.sum(refmsk)
    mskcube[ww, idx[1], idx[2]] = refmsk.copy()
    # Loop through all pixels and produce a mask
    embed()
    for xx in range(nxx):
        for yy in range(nyy):
            spec = datcube[ww, xx, yy]
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
                mskcube[ww, idx[1], idx[2]] = refmsk.copy()
            elif np.sum(thismsk) < refsum:
                # Just use the reference mask to be conservative
                mskcube[ww+smx-refidx, idx[1], idx[2]] = refmsk.copy()
            else:
                mskcube[ww, idx[1], idx[2]] = thismsk.copy()
    return mskcube


def mask_one(spec, idx, pad=5):
    # Starting with spec[idx], mask all pixels deemed to continue flux from an emission line
    try:
        diff = spec[1:]-spec[:-1]
        # Look for pixels to the red of an emission line
        wgd = np.where(diff>0)[0]
        wup = wgd[np.where(wgd>idx)][0]
        # Look for pixels to the blue of an emission line
        wgd = np.where(diff<0)[0]
        wlo = wgd[np.where(wgd<idx)][-1]
        msk = np.zeros_like(spec)
        msk[wlo-pad:wup+pad] = 1
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
    data = np.load(mapname)
    maps = dict(flux=data[:,:,0], errs=data[:,:,1])
    return maps


def save_maps(mapname, map_flux, map_errs):
    print(f"Saving {mapname}")
    data = np.transpose((map_flux.T, map_errs.T))
    np.save(mapname, data)
    return


def get_mapname(dirc, name, line):
    mapname = dirc + "maps/" + name.replace(".fits","") + f"_{line}.npy"
    return mapname


if __name__ == "__main__":
    # Datacube
    dirc = "../../../IZw18_KCWI/final_cubes/"
    filename = "IZw18_BH2_newSensFunc.fits"
    refit = True
    line = "HIg"
    zem = (751.0 / 299792.458)
    hdus = fits.open(dirc+filename)
    wcs = WCS(hdus[1].header)
    datcube = hdus[1].data
    sigcube = np.sqrt(hdus[2].data)
    wave = wcs.wcs_pix2world(0.0, 0.0, np.arange(datcube.shape[0]), 0)[2]*1.0E10
    # Open the Cube Fitter
    cubefitr = CubeFitter.initialise(datcube, sigcube, wave, line, zem, dirc, filename, refit=refit)
