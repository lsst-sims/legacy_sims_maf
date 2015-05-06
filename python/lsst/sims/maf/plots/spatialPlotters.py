import numpy as np
import warnings
import healpy as hp
from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter

from lsst.sims.maf.utils import optimalBins, percentileClipping

__all__ = ['BasePlotter', 'HealpixSkyMap', 'HealpixPowerSpectrum', 'HealpixHistogram', 'BaseHistogram']

class BasePlotter(object):
    """
    Serve as the base type for MAF plotters and example of API.
    """
    def __init__(self):
        self.plotType = None
        self.defaultPlotDict = None
    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        pass

class HealpixSkyMap(BasePlotter):
    def __init__(self):
        # Set the plotType
        self.plotType = 'SkyMap'
        # Set up the default plotting parameters.
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'label':None,
                                'logScale':False, 'cbarFormat':'%.2f', 'cmap':cm.jet,
                                'percentileClip':None, 'colorMin':None, 'colorMax':None,
                                'zp':None, 'normVal':None,
                                'cbar_edge':True, 'nTicks':None, 'rot1':0, 'rot2':0, 'rot3':0}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Generate a sky map of healpix metric values using healpy's mollweide view.
        """
        if slicer.slicerName != 'HealpixSlicer':
            raise ValueError('HealpixSkyMap is for use with healpix slicers')
        fig = plt.figure(fignum)
        # Override the default plotting parameters with user specified values.
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        # Generate a Mollweide full-sky plot.
        norm = None
        if plotDict['logScale']:
            norm = 'log'
        cmap = plotDict['cmap']
        if type(cmap) == str:
            cmap = getattr(cm, cmap)
        # Make colormap compatible with healpy
        cmap = colors.LinearSegmentedColormap('cmap', cmap._segmentdata, cmap.N)
        cmap.set_over(cmap(1.0))
        cmap.set_under('w')
        cmap.set_bad('gray')
        if plotDict['zp']:
            metricValue = metricValueIn - plotDict['zp']
        elif plotDict['normVal']:
            metricValue = metricValueIn/plotDict['normVal']
        else:
            metricValue = metricValueIn
        # Set up color bar limits.
        if plotDict['percentileClip']:
            pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=plotDict['percentileClip'])
        colorMin = plotDict['colorMin']
        colorMax = plotDict['colorMax']
        if colorMin is None and plotDict['percentileClip']:
            colorMin = pcMin
        if colorMax is None and plotDict['percentileClip']:
            colorMax = pcMax
        if (colorMin is not None) or (colorMax is not None):
            clims = [colorMin, colorMax]
        else:
            clims = None
        # Make sure there is some range on the colorbar
        if clims is None:
            if metricValue.compressed().size > 0:
                clims=[metricValue.compressed().min(), metricValue.compressed().max()]
            else:
                clims = [-1, 1]
            if clims[0] == clims[1]:
                clims[0] =  clims[0] - 1
                clims[1] =  clims[1] + 1
        rot = (plotDict['rot1'], plotDict['rot2'], plotDict['rot3'])
        hp.mollview(metricValue.filled(slicer.badval), title=plotDict['title'], cbar=False,
                    min=clims[0], max=clims[1], rot=rot, flip='astro',
                    cmap=cmap, norm=norm)
        # This graticule call can fail with old versions of healpy and matplotlib 1.4.0.
        # Make sure the latest version of healpy in the stack is setup
        hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because we want more tickmarks).
        ax = plt.gca()
        im = ax.get_images()[0]
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.8, 0.8, '%s' %(plotDict['label']))
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = plt.colorbar(im, shrink=0.75, aspect=25, orientation='horizontal',
                            extend='both', extendrect=True, format=plotDict['cbarFormat'])
            cb.set_label(plotDict['xlabel'])
            if plotDict['nTicks'] is not None:
                tick_locator = ticker.MaxNLocator(nbins=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        return fig.number

class HealpixPowerSpectrum(BasePlotter):
    def __init__(self):
        self.plotType = 'PowerSpectrum'
        self.defaultPlotDict = {'title':None, 'label':None,
                                'maxl':500., 'removeDipole':True,
                                'logScale':True}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Generate and plot the power spectrum of metricValue (calculated on a healpix grid).
        """
        if slicer.slicerName != 'HealpixSlicer':
            raise ValueError('HealpixPowerSpectrum for use with healpix metricBundles.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        fig = plt.figure(fignum)
        # If the mask is True everywhere (no data), just plot zeros
        if False not in metricValue.mask:
            return None
        if plotDict['removeDipole']:
            cl = hp.anafast(hp.remove_dipole(metricValue.filled(slicer.badval)))
        else:
            cl = hp.anafast(metricValue.filled(slicer.badval))
        l = np.arange(np.size(cl))
        # Plot the results.
        if plotDict['removeDipole']:
            condition = ((l < plotDict['maxl']) & (l > 1))
        else:
            condition = (l < plotDict['maxl'])
        plt.plot(l[condition], (cl[condition]*l[condition]*(l[condition]+1))/2.0/np.pi, label=plotDict['label'])
        if cl[condition].max() > 0 and plotDict['logScale']:
            plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$')
        if plotDict['title'] is not None:
            plt.title(plotDict['title'])
        # Return figure number (so we can reuse/add onto/save this figure if desired).
        return fig.number

class HealpixHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.defaultPlotDict = {'title':None, 'xlabel':None,
                                'ylabel':'Area (1000s of square degrees)', 'label':None,
                                'bins':None, 'binsize':None, 'cumulative':False,
                                'scale':None, 'xMin':None, 'xMax':None,
                                'logScale':False, 'color':'b'}
        self.baseHist = BaseHistogram()
    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Histogram metricValue for all healpix points.
        """
        if slicer.slicerName != 'HealpixSlicer':
            raise ValueError('HealpixHistogram is for use with healpix slicer.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if plotDict['scale'] is None:
            plotDict['scale'] = (hp.nside2pixarea(slicer.nside, degrees=True)  / 1000.0)
        fignum = self.baseHist(metricValue, slicer, plotDict, fignum=fignum)
        return fignum


class BaseHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'ylabel':None, 'label':None,
                                'bins':None, 'binsize':None, 'cumulative':False,
                                'scsale':1.0, 'xMin':None, 'xMax':None,
                                'logScale':'auto', 'color':'b',
                                'yaxisformat':'%.3f',
                                'zp':None, 'normVal':None, 'percentileClip':None}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Plot a histogram of metricValues (such as would come from a spatial slicer).
        """
        plotType = 'Histogram'
        # Adjust metric values by zeropoint or normVal, and use 'compressed' version of masked array.
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if plotDict['zp']:
            metricValue = metricValueIn.compressed() - zp
        elif plotDict['normVal']:
            metricValue = metricValueIn.compressed()/normVal
        else:
            metricValue = metricValueIn.compressed()
        # Determine percentile clipped X range, if set. (and xmin/max not set).
        if plotDict['xMin'] is None and plotDict['xMax'] is None:
            if plotDict['percentileClip']:
                plotDict['xMin'], plotDict['xMax'] = percentileClipping(metricValue, percentile=plotDict['percentileClip'])
        # Determine range for histogram. Note that if xmin/max are None, this will just be [None, None].
        histRange = [plotDict['xMin'], plotDict['xMax']]
        # Should we use log scale on y axis? (if 'auto')
        if plotDict['logScale'] == 'auto':
            plotDict['logScale'] = False
            if np.min(histRange) > 0:
                if (np.log10(np.max(histRange)-np.log10(np.min(histRange))) > 3 ):
                    plotDict['logScale'] = True
        # If binsize was specified, set up an array of bins for the histogram.
        if plotDict['binsize'] is not None:
            #  If generating cumulative histogram, want to use full range of data (but with given binsize).
            #    .. but if user set histRange to be wider than full range of data, then
            #       extend bins to cover this range, so we can make prettier plots.
            if plotDict['cumulative'] is not False:
                if histRange[0] is not None:
                    bmin = np.min([metricValue.min(), histRange[0]])
                else:
                    bmin = metricValue.min()
                if histRange[1] is not None:
                    bmax = np.max([metricValue.max(), histRange[1]])
                else:
                    bmax = metricValue.max()
                bins = np.arange(bmin, bmax+plotDict['binsize']/2.0, plotDict['binsize'])
            #  Else try to set up bins using min/max values if specified, or full data range.
            else:
                if histRange[0] is not None:
                    bmin = histRange[0]
                else:
                    bmin = metricValue.min()
                if histRange[1] is not None:
                    bmax = histRange[1]
                else:
                    bmax = metricValue.max()
                bins = np.arange(bmin, bmax+plotDict['binsize']/2.0, plotDict['binsize'])
        # Otherwise, determine number of bins, if neither 'bins' or 'binsize' were specified.
        else:
            if plotDict['bins'] is None:
                bins = optimalBins(metricValue)
            else:
                bins = plotDict['bins']
        # Generate plots.
        fig = plt.figure(fignum)
        if plotDict['cumulative'] is not False:
            # If cumulative is set, generate histogram without using histRange (to use full range of data).
            n, b, p = plt.hist(metricValue, bins=bins, histtype='step', log=plotDict['logScale'],
                                cumulative=plotDict['cumulative'], label=plotDict['label'], color=plotDict['color'])
        else:
            # Plot non-cumulative histogram.
            # First, test if data falls within histRange, because otherwise histogram generation will fail.
            if np.min(histRange) is not None:
                if (histRange[0] is None) and (histRange[1] is not None):
                    condition = (metricValue <= histRange[1])
                elif (histRange[1] is None) and (histRange[0] is not None):
                    condition = (metricValue >= histRange[0])
                else:
                    condition = ((metricValue >= histRange[0]) & (metricValue <= histRange[1]))
                plotValue = metricValue[condition]
            else:
                plotValue = metricValue
            # If there is only one value to histogram, need to set histRange, otherwise histogram will fail.
            rangePad = 20.
            if (np.unique(plotValue).size == 1) & (np.min(histRange) is None):
                warnings.warn('Only one metric value, making a guess at a good histogram range.')
                histRange = [plotValue.min()-rangePad, plotValue.max()+rangePad]
                if (plotValue.min() >= 0) & (histRange[0] < 0):
                    # Reset histogram range if it went below 0.
                    histRange[0] = 0.
                if 'binsize' in plotDict:
                    bins = np.arange(histRange[0], histRange[1], plotDict['binsize'])
                else:
                    bins = np.arange(histRange[0], histRange[1], (histRange[1] - histRange[0])/50.)
            # If there is no data within the histogram range, we will generate an empty plot.
            # If there is data, make the histogram.
            if plotValue.size > 0:
                # Generate histogram.
                if np.min(histRange) is None:
                    histRange = None
                n, b, p = plt.hist(plotValue, bins=bins, histtype='step', log=plotDict['logScale'],
                                    cumulative=plotDict['cumulative'], range=histRange,
                                    label=plotDict['label'], color=plotDict['color'])

        # Fill in axes labels and limits.
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(y,  pos):
            return plotDict['yaxisformat'] % (y * plotDict['scale'])
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        # Set y limits.
        if 'yMin' in plotDict:
            plt.ylim(ymin=plotDict['yMin'])
        else:
            # There is a bug in histype='step' that can screw up the ylim.  Comes up when running allSlicer.Cfg.py
            try:
                if plt.axis()[2] == max(n):
                    plt.ylim([n.min(),n.max()])
            except UnboundLocalError:
                # This happens if we were generating an empty plot (no histogram).
                # But in which case, the above error was probably not relevant. So skip it.
                pass
        if 'yMax' in plotDict:
            plt.ylim(ymax=plotDict['yMax'])
        # Set x limits.
        if plotDict['xMin'] is not None:
            plt.xlim(xmin=plotDict['xMin'])
        if plotDict['xMax'] is not None:
            plt.xlim(xmax=plotDict['xMax'])
        # Set/Add various labels.
        if plotDict['xlabel'] is not None:
            plt.xlabel(plotDict['xlabel'])
        elif 'units' in plotDict:
            plt.xlabel(plotDict['units'])
        if plotDict['ylabel'] is not None:
            plt.ylabel(plotDict['ylabel'])
        if plotDict['title'] is not None:
            plt.title(plotDict['title'])
        # Return figure number
        return fig.number


class HealpixSDSSSkyMap(BasePlotter):
    def __init__(self):
        self.plotType = 'SkyMap'
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'logScale':False,
                                'cbarFormat':'%.2f', 'cmap':cm.jet,
                                'percentileClip':None, 'colorMin':None,
                                'colorMax':None, 'zp':None, 'normVal':None,
                                'cbar_edge':True, 'label':None}
    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None, raMin=-90,
                 raMax=90, raLen=45, decMin=-2., decMax=2.):
        """
        Plot the sky map of metricValue using healpy cartview plots in thin strips.
        raMin: Minimum RA to plot (deg)
        raMax: Max RA to plot (deg).  Note raMin/raMax define the centers that will be plotted.
        raLen:  Length of the plotted strips in degrees
        decMin: minimum dec value to plot
        decMax: max dec value to plot
        metricValueIn: metric values
        """

        fig = plt.figue(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        norm = None
        if self.plotDict['logScale']:
            norm = 'log'
        if self.plotDict['cmap'] is None:
            cmap = cm.jet
        if type(cmap) == str:
            cmap = getattr(cm,cmap)
        # Make colormap compatible with healpy
        cmap = colors.LinearSegmentedColormap('cmap', cmap._segmentdata, cmap.N)
        cmap.set_over(cmap(1.0))
        cmap.set_under('w')
        cmap.set_bad('gray')
        if self.plotDict['zp']:
            metricValue = metricValueIn - self.plotDict['zp']
        elif self.plotDict['normVal']:
            metricValue = metricValueIn/self.plotDict['normVal']
        else:
            metricValue = metricValueIn

        if self.plotDict['percentileClip']:
            pcMin, pcMax = percentileClipping(metricValue.compressed(),
                                              percentile=self.plotDict['percentileClip'])
        if self.plotDict['colorMin'] is None and self.plotDict['percentileClip']:
            self.plotDict['colorMin'] = pcMin
        if self.plotDict['colorMax'] is None and self.plotDict['percentileClip']:
            self.plotDict['colorMax'] = pcMax
        if (self.plotDict['colorMin'] is not None) or (self.plotDict['colorMax'] is not None):
            clims = [self.plotDict['colorMin'], self.plotDict['colorMax']]
        else:
            clims = None

        # Make sure there is some range on the colorbar
        if clims is None:
            if metricValue.compressed().size > 0:
                clims=[metricValue.compressed().min(), metricValue.compressed().max()]
            else:
                clims = [-1,1]
            if clims[0] == clims[1]:
                clims[0] =  clims[0]-1
                clims[1] =  clims[1]+1
        racenters=np.arange(raMin,raMax,raLen)
        nframes = racenters.size
        for i, racenter in enumerate(racenters):
            if i == 0:
                useTitle = title +' /n'+'%i < RA < %i'%(racenter-raLen, racenter+raLen)
            else:
                useTitle = '%i < RA < %i'%(racenter-raLen, racenter+raLen)
            hp.cartview(metricValue.filled(self.badval), title=useTitle, cbar=False,
                        min=clims[0], max=clims[1], flip='astro', rot=(racenter,0,0),
                        cmap=cmap, norm=norm, lonra=[-raLen,raLen],
                        latra=[decMin,decMax], sub=(nframes+1,1,i+1), fig=fig)
            hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        fig = plt.gcf()
        ax1 = fig.add_axes([0.1, .15,.8,.075]) #left, bottom, width, height
        # Add label.
        if label is not None:
            plt.figtext(0.8, 0.9, '%s' %label)
        # Make the colorbar as a seperate figure,
        # from http://matplotlib.org/examples/api/colorbar_only.html
        cnorm = colors.Normalize(vmin=clims[0], vmax=clims[1])
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=cnorm,
                                            orientation='horizontal', format=self.plotDict['cbarFormat'])
            cb1.set_label(self.plotDict['xlabel'])
        # If outputing to PDF, this fixes the colorbar white stripes
        if self.plotDict['cbar_edge']:
            cb1.solids.set_edgecolor("face")
        fig = plt.gcf()
        return fig.number
