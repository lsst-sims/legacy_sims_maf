from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .plotHandler import BasePlotter
from .perceptual_rainbow import makePRCmap

__all__=['TwoDMap', 'VisitPairsHist']


perceptual_rainbow = makePRCmap()

class TwoDMap(BasePlotter):
    def __init__(self):
        self.plotType = 'TwoD'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'ylabel':None, 'label':None,
                                'logScale':False, 'cbarFormat':None, 'cbarTitle': 'Count' ,
                                'cmap':perceptual_rainbow,
                                'percentileClip':None, 'colorMin':None, 'colorMax':None,
                                'zp':None, 'normVal':None,
                                'cbar_edge':True, 'nTicks':None, 'aspect':'auto'}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):

        if 'Healpix' in slicer.slicerName:
            self.defaultPlotDict['ylabel'] = 'Healpix ID'
        elif 'Opsim' in slicer.slicerName:
            self.defaultPlotDict['ylabel'] = 'Field ID'

        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        # Don't clobber with None
        for key in userPlotDict.keys():
            if userPlotDict[key] is not None:
                plotDict[key] = userPlotDict[key]

        # Decide if we want all the zeropoint sillyness

        metricValue = metricValueIn.copy()



        figure = plt.figure(fignum)
        ax = figure.add_subplot(111)

        image = ax.imshow(metricValue, vmin=plotDict['colorMin'], vmax=plotDict['colorMax'],
                          aspect=plotDict['aspect'], cmap=plotDict['cmap'])
        cb =  plt.colorbar(image)

        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])
        cb.set_label(plotDict['cbarTitle'])

        # Fix white space on pdf's
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        return figure.number

class VisitPairsHist(BasePlotter):
    """
    Given an opsim2dSlicer, figure out what fraction of observations are in singles, pairs, tripples, etc.
    """
    def __init__(self):
        self.plotType = 'TwoD'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'label':None,
                                'logScale':False, 'cbarFormat':None, 'cmap':perceptual_rainbow,
                                'percentileClip':None, 'colorMin':None, 'colorMax':None,
                                'zp':None, 'normVal':None,
                                'cbar_edge':True, 'nTicks':None, 'rot':(0,0,0)}

    def call(self, metricValueIn, slicer, userPlotDict, fignum=None):

        maxVal = metricValueIn.max()
        bins = np.arange(0.5,maxVal+0.5,1)

        vals, bins = np.histogram(metricValueIn, bins)
        xvals = (bins[:-1]+bins[1:])/2.

        figure = plt.figure(fignum)
        ax = figure.add_subplot(111)
        ax.plot(xvals, vals*xvals, color=plotDict['color'] )

        return figure.number
