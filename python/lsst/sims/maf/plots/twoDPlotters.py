from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .plotHandler import BasePlotter

__all__=['TwoDMap', 'VisitPairsHist']

class TwoDMap(BasePlotter):
    def __init__(self):
        self.plotType = 'TwoD'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'ylabel':None, 'label':None,
                                'logScale':False, 'cbarFormat':None, 'cbarTitle': None ,
                                'cmap':cm.jet,
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
        plotDict.update(userPlotDict)

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
    def __init__(self):
        self.plotType = 'TwoD'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'label':None,
                                'logScale':False, 'cbarFormat':None, 'cmap':cm.cubehelix,
                                'percentileClip':None, 'colorMin':None, 'colorMax':None,
                                'zp':None, 'normVal':None,
                                'cbar_edge':True, 'nTicks':None, 'rot':(0,0,0)}

    def call(self, metricValueIn, slicer, userPlotDict, fignum=None):
        pass
