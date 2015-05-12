import numpy as np
import matplotlib.pyplot as plt
from .plotHandler import BasePlotter
from matplotlib.patches import Ellipse

__all__ = ['NeoDetectPlotter']

class NeoDetectPlotter(BasePlotter):
    def __init__(self, step=.001):

        self.plotType = 'neoxyPlotter'
        self.objectPlotter = True
        self.defaultPlotDict = {'title':None, 'xlabel':'X (AU)',
                                'ylabel':'Y (AU)', 'xMin':-1.5, 'xMax':1.5,
                                'yMin':-.25, 'yMax':2.5}
        self.filter2color={'u':'purple','g':'blue','r':'green',
                           'i':'cyan','z':'orange','y':'red'}
        self.filterColName = 'filter'
        self.step = step

    def __call__(self, metricValue, slicer,userPlotDict, fignum=None):

        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)


        planetProps = {'Earth': 1., 'Venus':0.72, 'Mars':1.52, 'Mercury':0.39}

        planets = []
        for prop in planetProps:
            planets.append(Ellipse((0,0), planetProps[prop]*2, planetProps[prop]*2, fill=False ))

        for planet in planets:
            ax.add_artist(planet)

        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

 #       xvec = np.arange(-plotDict['xMax']-self.step,plotDict['xMax']+self.step, self.step)
 #       y = np.arange(-plotDict['yMax']-self.step,plotDict['yMax']+self.step, self.step)

 #       xv,yv = np.meshgrid(x,y, indexing='xy')
 #       image = xv*0



#        fov = np.radians(3.5.)
#        for x,y in zip(metricValue[0].data['NEOX'],metricValue[0].data['NEOY']):
#
#            if x <0:
#                xind = np.where( (xvec > x) & (xvec < 0) )
#            else:
#                xind = np.where( (xvec < x) & (xvec > 0) )
#
#            theta = np.arctan(y/x)
#            theta1 = theta-fov/2.
#            y1 =  ( (x**2.+y**2)/(1.+1./(np.tan(theta1)**2)) )**0.5
#            x1 = ( ( x**2+y**2)/y1 )**0.5
#
#            y1line = y1/x1*xvec
#
#            theta2 = theta+fov/2.
#            y2 = ( (x**2.+y**2)/(1.+1./(np.tan(theta1)**2)) )**0.5
#            x2 = ( ( x**2+y**2)/y2 )**0.5
#
#            y2line = y2/x2*xvec




        for filterName in self.filter2color:
            good = np.where(metricValue[0].data[self.filterColName] == filterName)
            if np.size(good[0]) > 0:
                ax.plot( metricValue[0].data['NEOX'], metricValue[0].data['NEOY'], 'o',
                            color=self.filter2color[filterName], alpha=0.1, markeredgecolor=None)

        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])
        ax.set_ylim([plotDict['yMin'], plotDict['yMax']])
        ax.set_xlim([plotDict['xMin'], plotDict['xMax']])

        ax.plot([0],[1],marker='o', color='b')
        ax.plot([0],[0], marker='o', color='y')




        return fig.number
