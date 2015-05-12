import numpy as np
import matplotlib.pyplot as plt
from .plotHandler import BasePlotter
from matplotlib.patches import Ellipse

__all__ = ['NeoDetectPlotter']

class NeoDetectPlotter(BasePlotter):
    def __init__(self, step=.01):

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

        xvec = np.arange(-plotDict['xMax']-self.step,plotDict['xMax']+self.step, self.step)
        yvec = np.arange(-plotDict['yMax']-self.step,plotDict['yMax']+self.step, self.step)

        xv,yv = np.meshgrid(xvec,yvec, indexing='xy')
        image = xv*0

        R = (xv**2+(yv-1.)**2)**0.5
        # make theta span from 0 to 2 pi
        theta = np.arctan2(xv,yv-1.)


        fov = np.radians(3.5)
        for dist,x,y in zip(metricValue[0].data['NEODist'],metricValue[0].data['NEOX'],
                            metricValue[0].data['NEOY']):
            polarTheta = np.arctan2(x,y-1.)
            theta1 = polarTheta+np.radians(fov/2.)
            theta2 = polarTheta-np.radians(fov/2.)
            thetas = np.sort(np.array([theta1,theta2]))



            # XXX--need to catch case where points are in quadrant 1 and 4.
            if np.abs(theta1 - theta2) > np.pi:
                good = np.where( (theta >= thetas[0]) & (theta >= thetas[1]) & (R <= dist))
            else:
                good = np.where( (theta >= thetas[0]) & (theta <= thetas[1]) & (R <= dist))


            image[good] += 1

        #blah = ax.imshow(image, extent=[xvec.min(), xvec.max(), yvec.min(),yvec.max()])
        blah = ax.pcolormesh(image,xv,yv)
        cb = plt.colorbar(blah, ax=ax)


#        for filterName in self.filter2color:
#            good = np.where(metricValue[0].data[self.filterColName] == filterName)
#            if np.size(good[0]) > 0:
#                ax.plot( metricValue[0].data['NEOX'], metricValue[0].data['NEOY'], 'o',
#                            color=self.filter2color[filterName], alpha=0.1, markeredgecolor=None)

        ax.set_xlabel(plotDict['xlabel'])
        ax.set_ylabel(plotDict['ylabel'])
        ax.set_title(plotDict['title'])
        ax.set_ylim([plotDict['yMin'], plotDict['yMax']])
        ax.set_xlim([plotDict['xMin'], plotDict['xMax']])

        ax.plot([0],[1],marker='o', color='b')
        ax.plot([0],[0], marker='o', color='y')




        return fig.number
