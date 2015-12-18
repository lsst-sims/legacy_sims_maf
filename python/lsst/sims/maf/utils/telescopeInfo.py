__all__ = ['TelescopeInfo']

class TelescopeInfo(object):
    """
    Holds information on the location of the telescope.
    """
    def __init__(self, name):
        """
        Set the latitude, longitude and elevation of the telescope site.

        Parameters
        ---------
        name : str
            The name of the telescope. If "LSST", uses default lat/long/elevation values.
        """
        if name == 'LSST':
            self.lat = -0.527868529 #radians of '-30:14:40.7'
            self.lon =-1.2348102646986 #radians of '-70:44:57.9'
            self.elev = 2662.75 #meters
        else:
            self.lat = None
            self.lon = None
            self.elev = None
