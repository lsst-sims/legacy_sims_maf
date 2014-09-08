import numpy as np

def stellarMags(stellarType, rmag=19.):
    """
    Given a spectral type (O,B,A,F,G,K,M), return a
    dict with expected mags in LSST filters, normalized to the r-band.

    Based on mapping of Kuruz models to specrtal types here:
    http://www.stsci.edu/hst/observatory/crds/k93models.html
    """

    # If this is the first time running the function, set up the data array
    if not hasattr(stellarMags, 'data'):
        names= ['stellarType', 'Kurucz Model', 'u-g', 'g-r', 'r-i','i-z','z-y']
        types = ['|S1', '|S20', float,  float,  float,  float,  float]
        stellarMags.data = np.core.records.fromrecords([
                ['O','kp00_50000[g50]',-0.4835688497,-0.5201721327,-0.3991733698,-0.3106800468,-0.2072290744],
                ['B','kp00_30000[g40]',-0.3457202828,-0.4834762052,-0.3812792176,-0.2906072887,-0.1927230035],
                ['A','kp00_9500[g40]',0.8823182684,-0.237288029,-0.2280783991,-0.1587960264,-0.03043824335],
                ['F','kp00_7250[g45]',0.9140316091,0.1254277486,-0.03419150003,-0.0802010739,-0.03802756413],
                ['G','kp00_6000[g45]',1.198219095,0.3915608688,0.09129426676,0.002604263747,-0.004659443668],
                ['K','kp00_5250[g45]',1.716635024,0.6081567546,0.1796910856,0.06492278686,0.0425155827],
                ['M','kp00_3750[g45]',2.747842719,1.287599638,0.5375622482,0.4313486709,0.219308065]],
                                            dtype=zip(names,types))
        
    results = {}
    good = np.where(stellarMags.data['stellarType'] == stellarType)
    if len(good[0]) == 0:
        raise ValueError('stellarType must be one of O,B,A,F,G,K,M')
    results['r'] = rmag
    results['i'] = rmag-stellarMags.data[good]['r-i']
    results['z'] = results['i']-stellarMags.data[good]['i-z']
    results['y'] = results['z']-stellarMags.data[good]['z-y']
    results['g'] = stellarMags.data[good]['g-r']+results['r']
    results['u'] = stellarMags.data[good]['u-g']+results['g']

    return results

