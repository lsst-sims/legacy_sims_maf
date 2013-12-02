import numpy.lib.recfunctions as rfn

def opsimStack(arrays):
    """Easy way to add columns to a numpy recarray.  Takes a list of numpy rec arrays and merges them """
    result = rfn.merge_arrays(arrays, flatten=True, usemask=False)
    return result
