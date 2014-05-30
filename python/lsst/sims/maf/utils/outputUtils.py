import os, sys
import warnings
import numpy as np

def _myformat(args, delimiter=' '):
    """
    Generic line formatter (to let you specify delimiter between text fields).
    """    
    writestring = ''
    for a in args:
        if isinstance(a, list):
            if len(a) > 1:
                ap = ','.join(map(str, a))
            else:
                ap = ''.join(map(str, a))
            writestring += '%s%s' %(ap, delimiter)
        else:
            writestring += '%s%s' %(a, delimiter)
    return writestring

def printDict(content, label, filehandle=None, delimiter=' ',  _level=0):
    """
    Print dictionaries (and/or nested dictionaries) nicely.
    Can also print other simpler items (such as numpy ndarray) nicely too.
    
    filehandle = the file object for output .. if 'None' (default) prints to standard out.
    delimiter = the user specified delimiter between fields.
    _level is for internal use (controls level of indent). 
    """
    # Get set up with basic file output information.
    if filehandle is None:
        filehandle = sys.stdout
    # And set character to use to indent sets of parameters related to a single dictionary.
    baseindent = '%s' %(delimiter)
    indent = ''
    for i in range(level-1):
        indent += '%s' %(baseindent)    
    # Print data (this is also the termination of the recursion if given nested dictionaries).
    if not isinstance(content, dict):
        if isinstance(content, str) or isinstance(content, float) or isinstance(content, int):
            print >> filehandle, '%s%s%s%s' %(indent, label, delimiter, str(content))
        else:
            if isinstance(content, np.ndarray):
                if content.dtype.names is not None:
                    print >> filehandle, '%s%s%s' %(indent, delimiter, label)
                    for element in content:
                        print >> filehandle, '%s%s%s%s%s' %(indent, delimiter, indent, delimiter, _myformat(element))
                else:
                    print >> filehandle, '%s%s%s%s' %(indent, label, delimiter, _myformat(content))
            else:
                print >> filehandle, '%s%s%s%s' %(indent, label, delimiter, _myformat(content))
        return
    # Allow user to specify print order of (some or all) items in order via 'keyorder'.
    #  'keyorder' is list stored in the dictionary.
    if 'keyorder' in content:
        orderkeys = content['keyorder']
        # Check keys in 'keyorder' are actually present in dictionary : remove those which aren't.
        missingkeys = set(orderkeys).difference(set(content.keys()))
        for m in missingkeys:
            orderkeys.remove(m)
        otherkeys = sorted(list(set(content.keys()).difference(set(orderkeys))))        
        keys = orderkeys + otherkeys
        keys.remove('keyorder')
    else:
        keys = sorted(content.keys())
    # Print data from dictionary.
    print >> filehandle, '%s%s%s' %(indent, delimiter, label)
    _level += 1
    for k in keys:
        _printdict(content[k], k, filehandle, _level, delimiter)
    _level -= 1
