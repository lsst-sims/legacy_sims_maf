import os, sys
import warnings
import numpy as np

# Generic line formatter (to let you specify delimiter between text fields)
def _myformat(args, delimiter=' '):
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

def _printdict(content, label, filehandle=None, level=0, delimiter=' '):
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
    level += 1
    for k in keys:
        _printdict(content[k], k, filehandle, level, delimiter)
    level -= 1



def pp(config, outfile=None, delimiter=' ', header=None):
    if outfileRoot != None:
        f = open(outfileRoot, 'w')
    else:
        f = sys.stdout
    if header is not None:
        print >>f, header


# Save config info (for use in presentation layer)
def printConfigs(configDict, outfileRoot=None, delimiter=' ', printPropConfig=True, printGeneralConfig=True):
    """Utility to pretty print the configDict, grouping data from different modules and proposals and providing some
    ordering to the proposal information.
    Writes config data out to 'outfile' (default stdout), using 'delimiter' between parameter entries.
    
    'printProposalConfig' (default True) toggles detailed proposal config info on/off.
    'printGeneralConfig' (default True) toggles detailed general config info on/off. """
    if outfileRoot != None:
        f = open(outfileRoot+'_configSummary.txt', 'w')
    else:
        f = sys.stdout
    # Summarize proposals in run.
    print >>f, '## Proposal summary information'
    print >>f, '## '
    line = _myformat(['ProposalName', 'PropID', 'PropType', 'RelativePriority',
                  'NumUserRegions', 'NumFields', 'Filters', 'VisitsPerFilter(Req)'], delimiter=delimiter)
    print >>f, line
    for k in configDict:
        if configDict[k]['propID'] != 0:            
            line = _myformat([configDict[k]['groupName'], configDict[k]['propID'],
                                configDict[k]['proposalType'], configDict[k]['RelativeProposalPriority'],
                                configDict[k]['numUserRegions'], configDict[k]['numFields'],
                                configDict[k]['Filter'], configDict[k]['numVisitsReq']], delimiter=delimiter)
            print >>f, line
    if outfileRoot != None:
        f.close()
    # Print general info for each proposal.
    if printPropConfig:
        if outfileRoot != None:
            f = open(outfileRoot+'_configProps.txt', 'w')
        else:
            f = sys.stdout
        print >>f, '## Detailed proposal information'
        for k in configDict:
            if configDict[k]['propID'] != 0:
                print >>f, '## '
                print >>f, '## Information for proposal %s : propID %d (%s)' %(configDict[k]['groupName'],
                                                                          configDict[k]['propID'], k)
                print >>f, '## '
                # This is a proposal; print the information in alphabetical order.
                propkeys = sorted(configDict[k].keys())
                propkeys.remove('groupName')
                for p in propkeys:
                    line = _myformat([p, configDict[k][p]], delimiter=delimiter)
                    print >>f, line
        if outfileRoot != None:
            f.close()
    # Print general config info.
    if printGeneralConfig:
        if outfileRoot != None:
            f = open(outfileRoot+'_configGeneral.txt', 'w')
        else:
            f = sys.stdout
        groupOrder = ['LSST', 'site', 'filters', 'instrument', 'scheduler', 'schedulingData', 'schedDown', 'unschedDown']
        keyOrder = []
        # Identify dictionary key name that goes with group name.
        for g in groupOrder:
            for k in configDict:
                if g in k:
                    keyOrder.append(k)
        for g, k in zip(groupOrder, keyOrder):
            print >>f, '## '
            print >>f, '## Printing config information for %s group (%s)' %(g, k)
            print >>f, '## '
            paramkeys = sorted(configDict[k].keys())
            paramkeys.remove('groupName')
            paramkeys.remove('propID')
            for param in paramkeys:
                line = _myformat([param, configDict[k][param]], delimiter=delimiter)
                print >>f, line
        if outfileRoot != None:
            f.close()
