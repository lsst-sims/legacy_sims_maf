import os, sys
import numpy as np
import warnings

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
