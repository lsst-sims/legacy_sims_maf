# Utilities for dealing with the opsim config files (reading the config parameters and pretty printing them)

import os, sys
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning) # Ignore db warning
    import lsst.sims.maf.db as db


def _ModulePropID2LongName(moduleName, propID):
    return '__'.join([moduleName, str(propID)])

def _LongName2ModulePropID(longName):
    moduleName = longName.split('__')[0]
    propID = int(longName.split('__')[1])
    return moduleName, propID

def fetchConfigs(dbAddress, configTable='Config', proposalTable='Proposal', proposalFieldTable='Proposal_Field'):
    """Utility to fetch config data from configTable, match proposal IDs with proposal names and some field data,
       and do a little manipulation of the data to make it easier to add to the presentation layer.
    
    Returns dictionary keyed by proposals and module names, and within each of these is another dictionary
    containing the paramNames and paramValues relevant for that module or proposal.
    """
    # Get config table data.
    table = db.Table(configTable, 'configID', dbAddress)
    # If opsim add descriptions to the 'comment' variable, grab that here too and use as 'description' in outputs.
    cols = ['moduleName', 'paramName', 'paramValue', 'nonPropID']
    configdata = table.query_columns_RecArray(colnames=cols)
    # Get proposal table data.
    table = db.Table(proposalTable, 'propID', dbAddress)
    cols = ['propID', 'propConf', 'propName']
    propdata = table.query_columns_RecArray(colnames=cols)
    # Get counts of fields from proposal_field data.
    table = db.Table(proposalFieldTable, 'proposal_field_id', dbAddress)
    cols = ['proposal_field_id', 'Proposal_propID']
    propfielddata = table.query_columns_RecArray(colnames=cols)    
    # Test that proposal ids are present in both proposal and config tables.
    configPropIDs = set(configdata['nonPropID'])
    configPropIDs.remove(0)
    propPropIDs = set(propdata['propID'])
    if configPropIDs.intersection(propPropIDs) != propPropIDs:
        raise Exception('Found proposal IDs in proposal table which are not present in config table.')
    if configPropIDs.intersection(propPropIDs) != configPropIDs:
        raise Exception('Found proposal IDs in config table which are not present in proposal table.')
    # Identify unique proposals and modules by joining moduleName and nonPropID.
    longNames = []
    for modName, propID in zip(list(configdata['moduleName']), list(configdata['nonPropID'])):
        longNames.append(_ModulePropID2LongName(modName, propID))
    longNames = set(longNames)
    configDict = {}
    # Group module data together.
    for name in longNames:
        configDict[name] = {}
        moduleName, propID = _LongName2ModulePropID(name)
        # Add propID and module name.
        configDict[name]['propID'] = propID
        configDict[name]['moduleName'] = moduleName
        # Add key/value pairs to dictionary for most paramName/paramValue pairs in module.
        condition1 = np.where(configdata['moduleName'] == moduleName, True, False)
        condition2 = np.where(configdata['nonPropID'] == propID, True, False)
        condition = condition1 * condition2
        for key, value in zip(configdata['paramName'][condition], configdata['paramValue'][condition]):
            if key != 'userRegion':
                if key not in configDict[name]:           
                    configDict[name][key] = [value,]
                else:
                    configDict[name][key].append(value)
        # Just count user regions and add summary to config info.
        condition2 = (configdata['paramName'][condition] == 'userRegion')
        numberUserRegions = configdata['paramName'][condition2].size
        if numberUserRegions > 0:
            configDict[name]['numUserRegions'] = numberUserRegions
        # For actual proposals:
        if propID != 0:
            # And add a count of the numer of actual fields used in proposal.
            condition3 = (propfielddata['Proposal_propID'] == propID)
            configDict[name]['numFields'] = propfielddata[condition3].size
            # Add full proposal names.
            condition3 = (propdata['propID'] == propID)
            configDict[name]['proposalFile'] = propdata['propConf'][condition3][0]
            configDict[name]['proposalType'] = propdata['propName'][condition3][0]
            # Calculate the number of visits requested per filter
            if 'Filter_Visits' in configDict[name]:
                # This is a 'normal' WLprop type, simple request of visits per filter.
                configDict[name]['numVisitsReq'] = configDict[name]['Filter_Visits']
            else:
                # This is one of the other types of proposals and must look at subsequences.
                configDict[name]['numVisitsReq'] = []
                for f in configDict[name]['Filter']:
                    configDict[name]['numVisitsReq'].append(0)
                for subevents, subexposures, subfilters in zip(configDict[name]['SubSeqEvents'],
                                                               configDict[name]['SubSeqExposures'],
                                                               configDict[name]['SubSeqFilters']):
                    # If non-multi-filter subsequence (i.e. just one filter per subseq)
                    if subfilters in configDict[name]['Filter']:
                        idx = configDict[name]['Filter'].index(subfilters)
                        configDict[name]['numVisitsReq'][idx] = int(subevents) * int(subexposures)
                    # Else we may have multiple filters in this subsequence, so split.
                    else:
                        splitsubfilters = subfilters.split(',')
                        splitsubexposures = subexposures.split(',')
                        for f, exp in zip(splitsubfilters, splitsubexposures):
                            if f in configDict[name]['Filter']:
                                idx = configDict[name]['Filter'].index(f)
                                configDict[name]['numVisitsReq'][idx] = int(subevents) * int(exp)
        # Find a pretty name to label each group of configs.
        if propID == 0:
            groupName = moduleName
            configDict[name]['groupName'] = os.path.split(groupName)[1]
        else:            
            groupName = configDict[name]['proposalFile']
            configDict[name]['groupName'] = os.path.split(groupName)[1]
    return configDict

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
        groupOrder = ['LSST', 'site', 'filters', 'instrument', 'scheduler', 'schedulingData',
                      'schedDown', 'unschedDown']
        keyOrder = []
        # Identify dictionary key name that goes with group name.
        for g in groupOrder:
            for k in configDict:
                if configDict[k]['groupName'] == g:
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
