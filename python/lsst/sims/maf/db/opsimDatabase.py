from __future__ import print_function
from builtins import str
from builtins import zip
import os, sys, re
import numpy as np
import warnings
from .database import Database
from lsst.sims.utils import Site
from lsst.sims.maf.utils import getDateVersion

__all__ = ['OpsimDatabase']

class OpsimDatabase(Database):
    def __init__(self, database, driver='sqlite', host=None, port=None, dbTables=None, *args, **kwargs):
        """
        Instantiate object to handle queries of the opsim database.
        (In general these will be the sqlite database files produced by opsim, but could
        be any database holding those opsim output tables.).

        database = Name of database or sqlite filename
        driver =  Name of database dialect+driver for sqlalchemy (e.g. 'sqlite', 'pymssql+mssql')
        host = Name of database host (optional)
        port = String port number (optional)

        """
        # Default dbTables and dbTableIDKey values:
        if 'defaultdbTables' in kwargs:
            defaultdbTables = kwargs.get('defaultdbTables')
            # Remove this kwarg since we're sending it on explicitly
            del kwargs['defaultdbTables']
        else:
            defaultdbTables={'Summary':['Summary', 'obsHistID'],
                             'Cloud':['Cloud', 'cloudID'],
                             'Seeing':['Seeing', 'seeingID'],
                             'Field':['Field', 'fieldID'],
                             'Session':['Session', 'sessionID'],
                             'Config':['Config', 'configID'],
                             'Proposal':['Proposal', 'propID'],
                             'Proposal_Field':['Proposal_Field', 'proposal_field_id'],
                             'ObsHistory':['ObsHistory', 'obsHistID'],
                             'ObsHistory_Proposal':['ObsHistory_Proposal', 'obsHistory_propID'],
                             'SeqHistory':['SeqHistory', 'sequenceID'],
                             'SeqHistory_ObsHistory':['SeqHistory_ObsHistory', 'seqhistory_obsHistID'],
                             'MissedHistory':['MissedHistory', 'missedHistID'],
                             'SeqHistory_MissedHistory':['SeqHistory_MissedHistory',
                                                                  'seqhistory_missedHistID'],
                             'SlewActivities':['SlewActivities', 'slewActivityID'],
                             'SlewHistory':['SlewHistory', 'slewID'],
                             'SlewMaxSpeeds':['SlewMaxSpeeds', 'slewMaxSpeedID'],
                             'SlewState':['SlewState', 'slewIniStatID']
                             }
        # Call base init method to set up all tables and place default values
        # into dbTable/dbTablesIdKey if not overriden.
        super(OpsimDatabase, self).__init__(driver=driver, database=database, host=host, port=port,
                                            dbTables=dbTables,
                                            defaultdbTables=defaultdbTables,
                                            *args, **kwargs)
        # Save filterlist so that we get the filter info per proposal in this desired order.
        self.filterlist = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # Set internal variables for column names.
        self._colNames()

    def _colNames(self):
        """
        Set variables to represent the common column names used in this class directly.

        This should make future schema changes a little easier to handle.
        It is NOT meant to function as a general column map, just to abstract values
        which are used *within this class*.
        """
        self.mjdCol = 'expMJD'
        self.fieldIdCol = 'fieldID'
        self.raCol = 'fieldRA'
        self.decCol = 'fieldDec'
        self.propIdCol = 'propID'
        self.propConfCol = 'propConf'
        self.propNameCol = 'propName' #(propname == proptype)
        # For config parsing.
        self.versionCol = 'version'
        self.sessionDateCol = 'sessionDate'
        self.runCommentCol = 'runComment'

    def fetchMetricData(self, colnames, sqlconstraint, distinctExpMJD=True, groupBy='expMJD',
                        tableName='Summary'):
        """
        Fetch 'colnames' from 'tableName'.

        colnames = the columns to fetch from the table.
        sqlconstraint = sql constraint to apply to data (minus "WHERE").
        distinctExpMJD = group by expMJD to get unique observations only (default True).
        groupBy = group by col 'groupBy' (will override group by expMJD).
        tableName = the opsim table to query.
        """
        # To fetch data for a particular proposal only, add 'propID=[proposalID number]' as constraint,
        #  and to fetch data for a particular filter only, add 'filter ="[filtername]"' as a constraint.
        if (groupBy is None) and (distinctExpMJD is False):
            warnings.warn('Doing no groupBy, data could contain repeat visits that satisfy multiple proposals')

        table = self.tables[tableName]
        if (groupBy is not None) and (groupBy != 'expMJD'):
            if distinctExpMJD:
                warnings.warn('Cannot group by more than one column. Using explicit groupBy col %s' %(groupBy))
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                   constraint = sqlconstraint,
                                                   colnames = colnames, groupByCol = groupBy)
        elif distinctExpMJD:
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                    constraint = sqlconstraint,
                                                    colnames = colnames,
                                                    groupByCol = self.mjdCol)
        else:
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                   constraint = sqlconstraint,
                                                   colnames = colnames)
        return metricdata


    def fetchFieldsFromSummaryTable(self, sqlconstraint, raColName=None, decColName=None):
        """
        Fetch field information (fieldID/RA/Dec) from Output table.
        """
        # Fetch field info from the Output table, by selecting unique fieldID + ra/dec values.
        # This implicitly only selects fields which were actually observed by opsim.
        if raColName is None:
            raColName = self.raCol
        if decColName is None:
            decColName = self.decCol
        table = self.tables['Summary']
        fielddata = table.query_columns_Array(constraint=sqlconstraint,
                                              colnames=[self.fieldIdCol, raColName, decColName],
                                              groupByCol=self.fieldIdCol)
        return fielddata


    def fetchFieldsFromFieldTable(self, propID=None, degreesToRadians=True):
        """
        Fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.

        propID = the proposal ID (default None), if selecting particular proposal - can be a list
        degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians).
        """
        # Note that you can't select any other sql constraints (such as filter).
        # This will select fields which were requested by a particular proposal or proposals,
        #   even if they didn't get any observations.
        tableName = 'Field'
        if propID is not None:
            query = 'select f.%s, f.%s, f.%s from %s as f' %(self.fieldIdCol, self.raCol, self.decCol,
                                                             self.dbTables['Field'][0])
            query += ', %s as p where (p.Field_%s = f.%s) ' %(self.dbTables['Proposal_Field'][0],
                                                            self.fieldIdCol, self.fieldIdCol)
            if hasattr(propID, '__iter__'): # list of propIDs
                query += ' and ('
                for pID in propID:
                    query += '(p.Proposal_%s = %d) or ' %(self.propIdCol, int(pID))
                # Remove the trailing 'or' and add a closing parenthesis.
                query = query[:-3]
                query += ')'
            else: # single proposal ID.
                query += ' and (p.Proposal_%s = %d) ' %(self.propIdCol, int(propID))
            query += ' group by f.%s' %(self.fieldIdCol)
            fielddata = self.queryDatabase(tableName, query)
            if len(fielddata) == 0:
                fielddata = np.zeros(0, dtype=list(zip([self.fieldIdCol, self.raCol, self.decCol],
                                                  ['int', 'float', 'float'])))
        else:
            table = self.tables[tableName]
            fielddata = table.query_columns_Array(colnames=[self.fieldIdCol, self.raCol, self.decCol],
                                                  groupByCol = self.fieldIdCol)
        if degreesToRadians:
            fielddata[self.raCol] = fielddata[self.raCol] * np.pi / 180.
            fielddata[self.decCol] = fielddata[self.decCol] * np.pi / 180.
        return fielddata

    def fetchPropInfo(self):
        """
        Fetch the proposal IDs as well as their (short) proposal names and science type tags from the
        full opsim database.
        Returns dictionary of propID / propname, and dictionary of propTag / propID.
        If not using a full database, will return dict of propIDs with empty propnames + empty propTag dict.
        """
        propIDs = {}
        # Add WFD and DD tags by default to propTags as we expect these every time. (avoids key errors).
        propTags = {'WFD':[], 'DD':[]}
        # If do not have full database available:
        if 'Proposal' not in self.tables:
            propData = self.tables['Summary'].query_columns_Array(colnames=[self.propIdCol])
            for propid in propData[self.propIdCol]:
                propIDs[int(propid)] = propid
        else:
            table = self.tables['Proposal']
            # Query for all propIDs.
            propData = table.query_columns_Array(colnames=[self.propIdCol, self.propConfCol,
                                                           self.propNameCol], constraint='')
            for propid, propname in zip(propData[self.propIdCol], propData[self.propConfCol]):
                # Strip '.conf', 'Prop', and path info.
                propIDs[int(propid)] = re.sub('Prop','', re.sub('.conf','', re.sub('.*/', '', propname)))
            # Find the 'ScienceType' from the config table, to indicate DD/WFD/Rolling, etc.
            table = self.tables['Config']
            sciencetypes = table.query_columns_Array(colnames=['paramValue', 'nonPropID'],
                                                     constraint="paramName like 'ScienceType'")
            if len(sciencetypes) == 0:
                # Then this was an older opsim run without 'ScienceType' tags,
                #   so fall back to trying to guess what proposals are WFD or DD.
                for propid, propname in propIDs.items():
                    if 'universal' in propname.lower():
                        propTags['WFD'].append(propid)
                    if 'deep' in propname.lower():
                        propTags['DD'].append(propid)
            else:
                # Newer opsim output with 'ScienceType' fields in conf files.
                for sc in sciencetypes:
                    # ScienceType tag can be multiple values, separated by a ','
                    tags = [x.strip(' ') for x in sc['paramValue'].split(',')]
                    for sciencetype in tags:
                        if sciencetype in propTags:
                            propTags[sciencetype].append(int(sc['nonPropID']))
                        else:
                            propTags[sciencetype] = [int(sc['nonPropID']),]
        return propIDs, propTags

    def fetchRunLength(self, runLengthParam='nRun'):
        """
        Returns the run length for a particular opsim run (years).

        runLengthParam = the 'paramName' in the config table identifying the run length (default nRun).
        """
        if 'Config' not in self.tables:
            print('Cannot access Config table to retrieve runLength; using default 10 years')
            runLength = 10.0
        else:
            table = self.tables['Config']
            runLength = table.query_columns_Array(colnames=['paramValue'],
                                                  constraint=" paramName = '%s'"%runLengthParam)
            runLength = float(runLength['paramValue'][0]) # Years
        return runLength

    def fetchLatLonHeight(self):
        """
        Returns the latitude, longitude, and height of the telescope used by the config file.
        """
        if 'Config' not in self.tables:
            print('Cannot access Config table to retrieve site parameters; using sims.utils.Site instead.')
            site = Site(name='LSST')
            lat = site.latitude_rad
            lon = site.longitude_rad
            height = site.elev
        else:
            table = self.tables['Config']
            lat = table.query_columns_Array(colnames=['paramValue'],
                                            constraint="paramName = 'latitude'")
            lat = float(lat['paramValue'][0])
            lon = table.query_columns_Array(colnames=['paramValue'],
                                            constraint="paramName = 'longitude'")
            lon = float(lon['paramValue'][0])
            height = table.query_columns_Array(colnames=['paramValue'],
                                               constraint="paramName = 'height'")
            height = float(height['paramValue'][0])
        return lat, lon, height

    def fetchNVisits(self, propID=None):
        """
        Returns the total number of visits in the simulation (or total number of visits for a particular propoal).
        param: propID = the proposal ID (default None), if selecting particular proposal - can be a list
        """
        if 'ObsHistory' in self.dbTables:
            tableName = 'ObsHistory'
            query = 'select count(ObsHistID) from %s' %(self.dbTables[tableName][0])
            if propID is not None:
                query += ', %s where obsHistID=ObsHistory_obsHistID' %(self.dbTables['ObsHistory_Proposal'][0])
                if hasattr(propID, '__iter__'): # list of propIDs
                    query += ' and ('
                    for pID in propID:
                        query += '(Proposal_%s = %d) or ' %(self.propIdCol, int(pID))
                    # Remove the trailing 'or' and add a closing parenthesis.
                    query = query[:-3]
                    query += ')'
                else: # single proposal ID.
                    query += ' and (Proposal_%s = %d) ' %(self.propIdCol, int(propID))
        else:
            tableName = 'Summary'
            query = 'select count(distinct(expMJD)) from %s' %(self.dbTables[tableName][0])
            if propID is not None:
                query += ' where '
                if hasattr(propID, '__iter__'):
                    for pID in propID:
                        query += 'propID=%d or ' %(int(pID))
                    query = query[:-3]
                else:
                    query += 'propID = %d' %(int(propID))
        data = self.tables[tableName].execute_arbitrary(query)
        return int(data[0][0])

    def fetchSeeingColName(self):
        """
        Check whether the seeing column is 'seeing' or 'finSeeing' (v2.x simulator vs v3.0 simulator).
        Returns the name of the seeing column.
        """
        # Really this is just a bit of a hack to see whether we should be using seeing or finseeing.
        # With time, this should probably just go away.
        table = self.tables['Summary']
        try:
            table.query_columns_Array(colnames=['seeing',], numLimit=1)
            seeingcol = 'seeing'
        except ValueError:
            try:
                table.query_columns_Array(colnames=['finSeeing',], numLimit=1)
                seeingcol = 'finSeeing'
            except ValueError:
                raise ValueError('Cannot find appropriate column name for seeing.')
        print('Using %s for seeing column name.' %(seeingcol))
        return seeingcol

    def fetchOpsimRunName(self):
        """
        Returns opsim run name (machine name + session ID) from Session table.
        """
        if 'Session' not in self.tables:
            print('Could not access Session table to find this information.')
            runName = 'opsim'
        else:
            table = self.tables['Session']
            res = table.query_columns_Array(colnames=['sessionID', 'sessionHost'])
            runName = str(res['sessionHost'][0]) + '_' + str(res['sessionID'][0])
        return runName

    def fetchTotalSlewN(self):
        """
        Returns the total slew time.
        """
        if 'SlewActivities' not in self.tables:
            print('Could not access SlewActivities table to find this information.')
            nslew = -1
        else:
            table = self.tables['SlewActivities']
            query = 'select count(distinct(slewHistory_slewID)) from slewActivities where actDelay >0'
            res = table.execute_arbitrary(query)
            nslew = int(res[0][0])
        return nslew

    def fetchRequestedNvisits(self, propId=None):
        """
        Find the requested number of visits for proposals in propId.
        Returns a dictionary - Nvisits{u/g/r/i/z/y}
        """
        visitDict = {}
        if propId is None:
            # Get all the available propIds.
            propData = self.tables['Proposal'].query_columns_Array(colnames=[self.propIdCol, self.propNameCol], constraint='')
        else:
            # Get the propType info to go with the propId(s).
            if hasattr(propId, '__iter__'):
                constraint = '('
                for pi in propId:
                    constraint += '(propId = %d) or ' %(pi)
                constraint = constraint[:-4] + ')'
            else:
                constraint = 'propId = %d' %(propId)
            propData = self.tables['Proposal'].query_columns_Array(colnames=[self.propIdCol, self.propNameCol],
                                                                   constraint=constraint)
        for pId, propType in zip(propData[self.propIdCol], propData[self.propNameCol]):
            perPropConfig = self.tables['Config'].query_columns_Array(colnames=['paramName', 'paramValue'],
                                                                    constraint = 'nonPropID = %d and paramName!="userRegion"'
                                                                                      %(pId))
            filterlist = self._matchParamNameValue(perPropConfig, 'Filter')
            if propType == 'WL':
                # For WL proposals, the simple 'Filter_Visits' == the requested number of observations.
                nvisits = np.array(self._matchParamNameValue(perPropConfig, 'Filter_Visits'), int)
            elif propType == 'WLTSS':
                seqDict, nvisits = self._parseSequences(perPropConfig, filterlist)
            visitDict[pId] = {}
            for f, N in zip(filterlist, nvisits):
                visitDict[pId][f] = N
        nvisits = {}
        for f in ['u', 'g', 'r', 'i', 'z', 'y']:
            nvisits[f] = 0
        for pId in visitDict:
            for f in visitDict[pId]:
                nvisits[f] += visitDict[pId][f]
        return nvisits

    def _matchParamNameValue(self, configarray, keyword):
        return configarray['paramValue'][np.where(configarray['paramName']==keyword)]

    def _parseSequences(self, perPropConfig, filterlist):
        """
        (Private). Given an array of config paramName/paramValue info for a given WLTSS proposal
        and the filterlist of filters used in that proposal, parse the sequences/subsequences and returns:
           a dictionary with all the per-sequence information (including subsequence names & events, etc.)
           a numpy array with the number of visits per filter
        """
        propDict = {}
        # Identify where subsequences start in config[propname] arrays.
        seqidxs = np.where(perPropConfig['paramName'] == 'SubSeqName')[0]
        for sidx in seqidxs:
            i = sidx
            # Get the name of this subsequence.
            seqname = perPropConfig['paramValue'][i]
            # Check if seqname is a nested subseq of an existing sequence:
            nestedsubseq = False
            for prevseq in propDict:
                if 'SubSeqNested' in propDict[prevseq]:
                    if seqname in propDict[prevseq]['SubSeqNested']:
                        seqdict = propDict[prevseq]['SubSeqNested'][seqname]
                        nestedsubseq = True
            # If not, then create a new subseqence key/dictionary for this subseq.
            if not nestedsubseq:
                propDict[seqname] = {}
                seqdict = propDict[seqname]
            # And move on to next parameters within subsequence set.
            i += 1
            if perPropConfig['paramName'][i] == 'SubSeqNested':
                subseqnestedname = perPropConfig['paramValue'][i]
                if subseqnestedname != '.':
                    # Have nested subsequence, so keep track of that here
                    #  but will fill in info later.
                    seqdict['SubSeqNested'] = {}
                    # Set up nested dictionary for nested subsequence.
                    seqdict['SubSeqNested'][subseqnestedname] = {}
                i += 1
            subseqfilters = perPropConfig['paramValue'][i]
            if subseqfilters != '.':
                seqdict['Filters'] = subseqfilters
            i += 1
            subseqexp = perPropConfig['paramValue'][i]
            if subseqexp != '.':
                seqdict['SubSeqExp'] = subseqexp
            i+= 1
            subseqevents = perPropConfig['paramValue'][i]
            seqdict['Events'] = int(subseqevents)
            i+=2
            subseqinterval = perPropConfig['paramValue'][i]
            subseqint = np.array([subseqinterval.split('*')], 'float').prod()
            # In days ..
            subseqint *= 1/24.0/60.0/60.0
            if subseqint > 1:
                seqdict['SubSeqInt'] = '%.2f days' %(subseqint)
            else:
                subseqint *= 24.0
                if subseqint > 1:
                    seqdict['SubSeqInt'] = '%.2f hours' %(subseqint)
                else:
                    subseqint *= 60.0
                    seqdict['SubSeqInt'] = '%.3f minutes' %(subseqint)
        # End of assigning subsequence info - move on to counting number of visits.
        nvisits = np.zeros(len(filterlist), int)
        for subseq in propDict:
            subevents = propDict[subseq]['Events']
            # Count visits from direct subsequences.
            if 'SubSeqExp' in propDict[subseq] and 'Filters' in propDict[subseq]:
                subfilters = propDict[subseq]['Filters']
                subexp = propDict[subseq]['SubSeqExp']
                # If just one filter ..
                if len(subfilters) == 1:
                    idx = np.where(filterlist == subfilters)[0]
                    nvisits[idx] += subevents * int(subexp)
                else:
                    splitsubfilters = subfilters.split(',')
                    splitsubexp = subexp.split(',')
                    for f, exp in zip(splitsubfilters, splitsubexp):
                        idx = np.where(filterlist == f)[0]
                        nvisits[idx] += subevents * int(exp)
            # Count visits if have nested subsequences.
            if 'SubSeqNested' in propDict[subseq]:
                for subseqnested in propDict[subseq]['SubSeqNested']:
                    events = subevents * propDict[subseq]['SubSeqNested'][subseqnested]['Events']
                    subfilters = propDict[subseq]['SubSeqNested'][subseqnested]['Filters']
                    subexp = propDict[subseq]['SubSeqNested'][subseqnested]['SubSeqExp']
                    # If just one filter ..
                    if len(subfilters) == 1:
                        idx = np.where(filterlist == subfilters)[0]
                        nvisits[idx] += events * int(subexp)
                    # Else may have multiple filters in the subsequence, so must split.
                    splitsubfilters = subfilters.split(',')
                    splitsubexp = subexp.split(',')
                    for f, exp in zip(splitsubfilters, splitsubexp):
                        idx = np.where(filterlist == f)[0]
                        nvisits[idx] += int(exp) * events
        return propDict, nvisits

    def fetchConfig(self):
        """
        Fetch config data from configTable, match proposal IDs with proposal names and some field data,
        and do a little manipulation of the data to make it easier to add to the presentation layer.
        """
        # Check to see if we're dealing with a full database or not. If not, just return (no config info to fetch).
        if 'Session' not in self.tables:
            warnings.warn('Cannot fetch opsim config info as this is not a full opsim database.')
            return {}, {}
        # Create two dictionaries: a summary dict that contains a summary of the run
        configSummary = {}
        configSummary['keyorder'] = ['Version', 'RunInfo', 'Proposals']
        #  and the other a general dict that contains all the details (by group) of the run.
        config = {}
        # Start to build up the summary.
        # MAF version
        mafdate, mafversion = getDateVersion()
        configSummary['Version'] = {}
        configSummary['Version']['MAFVersion'] = '%s' %(mafversion['__version__']) \
          + '  RunDate %s' %(mafdate)
        # Opsim date, version and runcomment info from session table
        table = self.tables['Session']
        results = table.query_columns_Array(colnames = [self.versionCol, self.sessionDateCol, self.runCommentCol])
        configSummary['Version']['OpsimVersion'] = '%s'  %(results['version'][0]) + \
            '  RunDate %s' %(results[self.sessionDateCol][0])
        configSummary['RunInfo'] = {}
        configSummary['RunInfo']['RunComment'] = results[self.runCommentCol]
        configSummary['RunInfo']['RunName'] = self.fetchOpsimRunName()
        # Pull out a few special values to put into summary.
        table = self.tables['Config']
        # This section has a number of configuration parameter names hard-coded.
        # I've left these here (rather than adding to self_colNames), because I think schema changes will in the config
        # files will actually be easier to track here (at least until the opsim configs are cleaned up).
        constraint = 'moduleName="Config" and paramName="sha1"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        try:
            configSummary['Version']['Config sha1'] = results['paramValue'][0]
        except IndexError:
            configSummary['Version']['Config sha1'] = 'Unavailable'
        constraint = 'moduleName="Config" and paramName="changedFiles"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        try:
            configSummary['Version']['Config changed files'] = results['paramValue'][0]
        except IndexError:
            configSummary['Version']['Config changed files'] = 'Unavailable'
        constraint = 'moduleName="instrument" and paramName="Telescope_AltMin"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['MinAlt'] = results['paramValue'][0]
        constraint = 'moduleName="instrument" and paramName="Telescope_AltMax"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['MaxAlt'] = results['paramValue'][0]
        constraint = 'moduleName="instrument" and paramName="Filter_MoveTime"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['TimeFilterChange'] = results['paramValue'][0]
        constraint = 'moduleName="instrument" and paramName="Readout_Time"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['TimeReadout'] = results['paramValue'][0]
        constraint = 'moduleName="scheduler" and paramName="MinDistance2Moon"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['MinDist2Moon'] = results['paramValue'][0]
        configSummary['RunInfo']['keyorder'] = ['RunName', 'RunComment', 'MinDist2Moon', 'MinAlt', 'MaxAlt',
                                                'TimeFilterChange', 'TimeReadout']
        # Now build up config dict with 'nice' group names (proposal name and short module name)
        #  Each dict entry is a numpy array with the paramName/paramValue/comment values.
        # Match proposal IDs with names.
        query = 'select %s, %s, %s from Proposal group by %s' %(self.propIdCol, self.propConfCol,
                                                                self.propNameCol, self.propIdCol)
        propdata = self.queryDatabase('Proposal', query)
        # Make 'nice' proposal names
        propnames = np.array([os.path.split(x)[1].replace('.conf', '') for x in propdata[self.propConfCol]])
        # Get 'nice' module names
        moduledata = table.query_columns_Array(colnames=['moduleName',], constraint='nonPropID=0')
        modulenames = np.array([os.path.split(x)[1].replace('.conf', '') for x in moduledata['moduleName']])
        # Grab the config information for each proposal and module.
        cols = ['paramName', 'paramValue', 'comment']
        for longmodname, modname in zip(moduledata['moduleName'], modulenames):
            config[modname] = table.query_columns_Array(colnames=cols,
                                                        constraint='moduleName="%s"' %(longmodname))
            config[modname] = config[modname][['paramName', 'paramValue', 'comment']]
        for propid, propname in zip(propdata[self.propIdCol], propnames):
            config[propname] = table.query_columns_Array(colnames=cols,
                                                         constraint=
                                                         'nonPropID="%s" and paramName!="userRegion"' %(propid))
            config[propname] = config[propname][['paramName', 'paramValue', 'comment']]
        config['keyorder'] = ['Comment', 'LSST', 'site', 'instrument', 'filters',
                              'AstronomicalSky', 'File', 'scheduler',
                              'schedulingData', 'schedDown', 'unschedDown']
        # Now finish building the summary to add proposal information.
        # Loop through all proposals to add summary information.
        configSummary['Proposals'] = {}
        propidorder = sorted(propdata[self.propIdCol])
        # Generate a keyorder to print proposals in order of propid.
        configSummary['Proposals']['keyorder'] = []
        for propid in propidorder:
            configSummary['Proposals']['keyorder'].append(propnames[np.where(propdata[self.propIdCol] == propid)][0])
        for propid, propname in zip(propdata[self.propIdCol], propnames):
            configSummary['Proposals'][propname] = {}
            propdict = configSummary['Proposals'][propname]
            propdict['keyorder'] = [self.propIdCol, self.propNameCol, 'PropType', 'RelPriority', 'NumUserRegions', 'NumFields']
            propdict[self.propNameCol] = propname
            propdict[self.propIdCol] = propid
            propdict['PropType'] = propdata[self.propNameCol][np.where(propnames == propname)]
            propdict['RelPriority'] = self._matchParamNameValue(config[propname], 'RelativeProposalPriority')
            # Get the number of user regions.
            constraint = 'nonPropID="%s" and paramName="userRegion"' %(propid)
            result = table.query_columns_Array(colnames=['paramName',], constraint=constraint)
            propdict['NumUserRegions'] = result.size
            # Get the number of fields requested in the proposal (all filters).
            propdict['NumFields'] = self.fetchFieldsFromFieldTable(propID=propid).size
            # Find number of visits requested per filter for the proposal, along with min/max sky and airmass values.
            # Note that config table has multiple entries for Filter/Filter_Visits/etc. with the same name.
            #   The order of these entries in the config array matters.
            propdict['PerFilter'] = {}
            for key, keyword in zip(['Filters', 'MaxSeeing', 'MinSky', 'MaxSky'],
                                    ['Filter', 'Filter_MaxSeeing', 'Filter_MinBrig', 'Filter_MaxBrig']):
                temp = self._matchParamNameValue(config[propname], keyword)
                if len(temp) > 0:
                    propdict['PerFilter'][key] = temp
            # Add exposure time, potentially looking for scaling per filter.
            exptime = float(self._matchParamNameValue(config[propname], 'ExposureTime')[0])
            temp = self._matchParamNameValue(config[propname], 'Filter_ExpFactor')
            if len(temp) > 0:
                propdict['PerFilter']['VisitTime'] = temp * exptime
            else:
                propdict['PerFilter']['VisitTime'] = np.ones(len(propdict['PerFilter']['Filters']), float)
                propdict['PerFilter']['VisitTime'] *= exptime
            # And count how many total exposures are requested per filter.
            # First check if 'RestartCompleteSequences' is true:
            #   if both are true, then basically an indefinite number of visits are requested, although we're
            #   not going to count this way (as the proposals still make an approximate number of requests).
            restartComplete = False
            temp = self._matchParamNameValue(config[propname], 'RestartCompleteSequences')
            if len(temp) > 0:
                if temp[0] == 'True':
                    restartComplete = True
            propdict['RestartCompleteSequences'] = restartComplete
            # Grab information on restarting lost sequences so we can print this too.
            restartLost = False
            tmp = self._matchParamNameValue(config[propname], 'RestartLostSequences')
            if len(temp) > 0:
                if temp[0]  == 'True':
                    restartLost = True
            propdict['RestartLostSequences'] = restartLost
            if propdict['PropType'] == 'WL':
                # Simple 'Filter_Visits' request for number of observations.
                propdict['PerFilter']['NumVisits'] = np.array(self._matchParamNameValue(config[propname],
                                                                                   'Filter_Visits'), int)
            elif propdict['PropType'] == 'WLTSS':
                # Proposal contains subsequences and possible nested subseq, so must delve further.
                # Make a dictionary to hold the subsequence info (keyed per subsequence).
                propdict['SubSeq'], Nvisits = self._parseSequences(config[propname], propdict['PerFilter']['Filters'])
                propdict['PerFilter']['NumVisits'] = Nvisits
                propdict['SubSeq']['keyorder'] = ['SubSeqName', 'SubSeqNested', 'Events', 'SubSeqInt']
            # Sort the filter information so it's ugrizy instead of order in opsim config db.
            idx = []
            for f in self.filterlist:
                filterpos = np.where(propdict['PerFilter']['Filters'] == f)
                if len(filterpos[0]) > 0:
                    idx.append(filterpos[0][0])
            idx = np.array(idx, int)
            for k in propdict['PerFilter']:
                propdict['PerFilter'][k] = propdict['PerFilter'][k][idx]
            propdict['PerFilter']['keyorder'] = ['Filters', 'VisitTime', 'MaxSeeing', 'MinSky',
                                                 'MaxSky', 'NumVisits']
        return configSummary, config
