import os, sys, re
import numpy as np
import warnings
from .Table import Table
from .Database import Database
from lsst.sims.maf.utils.getDateVersion import getDateVersion

class OpsimDatabase(Database):
    def __init__(self, dbAddress, dbTables=None, *args, **kwargs):
        """
        Instantiate object to handle queries of the opsim database.
        (In general these will be the sqlite database files produced by opsim, but could
        be any database holding those opsim output tables.).

        dbAddress = sqlalchemy connection string to database
        dbTables = dictionary of names of tables in the code : [names of tables in the database, names of primary keys]
        Note that for the dbTables there are defaults in the init --
          you can override (specific key:value pairs only if desired) by passing a dictionary in dbTables.

        The dbAddress sqlalchemy string should look like:
           dialect+driver://username:password@host:port/database
        Examples:
           sqlite:///opsim_sqlite.db   (sqlite is special -- the three /// indicate the start of the path to the file)
           mysql://lsst:lsst@localhost/opsim
        More information on sqlalchemy connection strings can be found at
          http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html     
          
        """
        self.dbAddress = dbAddress        
        # Default dbTables and dbTableIDKey values:        
        if 'defaultdbTables' in kwargs:
            defaultdbTables = kwargs.get('defaultdbTables')
            # Remove this kwarg since we're sending it on explicitly
            del kwargs['defaultdbTables']
        else:
            defaultdbTables={'summaryTable':['Summary', 'obsHistID'],
                             'cloudTable':['Cloud', 'cloudID'],
                             'seeingTable':['Seeing', 'seeingID'],
                             'fieldTable':['Field', 'fieldID'],
                             'sessionTable':['Session', 'sessionID'],
                             'configTable':['Config', 'configID'],
                             'proposalTable':['Proposal', 'propID'],
                             'proposalFieldTable':['Proposal_Field', 'proposal_field_id'],
                             'obsHistoryTable':['ObsHistory', 'obsHistID'],
                             'obsHistoryProposalTable':['Obshistory_Proposal', 'obsHistory_propID'],
                             'sequenceHistoryTable':['SeqHistory', 'sequenceID'],
                             'sequenceHistoryObsHistoryTable':['SeqHistory_ObsHistory', 'seqhistory_obsHistID'],
                             'missedHistoryTable':['MissedHistory', 'missedHistID'],
                             'sequenceHistoryMissedHistoryTable':['SeqHistory_MissedHistory',
                                                                  'seqhistory_missedHistID'],
                             'slewActivitiesTable':['SlewActivities', 'slewActivityID'],
                             'slewHistoryTable':['SlewHistory', 'slewID'],
                             'slewMaxSpeedsTable':['SlewMaxSpeeds', 'slewMaxSpeedID'],
                             'slewStateTable':['SlewState', 'slewIniStatID']
                             }
        # Call base init method to set up all tables and place default values
        # into dbTable/dbTablesIdKey if not overriden.
        super(OpsimDatabase, self).__init__(dbAddress, dbTables=dbTables,
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
        """
        self.mjdCol = 'expMJD'
        self.fieldIdCol = 'fieldID'
        self.raCol = 'fieldRA'
        self.decCol = 'fieldDec'
        self.propIdCol = 'propID'
        self.propConfCol = 'propConf'
        self.propNameCol = 'propName'
        # For config parsing.
        self.versionCol = 'version'
        self.sessionDateCol = 'sessionDate'
        self.runCommentCol = 'runComment'

    def fetchMetricData(self, colnames, sqlconstraint, distinctExpMJD=True, groupBy='expMJD'):
        """Fetch 'colnames' from 'Summary' table. 

        colnames = the columns to fetch from the table.
        sqlconstraint = sql constraint to apply to data (minus "WHERE").
        distinctExpMJD = group by expMJD to get unique observations only (default True).
        groupBy = group by col 'groupBy' (will override group by expMJD)."""
        # To fetch data for a particular proposal only, add 'propID=[proposalID number]' as constraint,
        #  and to fetch data for a particular filter only, add 'filter ="[filtername]"' as a constraint.

        if (groupBy is None) and (distinctExpMJD is False):
            warnings.warn('Doing no groupBy, data could contain repeat visits that satisfy multiple proposals')
        
        table = self.tables['summaryTable']
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
            raColName = self.raColName
        if decColName is None:
            decColName = self.decColName
        table = self.tables['summaryTable']
        fielddata = table.query_columns_Array(constraint=sqlconstraint,
                                              colnames=[self.fieldIdCol, raColName, decColName],
                                              groupByCol=self.fieldIdCol)
        return fielddata


    def fetchFieldsFromFieldTable(self, propID=None, degreesToRadians=True):
        """
        Fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.
    
        propID = the proposal ID (default None), if selecting particular proposal - can be a list
        degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians) """
        # Note that you can't select any other sql constraints (such as filter). 
        # This will select fields which were requested by a particular proposal or proposals,
        #   even if they didn't get any observations. 
        tableName = 'fieldTable'
        if propID is not None:
            query = 'select f.%s, f.%s, f.%s from %s as f' %(self.fieldIdCol, self.raCol, self.decCol,
                                                             self.dbTables['fieldTable'][0])
            query += ', %s as p where (p.Field_%s = f.%s) ' %(self.dbTables['proposalFieldTable'][0],
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
        else:
            table = self.tables[tableName]
            fielddata = table.query_columns_Array(colnames=[self.fieldIdCol, self.raCol, self.decCol],
                                                  groupByCol = self.fieldIdCol)
        if degreesToRadians:
            fielddata[self.raCol] = fielddata[self.raCol] * np.pi / 180.
            fielddata[self.decCol] = fielddata[self.decCol] * np.pi / 180.
        return fielddata

    def fetchPropIDs(self):
        """
        Fetch the proposal IDs from the full opsim run database.
        Return the full list of ID numbers as well as a list of
         WFD propIDs (proposals containing 'Universal' in the name) -- or tagged with wfd,
         deep drilling propIDs (proposals containing 'deep', 'Deep', 'dd' or 'DD' in the name) -- or tagged dd
         and a dict keyed with the ID and values of the truncated proposal config file.
         """
        # Check if using full database; otherwise can only fetch list of all propids. 
        if 'proposalTable' not in self.tables:
            propData = self.tables['summaryTable'].query_columns_Array(colnames=[self.propIdCol])
            propIDs = np.array(propData[self.propIdCol], int)
            wfdIDs = []
            ddIDs = []
            propID2Name = {}
        else:
            table = self.tables['proposalTable']
            try:
                propData = table.query_columns_Array(colnames=[self.propIdCol, self.propConfCol,
                                                               self.propNameCol, 'tag'], constraint='')
            except ValueError:
                propData = table.query_columns_Array(colnames=[self.propIdCol, self.propConfCol,
                                                               self.propNameCol], constraint='')
            propIDs = np.array(propData[self.propIdCol], int)
            propIDs = list(propIDs)
            if 'tag' in propData.dtype.names:
                wfdMatch = (propData['tag'] == 'wfd')
                wfdIDs = list(propData['propID'][wfdMatch])
            else:
                wfdIDs = []
                for name, propid in zip(propData[self.propConfCol] ,propIDs):
                    if 'universal' in name.lower():
                        wfdIDs.append(propid)
            # Parse on name for DD anyway.
            ddIDs = []
            for name, propid in zip(propData[self.propConfCol], propIDs):
                if ('deep' in name.lower()) or ('dd' in name.lower()):
                    ddIDs.append(propid)
            propID2Name = {}
            for propID, propName in zip(propData[self.propIdCol], propData[self.propConfCol] ):
                # Strip '.conf', 'Prop', and path info.
                propID2Name[propID] = re.sub('Prop','', re.sub('.conf','', re.sub('.*/', '', propName)))
        return propIDs, wfdIDs, ddIDs, propID2Name

    def fetchRunLength(self, runLengthParam='nRun'):
        """Fetch the run length for a particular opsim run.

        runLengthParam = the 'paramName' in the config table identifying the run length (default nRun)."""
        table = self.tables['configTable']
        runLength = table.query_columns_Array(colnames=['paramValue'], constraint=" paramName = '%s'"%runLengthParam)
        runLength = float(runLength['paramValue'][0]) # Years
        return runLength

    def fetchNVisits(self, propID=None):
        """Fetch the total number of visits in the simulation (or total number of visits for a particular propoal).
        Convenience function for setting user-defined benchmark values.
        
        propID = the proposal ID (default None), if selecting particular proposal - can be a list
        """
        tableName = 'obsHistoryTable'
        query = 'select %s from %s' %(self.mjdCol, self.dbTables[tableName][0])
        if propID is not None:
            query += ', %s where obsHistID=ObsHistory_obsHistID' %(self.dbTables['obsHistoryProposalTable'][0])
            if hasattr(propID, '__iter__'): # list of propIDs
                query += ' and ('
                for pID in propID:
                    query += '(Proposal_%s = %d) or ' %(self.propIdCol, int(pID))
                # Remove the trailing 'or' and add a closing parenthesis.
                query = query[:-3]
                query += ')'
            else: # single proposal ID.
                query += ' and (Proposal_%s = %d) ' %(self.propIdCol, int(propID))
        data = self.queryDatabase(tableName, query)
        return data.size

    def fetchSeeingColName(self):
        """Check whether the seeing column is 'seeing' or 'finSeeing' (v2.x simulator vs v3.0 simulator)."""
        # Really this is just a bit of a hack to see whether we should be using seeing or finseeing.
        # With time, this should probably just go away.
        table = self.tables['summaryTable']
        try:
            table.query_columns_Array(colnames=['seeing',], numLimit=1)
            seeingcol = 'seeing'
        except ValueError:
            try:
                table.query_columns_Array(colnames=['finSeeing',], numLimit=1)
                seeingcol = 'finSeeing'
            except ValueError:
                raise ValueError('Cannot find appropriate column name for seeing.')
        print 'Using %s for seeing column name.' %(seeingcol)
        return seeingcol

    def fetchOpsimRunName(self):
        """Pull opsim run name (machine name + session ID) from Session table. Return string."""
        table = self.tables['sessionTable']
        res = table.query_columns_Array(colnames=['sessionID', 'sessionHost'])
        runName = str(res['sessionHost'][0]) + '_' + str(res['sessionID'][0])
        return runName
    
    def fetchConfig(self):
        """Fetch config data from configTable, match proposal IDs with proposal names and some field data,
        and do a little manipulation of the data to make it easier to add to the presentation layer.    
        """
        # Check to see if we're dealing with a full database or not. If not, just return (no config info to fetch).
        if 'sessionTable' not in self.tables:
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
        table = self.tables['sessionTable']
        results = table.query_columns_Array(colnames = [self.versionCol, self.sessionDateCol, self.runCommentCol])
        configSummary['Version']['OpsimVersion'] = '%s'  %(results['version'][0]) + \
            '  RunDate %s' %(results[self.sessionDateCol][0])
        configSummary['RunInfo'] = {}        
        configSummary['RunInfo']['RunComment'] = results[self.runCommentCol]
        configSummary['RunInfo']['RunName'] = self.fetchOpsimRunName()
        # Pull out a few special values to put into summary.
        table = self.tables['configTable']
        # This section has a number of configuration parameter names hard-coded.
        # I've left these here (rather than adding to self_colNames), because I think schema changes will in the config
        # files will actually be easier to track here (at least until the opsim configs are cleaned up).
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
        propdata = self.queryDatabase('proposalTable', query)
        # Make 'nice' proposal names
        propnames = np.array([os.path.split(x)[1].replace('.conf', '') for x in propdata[self.propConfCol]])
        # Get 'nice' module names
        moduledata = table.query_columns_Array(colnames=['moduleName',], constraint='nonPropID=0')
        modulenames = np.array([os.path.split(x)[1].replace('.conf', '') for x in moduledata['moduleName']])
        # Grab the config information for each proposal and module.
        cols = ['paramName', 'paramValue', 'comment']
        for longmodname, modname in zip(moduledata['moduleName'], modulenames):
            config[modname] = table.query_columns_Array(colnames=cols, constraint='moduleName="%s"' %(longmodname))
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
        def _matchParamNameValue(configarray, keyword):
            return configarray['paramValue'][np.where(configarray['paramName']==keyword)]
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
            propdict['RelPriority'] = _matchParamNameValue(config[propname], 'RelativeProposalPriority')
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
                temp = _matchParamNameValue(config[propname], keyword)
                if len(temp) > 0:
                    propdict['PerFilter'][key] = temp
            # Add exposure time, potentially looking for scaling per filter.
            exptime = float(_matchParamNameValue(config[propname], 'ExposureTime')[0])
            temp = _matchParamNameValue(config[propname], 'Filter_ExpFactor')
            if len(temp) > 0:
                propdict['PerFilter']['VisitTime'] = temp * exptime
            else:
                propdict['PerFilter']['VisitTime'] = np.ones(len(propdict['PerFilter']['Filters']), float)
                propdict['PerFilter']['VisitTime'] *= exptime
            # And count how many total exposures are requested per filter.
            # First check if 'RestartCompleteSequences' are true:
            #   if both are true, then basically an indefinite number of visits are requested.
            restartComplete = False
            temp = _matchParamNameValue(config[propname], 'RestartCompleteSequences')
            if len(temp) > 0:
                if temp[0] == 'True':
                    restartComplete = True
            if propdict['PropType'] == 'WL':
                # Simple 'Filter_Visits' request for number of observations.
                propdict['PerFilter']['NumVisits'] = np.array(_matchParamNameValue(config[propname],
                                                                                   'Filter_Visits'), int)
                if restartComplete:
                    propdict['PerFilter']['NumVisits'] = 'Indefinite'
            elif propdict['PropType'] == 'WLTSS':
                # Proposal contains subsequences and possible nested subseq, so must delve further.
                # Make a dictionary to hold the subsequence info (keyed per subsequence).
                propdict['SubSeq'] = {}
                # Identify where subsequences start in config[propname] arrays.
                seqidxs = np.where(config[propname]['paramName'] == 'SubSeqName')[0]
                # Assign subsequence info to configSummary['Proposals'][propname]['SubSeq'][subseqname]
                for sidx in seqidxs:
                    # This is fragile and depends on order from database query. However, it's the
                    #  best I think we can do with the current method of storing these values in the Config table.
                    i = sidx                    
                    seqname = config[propname]['paramValue'][i]
                    # Check if seqname is a nested subseq of an existing sequence:
                    nestedsubseq = False
                    prevseqs = propdict['SubSeq'].keys()
                    for ps in prevseqs:
                        if 'SubSeqNested' in propdict['SubSeq'][ps]:
                            if seqname in propdict['SubSeq'][ps]['SubSeqNested']:
                                seqdict = propdict['SubSeq'][ps]['SubSeqNested'][seqname]
                                nestedsubseq = True
                    # If not, then create a new subseqence key/dictionary for this subseq.
                    if not nestedsubseq:
                        propdict['SubSeq'][seqname] = {}
                        seqdict = propdict['SubSeq'][seqname]
                    # And move on to next parameters within subsequence set.
                    i += 1
                    if config[propname]['paramName'][i] == 'SubSeqNested':
                        subseqnestedname = config[propname]['paramValue'][i]
                        if subseqnestedname != '.':
                            # Have nested subsequence, so keep track of that here
                            #  but will fill in info later.
                            seqdict['SubSeqNested'] = {}
                            # Set up nested dictionary for nested subsequence.
                            seqdict['SubSeqNested'][subseqnestedname] = {}
                        i += 1
                    subseqfilters = config[propname]['paramValue'][i]
                    if subseqfilters != '.':
                        seqdict['Filters'] = subseqfilters
                    i += 1
                    subseqexp = config[propname]['paramValue'][i]
                    if subseqexp != '.':
                        seqdict['Visits'] = subseqexp
                    i+= 1
                    subseqevents = config[propname]['paramValue'][i]
                    seqdict['Events'] = int(subseqevents)
                # End of assigning subsequence info - move on to counting number of visits.
                if restartComplete:
                    propdict['PerFilter']['NumVisits'] = 'Indefinite'
                else:
                    propdict['PerFilter']['NumVisits'] = np.zeros(len(propdict['PerFilter']['Filters']), int)
                    subseqs = propdict['SubSeq'].keys()
                    for subseq in subseqs:
                        subevents = propdict['SubSeq'][subseq]['Events']
                        # Count visits from direct subsequences.
                        if 'Visits' in propdict['SubSeq'][subseq] and 'Filters' in propdict['SubSeq'][subseq]:
                            subfilters = propdict['SubSeq'][subseq]['Filters']
                            subexp = propdict['SubSeq'][subseq]['Visits']
                            # If just one filter ..
                            if len(subfilters) == 1:
                                idx = (propdict['PerFilter']['Filters'] == subfilters)
                                propdict['PerFilter']['NumVisits'][idx] += subevents * int(subexp)
                            else:
                                splitsubfilters = subfilters.split(',')
                                splitsubexp = subexp.split(',')
                                for f, exp in zip(splitsubfilters, splitsubexp):
                                    idx = (propdict['PerFilter']['Filters'] == f)
                                    propdict['PerFilter']['NumVisits'][idx] += subevents * int(exp)
                        # Count visits if have nested subsequences.
                        if 'SubSeqNested' in propdict['SubSeq'][subseq]:
                            for subseqnested in propdict['SubSeq'][subseq]['SubSeqNested']:
                                events = subevents * propdict['SubSeq'][subseq]['SubSeqNested'][subseqnested]['Events']
                                subfilters = propdict['SubSeq'][subseq]['SubSeqNested'][subseqnested]['Filters']
                                subexp = propdict['SubSeq'][subseq]['SubSeqNested'][subseqnested]['Visits']
                                # If just one filter .. 
                                if len(subfilters) == 1:
                                    idx = (propdict['PerFilter']['Filters'] == subfilters)
                                    propdict['PerFilter']['NumVisits'][idx] += events * int(subexp)
                                # Else may have multiple filters in the subsequence, so must split.
                                splitsubfilters = subfilters.split(',')
                                splitsubexp = subexp.split(',')
                                for f, exp in zip(splitsubfilters, splitsubexp):
                                    idx = (propdict['PerFilter']['Filters'] == f)
                                    propdict['PerFilter']['NumVisits'][idx] += int(exp) * events
                        propdict['SubSeq']['keyorder'] = ['SubSeqName', 'SubSeqNested', 'Events']
            # Sort the filter information so it's ugrizy instead of order in opsim config db.
            idx = []
            for f in self.filterlist:
                filterpos = np.where(propdict['PerFilter']['Filters'] == f)
                if len(filterpos[0]) > 0:
                    idx.append(filterpos[0][0])
            idx = np.array(idx, int)
            for k in propdict['PerFilter']:
                # Skip numvisits if restartComplete (because just a string)
                if restartComplete and k == 'NumVisits':
                    continue
                else:
                    propdict['PerFilter'][k] = propdict['PerFilter'][k][idx]
            propdict['PerFilter']['keyorder'] = ['Filters', 'VisitTime', 'MaxSeeing', 'MinSky',
                                                 'MaxSky', 'NumVisits']
        return configSummary, config
