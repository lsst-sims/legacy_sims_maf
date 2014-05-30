import os, sys
import numpy as np
from .table import Table
from .database import Database
from lsst.sims.maf.utils.getDateVersion import getDateVersion

class OpsimDatabase(Database):
    def __init__(self, dbAddress, dbTables=None, dbTablesIdKey=None, *args, **kwargs):
        """Instantiate object to handle queries of the opsim database.
        (In general these will be the sqlite database files produced by opsim, but could
        be any database holding those opsim output tables.).

        dbAddress = sqlalchemy connection string to database
        dbTables = dictionary of names of tables in the code : [names of tables in the database, names of primary keys]
        Note that for the dbTables there are defaults in the init --
          you can override (specific key:value pairs only if desired) by passing a dictionary in dbTables.
        """
        self.dbAddress = dbAddress
        # Default dbTables and dbTableIDKey values:
        defaultdbTables={'outputTable':['Output', 'obsHistID'],
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
                         'sequenceHistoryMissedHistoryTable':['SeqHistory_MissedHistory', 'seqhistory_missedHistID'],
                         'slewActivitiesTable':['SlewActivities', 'slewActivityID'],
                         'slewHistoryTable':['SlewHistory', 'slewID'],
                         'slewMaxSpeedsTable':['SlewMaxSpeeds', 'slewMaxSpeedID'],
                         'slewStateTable':['SlewState', 'slewIniStatID']
                         }
        # Call base init method to set up all tables and place default values into dbTable/dbTablesIdKey if not overriden.
        super(OpsimDatabase, self).__init__(dbAddress, dbTables=dbTables,
                                            defaultdbTables=defaultdbTables, 
                                            *args, **kwargs)
        self.filterlist = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        
            
    def fetchMetricData(self, colnames, sqlconstraint, distinctExpMJD=True, groupBy=None):
        """Fetch 'colnames' from 'Output' table. 

        colnames = the columns to fetch from the table.
        sqlconstraint = sql constraint to apply to data (minus "WHERE").
        distinctExpMJD = group by expMJD to get unique observations only (default True).
        groupBy = group by col 'groupBy' (will override group by expMJD)."""
        # To fetch data for a particular proposal only, add 'propID=[proposalID number]' as constraint,
        #  and to fetch data for a particular filter only, add 'filter ="[filtername]"' as a constraint. 
        table = self.tables['outputTable']
        if groupBy is not None:
            if distinctExpMJD:
                warnings.warn('Cannot group by more than one column. Using explicit groupBy col %s' %(groupBy))
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                   constraint = sqlconstraint,
                                                   colnames = colnames, groupBy = groupBy)
        elif distinctExpMJD:
            metricdata = table.query_columns_Array(chunk_size = self.chunksize, 
                                                    constraint = sqlconstraint,
                                                    colnames = colnames, 
                                                    groupByCol = 'expMJD')
        else:
            metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                                   constraint = sqlconstraint,
                                                   colnames = colnames)
        return metricdata


    def fetchFieldsFromOutputTable(self, sqlconstraint, raColName='fieldID', decColName='fieldDec'):
        """Fetch field information (fieldID/RA/Dec) from Output table."""
        # Fetch field info from the Output table, by selecting unique fieldID + ra/dec values.
        # This implicitly only selects fields which were actually observed by opsim.
        table = self.tables['outputTable']
        fielddata = table.query_columns_Array(constraint=sqlconstraint,
                                              colnames=['fieldID', raColName, decColName],
                                              groupByCol='fieldID')
        return fielddata


    def fetchFieldsFromFieldTable(self, propID=None, degreesToRadians=True):
        """Fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.
    
        proposalID = the proposal ID (default None), if selecting particular proposal - can be a list
        degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians) """
        # Note that you can't select any other sql constraints (such as filter). 
        # This will select fields which were requested by a particular proposal or proposals,
        #   even if they didn't get any observations. 
        tableName = 'fieldTable'
        if propID != None:
            query = 'select f.fieldID, f.fieldRA, f.fieldDec from %s as f' %(self.dbTables['fieldTable'][0])
            if propID != None:
                query += ', %s as p where (p.Field_fieldID = f.fieldID) ' %(self.dbTables['proposalFieldTable'][0])
                if hasattr(propID, '__iter__'): # list of propIDs
                    query += ' and ('
                    for pID in propID:
                        query += '(p.Proposal_propID = %d) or ' %(int(pID))
                    # Remove the trailing 'or' and add a closing parenthesis.
                    query = query[:-3]
                    query += ')'
                else: # single proposal ID.
                    query += ' and (p.Proposal_propID = %d) ' %(int(propID))
            fielddata = self.queryDatabase(tableName, query)
        else:
            table = self.tables[tableName]
            fielddata = table.query_columns_Array(colnames=['fieldID', 'fieldRA', 'fieldDec'],
                                                  groupByCol = 'fieldID')
        if degreesToRadians:
            fielddata['fieldRA'] = fielddata['fieldRA'] * np.pi / 180.
            fielddata['fieldDec'] = fielddata['fieldDec'] * np.pi / 180.
        return fielddata

    def fetchPropIDs(self):
        """Fetch the proposal IDs from the full opsim run database.
        Return the full list of ID numbers as well as a list of
         WFD propIDs (proposals containing 'Universal' in the name),
         deep drilling propIDs (proposals containing 'deep', 'Deep', 'dd' or 'DD' in the name)."""
        # The methods to identify WFD and DD proposals will be updated in the future,
        #  when opsim adds flags to the config tables.
        table = self.tables['proposalTable']
        propData = table.query_columns_Array(colnames=['propID', 'propConf', 'propName'], constraint='')
        propIDs = list(propData['propID'])
        wfdIDs = []
        ddIDs = []
        # Parse on name for now.
        for name, propid in zip(propData['propConf'],propData['propID']):
            if 'Universal' in name:
                wfdIDs.append(propid)
            if ('deep' in name) or ('Deep' in name) or ('DD' in name) or ('dd' in name):
                ddIDs.append(propid)
        return propIDs, wfdIDs, ddIDs

    def fetchRunLength(self, runLengthParam='nRun'):
        """Fetch the run length for a particular opsim run.

        runLengthParam = the 'paramName' in the config table identifying the run length (default nRun)."""
        table = self.tables['configTable']
        runLength = table.query_columns_Array(colnames=['paramValue'], constraint=" paramName = '%s'"%runLengthParam)
        runLength = float(runLength['paramValue'][0]) # Years
        return runLength

    def fetchSeeingColName(self):
        """Check whether the seeing column is 'seeing' or 'finSeeing' (v2.x simulator vs v3.0 simulator)."""
        # Really this is just a bit of a hack to see whether we should be using seeing or finseeing.
        # With time, this should probably just go away.
        table = self.tables['outputTable']
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
        results = table.query_columns_Array(colnames = ['version', 'sessionDate', 'runComment'])
        configSummary['Version']['OpsimVersion'] = '%s'  %(results['version'][0]) + \
            '  RunDate %s' %(results['sessionDate'][0])
        configSummary['RunInfo'] = {}        
        configSummary['RunInfo']['RunComment'] = results['runComment']
        configSummary['RunInfo']['RunName'] = self.fetchOpsimRunName()
        # Pull out a few special values to put into summary.
        table = self.tables['configTable']
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
        query = 'select propID, propConf, propName from Proposal group by propID'
        propdata = self.queryDatabase('proposalTable', query)
        # Make 'nice' proposal names
        propnames = np.array([os.path.split(x)[1].replace('.conf', '') for x in propdata['propConf']])
        # Get 'nice' module names
        moduledata = table.query_columns_Array(colnames=['moduleName',], constraint='nonPropID=0')
        modulenames = np.array([os.path.split(x)[1].replace('.conf', '') for x in moduledata['moduleName']])
        # Grab the config information for each proposal and module.
        cols = ['paramName', 'paramValue', 'comment']
        for longmodname, modname in zip(moduledata['moduleName'], modulenames):
            config[modname] = table.query_columns_Array(colnames=cols, constraint='moduleName="%s"' %(longmodname))
            config[modname] = config[modname][['paramName', 'paramValue', 'comment']]
        for propid, propname in zip(propdata['propID'], propnames):
            config[propname] = table.query_columns_Array(colnames=cols,
                                                         constraint='nonPropID="%s" and paramName!="userRegion"' %(propid))
            config[propname] = config[propname][['paramName', 'paramValue', 'comment']]
        # Now finish building the summary to add proposal information.
        def _matchParamNameValue(configarray, keyword):
            return configarray['paramValue'][np.where(configarray['paramName']==keyword)]
        # Loop through all proposals to add summary information.
        configSummary['Proposals'] = {}
        propidorder = sorted(propdata['propID'])
        # Generate a keyorder to print proposals in order of propid.
        configSummary['Proposals']['keyorder'] = []
        for propid in propidorder:            
            configSummary['Proposals']['keyorder'].append(propnames[np.where(propdata['propID'] == propid)][0])
        for propid, propname in zip(propdata['propID'], propnames):
            configSummary['Proposals'][propname] = {}            
            thisdict = configSummary['Proposals'][propname]
            thisdict['keyorder'] = ['PropID', 'PropName',  'PropType', 'RelPriority', 'NumUserRegions', 'NumFields']
            thisdict['PropName'] = propname
            thisdict['PropID'] = propid
            thisdict['PropType'] = propdata['propName'][np.where(propnames == propname)]
            thisdict['RelPriority'] = _matchParamNameValue(config[propname], 'RelativeProposalPriority')
            # Get the number of user regions.
            constraint = 'nonPropID="%s" and paramName="userRegion"' %(propid)
            result = table.query_columns_Array(colnames=['paramName',], constraint=constraint)            
            thisdict['NumUserRegions'] = result.size
            # Get the number of fields requested in the proposal (all filters). 
            thisdict['NumFields'] = self.fetchFieldsFromFieldTable(propID=propid).size
            # Find number of visits requested per filter for the proposal, along with min/max sky and airmass values.
            # Note that config table has multiple entries for Filter/Filter_Visits/etc. with the same name.
            #   The order of these entries in the config array matters. 
            thisdict['PerFilter'] = {}
            for key, keyword in zip(['Filters', 'MaxSeeing', 'MinSky', 'MaxSky'],
                                    ['Filter', 'Filter_MaxSeeing', 'Filter_MinBrig', 'Filter_MaxBrig']):
                temp = _matchParamNameValue(config[propname], keyword)
                if len(temp) > 0:
                    thisdict['PerFilter'][key] = temp
            # And count how many total exposures are requested per filter.
            numVisits = np.array(_matchParamNameValue(config[propname], 'Filter_Visits'), int)
            if len(numVisits) > 0:
                # This is a weak lensing type, simple request of visit numbers per filter.
                thisdict['PerFilter']['NumVisits'] = numVisits
            else:
                # This is one of the other types of proposals and we must look at subsequences.
                numVisits = np.zeros(len(thisdict['PerFilter']['Filters']), int)
                allsubevents = np.array(_matchParamNameValue(config[propname], 'SubSeqEvents'), int)
                allsubexposures = _matchParamNameValue(config[propname], 'SubSeqExposures')
                allsubfilters = np.array(_matchParamNameValue(config[propname], 'SubSeqFilters'), str)
                for subevents, subexposures, subfilters in zip(allsubevents, allsubexposures, allsubfilters):
                    # If non multi-filter subsequence (i.e. just one filter per subseq):
                    if len(subfilters) < 2:
                        # Match this filter to list of filters.
                        idx = (thisdict['PerFilter']['Filters'] == subfilters)
                        numVisits[idx] += int(subevents) * int(subexposures)
                    # Else we may have multiple filters in this subsequence, so split apart. 
                    else: 
                        splitsubfilters = subfilters.split(',')
                        splitsubexposures = subexposures.split(',')
                        for f, exp in zip(splitsubfilters, splitsubexposures):
                            idx = (thisdict['PerFilter']['Filters'] == f)
                            numVisits[idx] += int(subevents) * int(exp)
                thisdict['PerFilter']['NumVisits'] = numVisits
            # Sort the filter information so it's ugrizy instead of order in opsim config db.
            idx = []
            for f in self.filterlist:
                filterpos = np.where(thisdict['PerFilter']['Filters'] == f)
                if len(filterpos[0]) > 0:
                    idx.append(filterpos[0][0])
            idx = np.array(idx, int)
            for k in thisdict['PerFilter']:
                thisdict['PerFilter'][k] = thisdict['PerFilter'][k][idx]
        return configSummary, config
