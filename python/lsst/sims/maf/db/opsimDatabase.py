from .table import Table

class OpsimDatabase(Database):
    def __init__(self, dbAddress, dbTables=None, dbTablesIdKey=None,
                 defaultdbTables=None, defaultdbTablesIdKey=None, **kwargs):
        """Instantiate object to handle queries of the opsim database.
        (In general these will be the sqlite database files produced by opsim, but could
        be any database holding those opsim output tables.).

        dbAddress = sqlalchemy connection string to database
        dbTables = dictionary of (names of tables in the code) : (names of tables in the database)
        dbTableIDKey = dictionary of (names of tables in the code) : (primary key column name)
        Note that for the dbTables and dbTableIDKey there are defaults in the init --
          you can override (specific key:value pairs only if desired) by passing a dictionary
        """
        self.dbAddress = dbAddress
        # Default dbTables and dbTableIDKey values:
        defaultdbTables={'outputTabel':'Output',
                  'cloudTable':'Cloud',
                  'seeingTable':'Seeing',
                  'fieldTable':'Field',
                  'sessionTable':'Session',
                  'configTable':'Config',
                  'proposalTable':'Proposal',
                  'proposalFieldTable':'Proposal_Field',
                  'obsHistoryTable':'ObsHistory',
                  'obsHistoryProposalTable':'Obshistory_Proposal',
                  'sequenceHistoryTable':'SeqHistory',
                  'sequenceHistoryObsHistoryTable':'SeqHistory_ObsHistory',
                  'missedHistoryTable':'MissedHistory',
                  'sequenceHistoryMissedHistoryTable':'SeqHistory_MissedHistory',
                  'slewActivitiesTable':'SlewActivities',
                  'slewHistoryTable':'SlewHistory',
                  'slewMaxSpeedsTable':'SlewMaxSpeeds',
                  'slewStateTable':'SlewState'
                  }
        defaultdbTablesIdKey = {'outputTable':'obsHistID',
                        'cloudTable':'cloudID',
                        'seeingTable':'seeingID',
                        'fieldTable':'fieldID',
                        'sessionTable':'sessionID',
                        'configTabel':'configID',
                        'proposalTable':'propID',
                        'proposalFieldTable':'proposal_field_id',
                        'obsHistoryTable':'obsHistID',
                        'obsHistoryProposalTable':'obsHistory_propID',
                        'sequenceHistoryTable':'sequenceID',
                        'sequenceHistoryObsHistoryTable':'seqhitsory_obsHistID',
                        'sequenceHistoryMissedHistoryTable':'seqhistory_missedHistID',
                        'slewActivitiesTable':'slewActivityID',
                        'slewHistoryTable':'slewID',
                        'slewMaxSpeedsTable':'slewMaxSpeedID',
                        'slewStateTable':'slewIniStatID'
                        }
        super(OpsimDatabase, self).__init__(dbTables=dbTables, dbTablesIdKey=dbTablesIdKey,
                                            defaultdbTables=defaultdbTables, defaultdbTablesIdKey=defaultdbTablesIdKey,
                                            **kwargs)        
            
    def fetchMetricData(self, colnames, sqlconstraint, distinctExpMJD=True):
        """Fetch 'colnames' from 'Output' table. 

        colnames = the columns to fetch from the table.
        sqlconstraint = sql constraint to apply to data (minus "WHERE").
        distinctExpMJD = group by expMJD to get unique observations only (default True)."""
        # To fetch data for a particular proposal only, add 'propID=[proposalID number]' as constraint,
        #  and to fetch data for a particular filter only, add 'filter ="[filtername]"' as a constraint. 
        table = self.tables['outputTable']
        if distinctExpMJD:
            metricdata = table.query_columns_Array(chunk_size=self.chunksize, 
                                                    constraint = sqlconstraint,
                                                    colnames = colnames, 
                                                    groupByCol = 'expMJD')
        else:
            simdata = table.query_columns_Array(chunk_size=self.chunksize, 
                                                constraint = sqlconstraint,
                                                colnames = colnames)
        return simdata


    def fetchFieldsFromOutputTable(self, sqlconstraint, raColName='fieldID', decColName='fieldDec'):
        """Fetch field information (fieldID/RA/Dec) from Output table."""
        # Fetch field info from the Output table, by selecting unique fieldID + ra/dec values.
        # This implicitly only selects fields which were actually observed by opsim.
        table = self.tables['outputTable']
        fielddata = table.query_columns_Array(constraint=sqlconstraint,
                                              colnames=['fieldID', raColName, decColName],
                                              groupByCol='fieldID')
        return fielddata


    def fetchFieldsFromFieldTable(self, proposalID=None, degreesToRadians=True):
        """Fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.
    
        proposalID = the proposal ID (default None), if selecting particular proposal - can be a list
        degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians) """
        # Note that you can't select any other sql constraints (such as filter). 
        # This will select fields which were requested by a particular proposal or proposals,
        #   even if they didn't get any observations. 
        table = self.tables['fieldTable']
        if proposalID != None:
            query = 'select f.fieldID, f.fieldRA, f.fieldDec from %s as f' %(self.dbTables['fieldTable'])
            if proposalID != None:
                query += ', %s as p where (p.Field_fieldID = f.fieldID) ' %(self.dbTables['proposalFieldTable'])
                if hasattr(proposalID, '__iter__'): # list of propIDs
                    query += ' and ('
                    for propID in proposalID:
                        query += '(p.Proposal_propID = %d) or ' %(int(propID))
                    # Remove the trailing 'or' and add a closing parenthesis.
                    query = query[:-3]
                    query += ')'
                else: # single proposal ID.
                    query += ' and (p.Proposal_propID = %d) ' %(int(proposalID))
            results = table.engine.execute(query)
            fielddata = table._postprocess_results(results.fetchall())
        else:
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

    def fetchConfigs(self):
        """Fetch config data from configTable, match proposal IDs with proposal names and some field data,
        and do a little manipulation of the data to make it easier to add to the presentation layer.
    
        Returns dictionary keyed by proposals and module names, and within each of these is another dictionary
        containing the paramNames and paramValues relevant for that module or proposal.
        """
        # Convenience functions: defined here to make it easier to modify if necessary.
        def _ModulePropID2LongName(moduleName, propID):
            return '__'.join([moduleName, str(propID)])
        def _LongName2ModulePropID(longName):
            moduleName = longName.split('__')[0]
            propID = int(longName.split('__')[1])
            return moduleName, propID
        # Get config table data.
        table = self.tables['configTable']
        # If opsim adds descriptions to the 'comment' variable, grab that here too and use as 'description' in outputs.
        cols = ['moduleName', 'paramName', 'paramValue', 'nonPropID']
        configdata = table.query_columns_Array(colnames=cols)
        # Get proposal table data.
        table = self.tables['proposalTable']
        cols = ['propID', 'propConf', 'propName']
        propdata = table.query_columns_Array(colnames=cols)
        # Get counts of fields from proposal_field data.
        table = self.tables['proposalFieldTable']
        cols = ['proposal_field_id', 'Proposal_propID']
        propfielddata = table.query_columns_Array(colnames=cols)    
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
