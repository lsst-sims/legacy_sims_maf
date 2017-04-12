import os
import numpy as np
import warnings
from .database import Database
from lsst.sims.utils import Site
from lsst.sims.maf.utils import getDateVersion

__all__ = ['OpsimDatabase']


class OpsimDatabase(Database):
    """
    Database to class to interact with v4 versions of the opsim outputs.

    Parameters
    ----------
    database : str
        Name of the database or sqlite filename.
    driver : str, opt
        Name of the dialect + driver for sqlalchemy. Default 'sqlite'.
    host : str, opt
        Name of the database host. Default None (appropriate for sqlite files).
    port : str, opt
        String port number for the database. Default None (appropriate for sqlite files).
    dbTables : dict, opt
        Dictionary of the names of the tables in the database.
        The dict should be key = table name, value = [table name, primary key].
    """
    def __init__(self, database, driver='sqlite', host=None, port=None, dbTables=None,
                 *args, **kwargs):
        # Default dbTables and dbTableIDKey values:
        if 'defaultdbTables' in kwargs:
            defaultdbTables = kwargs.get('defaultdbTables')
            # Remove this kwarg since we're sending it on explicitly
            del kwargs['defaultdbTables']
        else:
            defaultdbTables = {'SummaryAllProps': ['SummaryAllProps', 'observationId'],
                                 'Config': ['Config', 'configId'],
                                 'Field': ['Field', 'fieldId'],
                                 'ObsExposures': ['ObsExposures', 'exposureId'],
                                 'ObsHistory': ['ObsHistory', 'observationId'],
                                 'ObsProposalHistory': ['ObsProposalHistory', 'propHistId'],
                                 'Proposal': ['Proposal', 'propId'],
                                 'ScheduledDowntime': ['ScheduledDowntime', 'night'],
                                 'Session': ['Session', 'sessionId'],
                                 'SlewActivities': ['SlewActivities', 'slewActivityId'],
                                 'SlewFinalState': ['SlewFinalState', 'slewStateId'],
                                 'SlewHistory': ['SlewHistory', 'slewCount'],
                                 'SlewInitialState': ['SlewInitialState', 'slewStateId'],
                                 'SlewMaxSpeeds': ['SlewMaxSpeeds', 'slewMaxSpeedId'],
                                 'TargetExposures': ['TargetExposures', 'exposureId'],
                                 'TargetHistory': ['TargetHistory', 'targetId'],
                                 'TargetProposalHistory': ['TargetProposalHistory', 'propHistId'],
                                 'UnscheduledDowntime': ['UnscheduledDowntime', 'night']}
        # Call base init method to set up all tables and place default values
        # into dbTable/dbTablesIdKey if not overriden.
        super(OpsimDatabase, self).__init__(driver=driver, database=database, host=host, port=port,
                                            dbTables=dbTables,
                                            defaultdbTables=defaultdbTables,
                                            *args, **kwargs)
        # Save filterlist so that we get the filter info per proposal in this desired order.
        self.filterlist = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        self.summaryTable = 'SummaryAllProps'
        # Set internal variables for column names.
        self._colNames()

    def _colNames(self):
        """
        Set variables to represent the common column names used in this class directly.

        This should make future schema changes a little easier to handle.
        It is NOT meant to function as a general column map, just to abstract values
        which are used *within this class*.
        """
        self.mjdCol = 'observationStartMJD'
        self.slewID = 'slewActivityId'
        self.delayCol = 'activityDelay'
        
        self.fieldIdCol = 'fieldId'
        self.raCol = 'fieldRA'
        self.decCol = 'fieldDec'
        self.propIdCol = 'propId'
        self.propNameCol = 'propName'
        self.propTypeCol = 'propType'
        # For config parsing.
        self.versionCol = 'version'
        self.sessionDateCol = 'sessionDate'
        self.runCommentCol = 'runComment'

    def fetchMetricData(self, colnames, sqlconstraint=None, groupBy='default', tableName='SummaryAllProps'):
        """
        Fetch 'colnames' from 'tableName'.

        Parameters
        ----------
        colnames : list
            The columns to fetch from the table.
        sqlconstraint : str, opt
            The sql constraint to apply to the data (minus "WHERE"). Default None.
            Examples: to fetch data for the r band filter only, set sqlconstraint to 'filter = "r"'.
        groupBy : str, opt
            The column to group the returned data by.
            Default (when using summaryTable) is the MJD, otherwise will be None.
        tableName : str, opt
            The table to query. The default is the summary table, detailed name is set by self.summaryTable.

        Returns
        -------
        np.recarray
            A structured array containing the data queried from the database.
        """
        if tableName not in self.dbTables:
            raise ValueError('Table %s not recognized; not in list of database tables.' % (tableName))

        if groupBy == 'default' and (tableName == self.summaryTable):
            groupBy = self.mjdCol

        if tableName == 'SummaryAllProps':
            table = self.tables[self.summaryTable]
        else:
            table = self.tables[tableName]

        metricdata = table.query_columns_Array(chunk_size = self.chunksize,
                                               constraint = sqlconstraint,
                                                colnames = colnames, groupByCol = groupBy)
        return metricdata

    def fetchFieldsFromSummaryTable(self, sqlconstraint=None, raColName=None, decColName=None):
        """
        Fetch field information (fieldID/RA/Dec) from the summary table.

        This implicitly only selects fields which were actually observed by opsim.

        Parameters
        ----------
        sqlconstraint : str, opt
            Sqlconstraint to apply before selecting observations to use for RA/Dec. Default None.
        raColName : str, opt
            Name of the RA column in the database.
        decColName
            Name of the Dec column in the database.

        Returns
        -------
        np.recarray
            Structured array containing the field data (fieldID, fieldRA, fieldDec). RA/Dec in degrees.
        """
        if raColName is None:
            raColName = self.raCol
        if decColName is None:
            decColName = self.decCol
        table = self.tables[self.summaryTable]
        fielddata = table.query_columns_Array(constraint=sqlconstraint,
                                              colnames=[self.fieldIdCol, raColName, decColName],
                                              groupByCol=self.fieldIdCol)
        return fielddata

    def fetchFieldsFromFieldTable(self, propId=None, degreesToRadians=False):
        """
        Fetch field information (fieldID/RA/Dec) from the Field table.

        This selects all fields possible to observe with opsim.
        ** Need to add capability to select only fields associated with a given proposal. **

        Parameters
        ----------
        propId : int or list of ints
            Proposal ID or list of proposal IDs to use to select fields.
            Deprecated with v4 currently.
        degreesToRadians : bool, opt
            If True, convert degrees in Field table into radians.

        Returns
        -------
        np.recarray
            Structured array containing the field data (fieldID, fieldRA, fieldDec).
        """
        if propId is not None:
            warnings.warn('Cannot select field IDs associated only with proposals at present.'
                          'Selecting all fields.')
        tableName = 'Field'
        table = self.tables[tableName]
        fielddata = table.query_columns_Array(colnames=[self.fieldIdCol, 'ra', 'dec'],
                                              groupByCol = self.fieldIdCol)
        if degreesToRadians:
            fielddata['ra'] = np.radians(fielddata['ra'])
            fielddata['dec'] = np.radians(fielddata['dec'])
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
        propTags = {'WFD': [], 'DD': []}

        table = self.tables['Proposal']
        propData = table.query_columns_Array(colnames=[self.propIdCol, self.propNameCol], constraint='')
        for propID, propName in zip(propData[self.propIdCol], propData[self.propNameCol]):
            # Fix these in the future, to use the proper tags that will be added to output database.
            propIDs[propID] = propName
            if 'widefastdeep' in propName.lower():
                propTags['WFD'].append(propID)
            if 'drill' in propName.lower():
                propTags['DD'].append(propID)
        return propIDs, propTags

    def fetchRunLength(self, runLengthParam='survey/duration'):
        """Find the survey duration for a particular opsim run (years).

        Parameters
        ----------
        runLengthParam : str, opt
            The paramName value in the config table identifying the run length.
            Default 'survey/duration'.

        Returns
        -------
        float
        """
        if 'Config' not in self.tables:
            print 'Cannot access Config table to retrieve runLength; using default 10 years'
            runLength = 10.0
        else:
            table = self.tables['Config']
            runLength = table.query_columns_Array(colnames=['paramValue'],
                                                  constraint=" paramName = '%s'" % (runLengthParam))
            runLength = float(runLength['paramValue'][0])  # Years
        return runLength

    def fetchLatLonHeight(self):
        """
        Returns the latitude, longitude, and height of the telescope used by the config file.
        """
        if 'Config' not in self.tables:
            print 'Cannot access Config table to retrieve site parameters; using sims.utils.Site instead.'
            site = Site(name='LSST')
            lat = site.latitude_rad
            lon = site.longitude_rad
            height = site.elev
        else:
            pre = 'observing_site/'
            table = self.tables['Config']
            lat = table.query_columns_Array(colnames=['paramValue'],
                                            constraint="paramName = '%slatitude'" % pre)
            lat = float(lat['paramValue'][0])
            lon = table.query_columns_Array(colnames=['paramValue'],
                                            constraint="paramName = '%slongitude'" % pre)
            lon = float(lon['paramValue'][0])
            height = table.query_columns_Array(colnames=['paramValue'],
                                               constraint="paramName = '%sheight'" % pre)
            height = float(height['paramValue'][0])
        return lat, lon, height

    def fetchNVisits(self, propID=None):
        """Return the total number of visits in the simulation or for a particular proposal.

        Parameters
        ----------
        propID : int or list of ints, opt
            The proposal id, can be a list. Default None.

        Returns
        -------
        int
            The number of visits achieved in a simulation or for a particular proposal(s).
        """
        tableName = self.summaryTable
        query = 'select count(distinct(%s)) from %s' % (self.mjdCol, self.dbTables[tableName][0])
        if propID is not None:
            query += ' where '
            if isinstance(propID, int):
                query += ' proposalId=%d' % (propID)
            else:
                for pId in propID:
                    query += 'proposalId=%d or ' %(int(pId))
                query = query[:-3]
        data = self.tables[tableName].execute_arbitrary(query)
        return int(data[0][0])

    def fetchOpsimRunName(self):
        """Find opsim run name (machine name + sessionID) from Session table.
        """
        if 'Session' not in self.tables:
            print 'Could not access Session table to find this information.'
            runName = 'OpsimV4'
        else:
            table = self.tables['Session']
            res = table.query_columns_Array(colnames=['sessionId', 'sessionHost'])
            runName = str(res['sessionHost'][0]) + '_' + str(res['sessionId'][0])
        return runName

    def fetchTotalSlewN(self):
        """Find the total slew time.
        """
        if 'SlewActivities' not in self.tables:
            print 'Could not access SlewActivities table to find this information.'
            nslew = -1
        else:
            table = self.tables['SlewActivities']
            query = 'select count(distinct(%s)) from slewActivities where %s >0' % (self.slewID,
                                                                                    self.delayCol)
            res = table.execute_arbitrary(query)
            nslew = int(res[0][0])
        return nslew

    def fetchRequestedNvisits(self, propId=None):
        """Find the requested number of visits for the simulation or proposal(s).

        Parameters
        ----------
        propId : int or list of ints, opt

        Returns
        -------
        dict
            Number of visits in u/g/r/i/z/y.
        """
        visitDict = {}
        if propId is None:
            constraint = ''
        else:
            if isinstance(propId, int):
                constraint = 'propId = %d' % (propId)
            else:
                constraint = ''
                for pId in propId:
                    constraint += 'propId = %d or ' % (int(pId))
                constraint = constraint[:-3]
        propData = self.tables['Proposal'].query_columns_Array(colnames=[self.propNameCol,
                                                                         self.propTypeCol],
                                                               constraint=constraint)
        nvisits = {}
        for f in self.filterlist:
            nvisits[f] = 0
        for pName, propType in zip(propData[self.propNameCol], propData[self.propTypeCol]):
            if propType.lower() == 'general':
                for f in self.filterlist:
                    constraint = 'paramName="science/general_props/values/%s/filters/%s/num_visits"' \
                                 % (pName, f)
                    val = self.tables['Config'].query_columns_Array(colnames=['paramValue'],
                                                                    constraint=constraint)
                    if len(val) > 0:
                        nvisits[f] += int(val['paramValue'][0])
            elif propType.lower == 'sequence':
                pass
                # Not clear yet.
        return nvisits

    def _matchParamNameValue(self, configarray, keyword):
        return configarray['paramValue'][np.where(configarray['paramName'] == keyword)]

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
        configSummary['Version']['MAFVersion'] = '%s' %(mafversion['__version__'])
        configSummary['Version']['MAFDate'] = '%s' %(mafdate)
        # Opsim date, version and runcomment info from session table
        table = self.tables['Session']
        query = 'select %s, %s, %s from Session' % (self.versionCol, self.sessionDateCol,
                                                    self.runCommentCol)
        results = table.execute_arbitrary(query)
        configSummary['Version']['OpsimVersion'] = '%s'  %(results[self.versionCol][0])
        configSummary['Version']['OpsimDate'] = '%s' %(results[self.sessionDateCol][0])
        configSummary['RunInfo'] = {}
        configSummary['RunInfo']['RunComment'] = results[self.runCommentCol][0]
        configSummary['RunInfo']['RunName'] = self.fetchOpsimRunName()
        # Pull out a few special values to put into summary.
        table = self.tables['Config']
        # This section has a number of configuration parameter names hard-coded.
        # I've left these here (rather than adding to self_colNames), because I think schema changes will in the config
        # files will actually be easier to track here (at least until the opsim configs are cleaned up).
        constraint = 'paramName="observatory/telescope/altitude_minpos"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['MinAlt'] = results['paramValue'][0]
        constraint = 'paramName="observatory/telescope/altitude_maxpos"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['MaxAlt'] = results['paramValue'][0]
        constraint = 'paramName="observatory/camera/filter_change_time"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['TimeFilterChange'] = results['paramValue'][0]
        constraint = 'paramName="observatory/camera/readout_time"'
        results = table.query_columns_Array(colnames=['paramValue', ], constraint=constraint)
        configSummary['RunInfo']['TimeReadout'] = results['paramValue'][0]
        configSummary['RunInfo']['keyorder'] = ['RunName', 'RunComment', 'MinAlt', 'MaxAlt',
                                                'TimeFilterChange', 'TimeReadout']

        # Echo config table into configDetails.
        configDetails = {}
        configs = table.query_columns_Array(['paramName', 'paramValue'])
        for name, value in zip(configs['paramName'], configs['paramValue']):
            configDetails[name] = value

        # Now finish building the summary to add proposal information.
        # Loop through all proposals to add summary information.
        propData = self.tables['Proposal'].query_columns_Array([self.propIdCol, self.propNameCol,
                                                                self.propTypeCol])
        configSummary['Proposals'] = {}
        for propid, propname, proptype in zip(propData[self.propIdCol],
                                              propData[self.propNameCol], propData[self.propTypeCol]):
            configSummary['Proposals'][propname] = {}
            propdict = configSummary['Proposals'][propname]
            propdict['keyorder'] = [self.propIdCol, self.propNameCol, self.propTypeCol]
            propdict['PropName'] = propname
            propdict['PropId'] = propid
            propdict['PropType'] = proptype
            # Find number of visits requested per filter for the proposal
            # along with min/max sky and airmass values.
            propdict['Filters'] = {}
            for f in self.filterlist:
                propdict['Filters'][f] = {}
                propdict['Filters'][f]['Filter'] = f
                dictkeys = ['MaxSeeing', 'BrightLimit', 'DarkLimit', 'NumVisits', 'GroupedVisits', 'Snaps']
                querykeys = ['max_seeing', 'bright_limit', 'dark_limit', 'num_visits',
                             'num_grouped_visits', 'exposures']
                for dk, qk in zip(dictkeys, querykeys):
                    constraint = 'paramName like "science/%s_props/values/%s/filters/%s/%s"' \
                                 % ("%", propname, f, qk)
                    results = table.query_columns_Array(['paramValue'], constraint=constraint)
                    if len(results) == 0:
                        propdict['Filters'][f][dk] = '--'
                    else:
                        propdict['Filters'][f][dk] = results['paramValue'][0]
                propdict['Filters'][f]['keyorder'] = ['Filter', 'MaxSeeing', 'MinSky', 'MaxSky',
                                                      'NumVisits', 'GroupedVisits', 'Snaps']
            propdict['Filters']['keyorder'] = list(self.filterlist)
        return configSummary, configDetails


