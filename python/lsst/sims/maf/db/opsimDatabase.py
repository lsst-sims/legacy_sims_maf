from __future__ import print_function
from builtins import str
from builtins import zip
import os, re
import numpy as np
import warnings
from .database import Database
from lsst.sims.utils import Site
from lsst.sims.maf.utils import getDateVersion

__all__ = ['testOpsimVersion', 'OpsimDatabase', 'OpsimDatabaseV4', 'OpsimDatabaseV3']

def testOpsimVersion(database, driver='sqlite', host=None, port=None):
    opsdb = Database(database, driver=driver, host=host, port=port)
    if 'SummaryAllProps' in opsdb.tableNames:
        version = "V4"
    elif 'Summary' in opsdb.tableNames:
        version = "V3"
    else:
        version = "Unknown"
    opsdb.close()
    return version

def OpsimDatabase(database, driver='sqlite', host=None, port=None, defaultTable=None,
                  longstrings=False, verbose=False):
    """Convenience method to return an appropriate OpsimDatabaseV3/V4 version.

    This is here for backwards compatibility, as 'opsdb = db.OpsimDatabase(dbFile)' will
    work as naively expected. However note that OpsimDatabase itself is no longer a class, but
    a simple method that will attempt to instantiate the correct type of OpsimDatabaseV3 or OpsimDatabaseV4.
    """
    version = testOpsimVersion(database)
    if version == 'V4':
        opsdb = OpsimDatabaseV4(database, driver=driver, host=host, port=port,
                                defaultTable=defaultTable, longstrings=longstrings, verbose=verbose)
    elif version == 'V3':
        opsdb =  OpsimDatabaseV3(database, driver=driver, host=host, port=port,
                                 defaultTable=defeaultTable, longstrings=longstrings, verbose=verbose)
    else:
        warnings.warn('Could not identify opsim database version; just using Database class instead')
        opsdb = Database(database, driver=driver, host=host, port=port,
                         defaultTable=defaultTable, longstrings=longstrings, verbose=verbose)
    return opsdb


class BaseOpsimDatabase(Database):
    """Base opsim database class to gather common methods among different versions of the opsim schema.

    Not intended to be used directly; use OpsimDatabaseV3 or OpsimDatabaseV4 instead."""
    def __init__(self, database, driver='sqlite', host=None, port=None, defaultTable=None,
                 longstrings=False, verbose=False):
        super(BaseOpsimDatabase, self).__init__(database=database, driver=driver, host=host, port=port,
                                                defaultTable=defaultTable, longstrings=longstrings,
                                                verbose=verbose)
        # Save filterlist so that we get the filter info per proposal in this desired order.
        self.filterlist = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        self.defaultTable = defaultTable
        self._colNames()

    def _colNames(self):
        # Add version-specific column names in subclasses.
        pass

    def fetchMetricData(self, colnames, sqlconstraint=None, groupBy='default', tableName=None):
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
            The table to query. The default (None) will use the summary table, set by self.summaryTable.

        Returns
        -------
        np.recarray
            A structured array containing the data queried from the database.
        """
        if tableName is None:
            tableName = self.defaultTable
        if groupBy is 'default' and tableName==self.defaultTable:
            groupBy = self.mjdCol
        if groupBy is 'default' and tableName!=self.defaultTable:
            groupBy = None
        metricdata = super(BaseOpsimDatabase, self).fetchMetricData(colnames=colnames,
                                                                sqlconstraint=sqlconstraint,
                                                                groupBy=groupBy, tableName=tableName)
        return metricdata

    def fetchFieldsFromSummaryTable(self, sqlconstraint=None, raColName=None, decColName=None,
                                    degreesToRadians=False):
        """
        Fetch field information (fieldID/RA/Dec) from the summary table.

        This implicitly only selects fields which were actually observed by opsim.

        Parameters
        ----------
        sqlconstraint : str, opt
            Sqlconstraint to apply before selecting observations to use for RA/Dec. Default None.
        raColName : str, opt
            Name of the RA column in the database.
        decColName : str, opt
            Name of the Dec column in the database.
        degreesToRadians : bool, opt
            Convert ra/dec into degrees?
            If field information in summary table is in degrees, degreesToRadians should be True.

        Returns
        -------
        np.recarray
            Structured array containing the field data (fieldID, fieldRA, fieldDec). RA/Dec in degrees.
        """
        if raColName is None:
            raColName = self.raCol
        if decColName is None:
            decColName = self.decCol
        fielddata = self.query_columns(self.defaultTable,
                                       colnames=[self.fieldIdCol, raColName, decColName],
                                       sqlconstraint=sqlconstraint, groupBy=self.fieldIdCol)
        if degreesToRadians:
            fielddata[raColName] = np.radians(fielddata[raColName])
            fielddata[decColName] = np.radians(fielddata[decColName])
        return fielddata

    def fetchRunLength(self):
        """Find the survey duration for a particular opsim run (years).

        Returns
        -------
        float
        """
        if 'Config' not in self.tables:
            print('Cannot access Config table to retrieve runLength; using default 10 years')
            runLength = 10.0
        else:
            runLength = self.query_columns('Config', colnames=['paramValue'],
                                           sqlconstraint=" paramName = '%s'" % (self.runLengthParam))
            runLength = float(runLength['paramValue'][0])  # Years
        return runLength

    def fetchLatLonHeight(self):
        """Returns the latitude, longitude, and height of the telescope used by the config file.
        """
        if 'Config' not in self.tables:
            print('Cannot access Config table to retrieve site parameters; using sims.utils.Site instead.')
            site = Site(name='LSST')
            lat = site.latitude_rad
            lon = site.longitude_rad
            height = site.elev
        else:
            lat = self.query_columns('Config', colnames=['paramValue'],
                                     sqlconstraint="paramName like '%latitude%'")
            lat = float(lat['paramValue'][0])
            lon = self.query_columns('Config', colnames=['paramValue'],
                                     sqlconstraint="paramName like '%longitude%'")
            lon = float(lon['paramValue'][0])
            height = self.query_columns('Config', colnames=['paramValue'],
                                        sqlconstraint="paramName like '%height%'")
            height = float(height['paramValue'][0])
        return lat, lon, height

    def fetchOpsimRunName(self):
        """Return opsim run name (machine name + session ID) from Session table.
        """
        if 'Session' not in self.tables:
            print('Could not access Session table to find this information.')
            runName = self.defaultRunName
        else:
            res = self.query_columns('Session', colnames=[self.sessionIdCol,
                                                          self.sessionHostCol])
            runName = str(res[self.sessionHostCol][0]) + '_' + str(res[self.sessionIdCol][0])
        return runName

    def fetchNVisits(self, propId=None):
        """Returns the total number of visits in the simulation or visits for a particular proposal.

        Parameters
        ----------
        propId : int or list of ints
            The ID numbers of the proposal(s).

        Returns
        -------
        int
        """
        if 'ObsHistory' in self.tables and propId is None:
            query = 'select count(*) from ObsHistory'
            data = self.execute_arbitrary(query, dtype=([('nvisits', int)]))
        else:
            query = 'select count(distinct(%s)) from %s' %(self.mjdCol, self.defaultTable)
            if propId is not None:
                query += ' where '
                if hasattr(propId, '__iter__'):
                    for pID in propId:
                        query += 'propID=%d or ' %(int(pID))
                    query = query[:-3]
                else:
                    query += 'propID = %d' %(int(propId))
            data = self.execute_arbitrary(query, dtype=([('nvisits', int)]))
        return data['nvisits'][0]

    def fetchTotalSlewN(self):
        """Return the total number of slews.
        """
        if 'SlewActivities' not in self.tables:
            print('Could not access SlewActivities table to find this information.')
            nslew = -1
        else:
            query = 'select count(distinct(%s)) from SlewActivities where %s >0' % (self.slewId,
                                                                                    self.delayCol)
            res = self.execute_arbitrary(query, dtype=([('slewN', int)]))
            nslew = res['slewN'][0]
        return nslew

    def fetchSeeingColName(self):
        """
        Check whether the seeing column is 'seeing' or 'finSeeing' or 'FWHMeff' (v2.x simulator vs v3.0 /4.0)
        Returns the name of the seeing column.
        This should go away eventually.
        """
        # Really this is just a bit of a hack to see whether we should be using seeing or finseeing.
        # With time, this should probably just go away.
        possibleSeeingColumns = ['seeing', 'finSeeing', 'FWHMeff', 'seeingFwhmEff']
        seeingcol = None
        for x in possibleSeeingColumns:
            try:
                self.query_columns(self.defaultTable, colnames=[x], numLimit=1)
                seeingcol = x
            except ValueError:
                pass
        return seeingcol


class OpsimDatabaseV4(BaseOpsimDatabase):
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
    def __init__(self, database, driver='sqlite', host=None, port=None, defaultTable='SummaryAllProps',
                 longstrings=False, verbose=False):
        super(OpsimDatabaseV4, self).__init__(database=database, driver=driver, host=host, port=port,
                                              defaultTable=defaultTable, longstrings=longstrings,
                                              verbose=verbose)

    def _colNames(self):
        """
        Set variables to represent the common column names used in this class directly.

        This should make future schema changes a little easier to handle.
        It is NOT meant to function as a general column map, just to abstract values
        which are used *within this class*.
        """
        self.mjdCol = 'observationStartMJD'
        self.slewId = 'slewHistory_slewCount'
        self.delayCol = 'activityDelay'
        self.fieldIdCol = 'fieldId'
        self.raCol = 'fieldRA'
        self.decCol = 'fieldDec'
        self.propIdCol = 'propId'
        self.propNameCol = 'propName'
        self.propTypeCol = 'propType'
        # For config parsing.
        self.versionCol = 'version'
        self.sessionIdCol = 'sessionId'
        self.sessionHostCol = 'sessionHost'
        self.sessionDateCol = 'sessionDate'
        self.runCommentCol = 'runComment'
        self.runLengthParam = 'survey/duration'

    def fetchFieldsFromSummaryTable(self, sqlconstraint=None, raColName=None, decColName=None,
                                    degreesToRadians=True):
        f = super(OpsimDatabaseV4, self).fetchFieldsFromSummaryTable(sqlconstraint=sqlconstraint,
                                                                     raColName=raColName, decColName=decColName,
                                                                     degreesToRadians=degreesToRadians)
        return f

    def fetchFieldsFromFieldTable(self, propId=None, degreesToRadians=True):
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
        fielddata = self.query_columns('Field', colnames=[self.fieldIdCol, 'ra', 'dec'],
                                       groupBy = self.fieldIdCol)
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

        propData = self.query_columns('Proposal', colnames=[self.propIdCol, self.propNameCol],
                                      sqlconstraint=None)
        for propID, propName in zip(propData[self.propIdCol], propData[self.propNameCol]):
            # Fix these in the future, to use the proper tags that will be added to output database.
            propIDs[propID] = propName
            if 'widefastdeep' in propName.lower():
                propTags['WFD'].append(propID)
            if 'drill' in propName.lower():
                propTags['DD'].append(propID)
        return propIDs, propTags

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
        propData = self.query_columns('Proposal', colnames=[self.propNameCol, self.propTypeCol],
                                      sqlconstraint=constraint)
        nvisits = {}
        for f in self.filterlist:
            nvisits[f] = 0
        for pName, propType in zip(propData[self.propNameCol], propData[self.propTypeCol]):
            if propType.lower() == 'general':
                for f in self.filterlist:
                    constraint = 'paramName="science/general_props/values/%s/filters/%s/num_visits"' \
                                 % (pName, f)
                    val = self.query_columns('Config', colnames=['paramValue'], sqlconstraint=constraint)
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

    def _queryParam(self, constraint):
        results = self.query_columns('Config', colnames=['paramValue'], sqlconstraint=constraint)
        if len(results) > 0:
            return results['paramValue'][0]
        else:
            return '--'

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
        results = self.query_columns('Session',
                                     [self.versionCol, self.sessionDateCol, self.runCommentCol])
        configSummary['Version']['OpsimVersion'] = '%s'  %(results[self.versionCol][0])
        configSummary['Version']['OpsimDate'] = '%s' %(results[self.sessionDateCol][0])
        configSummary['Version']['OpsimDate'] = configSummary['Version']['OpsimDate'][0:10]
        configSummary['RunInfo'] = {}
        configSummary['RunInfo']['RunComment'] = results[self.runCommentCol][0]
        configSummary['RunInfo']['RunName'] = self.fetchOpsimRunName()
        # Pull out a few special values to put into summary.
        # This section has a number of configuration parameter names hard-coded.
        # I've left these here (rather than adding to self_colNames), bc I think schema changes in the config
        # files will actually be easier to track here (at least until the opsim configs are cleaned up).
        constraint = 'paramName="observatory/telescope/altitude_minpos"'
        configSummary['RunInfo']['MinAlt'] = self._queryParam(constraint)
        constraint = 'paramName="observatory/telescope/altitude_maxpos"'
        configSummary['RunInfo']['MaxAlt'] = self._queryParam(constraint)
        constraint = 'paramName="observatory/camera/filter_change_time"'
        configSummary['RunInfo']['TimeFilterChange'] = self._queryParam(constraint)
        constraint = 'paramName="observatory/camera/readout_time"'
        configSummary['RunInfo']['TimeReadout'] = self._queryParam(constraint)
        constraint = 'paramName="sched_driver/propboost_weight"'
        configSummary['RunInfo']['PropBoostWeight'] = self._queryParam(constraint)
        configSummary['RunInfo']['keyorder'] = ['RunName', 'RunComment', 'MinAlt', 'MaxAlt',
                                                'TimeFilterChange', 'TimeReadout', 'PropBoostWeight']

        # Echo config table into configDetails.
        configDetails = {}
        configs = self.query_columns('Config', ['paramName', 'paramValue'])
        for name, value in zip(configs['paramName'], configs['paramValue']):
            configDetails[name] = value

        # Now finish building the summary to add proposal information.
        # Loop through all proposals to add summary information.
        propData = self.query_columns('Proposal', [self.propIdCol, self.propNameCol, self.propTypeCol])
        configSummary['Proposals'] = {}
        for propid, propname, proptype in zip(propData[self.propIdCol],
                                              propData[self.propNameCol], propData[self.propTypeCol]):
            configSummary['Proposals'][propname] = {}
            propdict = configSummary['Proposals'][propname]
            propdict['keyorder'] = ['PropId', 'PropName', 'PropType', 'Airmass bonus', 'Airmass max',
                                    'HA bonus', 'HA max', 'Time weight', 'Restart Lost Sequences',
                                    'Restart Complete Sequences', 'Filters']
            propdict['PropName'] = propname
            propdict['PropId'] = propid
            propdict['PropType'] = proptype
            # Add some useful information on the proposal parameters.
            constraint = 'paramName like "science/%s_props/values/%s/sky_constraints/max_airmass"'\
                         % ("%", propname)
            propdict['Airmass max'] = self._queryParam(constraint)
            constraint = 'paramName like "science/%s_props/values/%s/scheduling/airmass_bonus"'\
                         % ("%", propname)
            propdict['Airmass bonus'] = self._queryParam(constraint)
            constraint = 'paramName like "science/%s_props/values/%s/scheduling/hour_angle_max"'\
                         % ("%", propname)
            propdict['HA max'] = self._queryParam(constraint)
            constraint = 'paramName like "science/%s_props/values/%s/scheduling/hour_angle_bonus"'\
                         % ("%", propname)
            propdict['HA bonus'] = self._queryParam(constraint)
            constraint = 'paramName like "science/%s_props/values/%s/scheduling/time_weight"'\
                         % ("%", propname)
            propdict['Time weight'] = self._queryParam(constraint)
            constraint = 'paramName like "science/%s_props/values/%s/scheduling/restart_lost_sequences"'\
                         % ("%", propname)
            propdict['Restart Lost Sequences'] = self._queryParam(constraint)
            constraint = 'paramName like "science/%s_props/values/%s/scheduling/restart_complete_sequences"'\
                         % ("%", propname)
            propdict['Restart Complete Sequences'] = self._queryParam(constraint)
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
                    propdict['Filters'][f][dk] = self._queryParam(constraint)
                propdict['Filters'][f]['keyorder'] = ['Filter', 'MaxSeeing', 'MinSky', 'MaxSky',
                                                      'NumVisits', 'GroupedVisits', 'Snaps']
            propdict['Filters']['keyorder'] = list(self.filterlist)
        return configSummary, configDetails


class OpsimDatabaseV3(BaseOpsimDatabase):
    def __init__(self, database, driver='sqlite', host=None, port=None, defaultTable='Summary',
                 longstrings=False, verbose=False):
        """
        Instantiate object to handle queries of the opsim database.
        (In general these will be the sqlite database files produced by opsim, but could
        be any database holding those opsim output tables.).

        database = Name of database or sqlite filename
        driver =  Name of database dialect+driver for sqlalchemy (e.g. 'sqlite', 'pymssql+mssql')
        host = Name of database host (optional)
        port = String port number (optional)

        """
        super(OpsimDatabaseV3, self).__init__(database=database, driver=driver, host=host, port=port,
                                              defaultTable=defaultTable, longstrings=longstrings,
                                              verbose=verbose)

    def _colNames(self):
        """
        Set variables to represent the common column names used in this class directly.

        This should make future schema changes a little easier to handle.
        It is NOT meant to function as a general column map, just to abstract values
        which are used *within this class*.
        """
        self.mjdCol = 'expMJD'
        self.slewId = 'slewHistory_slewID'
        self.delayCol = 'actDelay'
        self.fieldIdCol = 'fieldID'
        self.raCol = 'fieldRA'
        self.decCol = 'fieldDec'
        self.propIdCol = 'propID'
        self.propConfCol = 'propConf'
        self.propNameCol = 'propName' #(propname == proptype)
        # For config parsing.
        self.versionCol = 'version'
        self.sessionIdCol = 'sessionID'
        self.sessionHostCol = 'sessionHost'
        self.sessionDateCol = 'sessionDate'
        self.runCommentCol = 'runComment'
        self.runLengthParam = 'nRun'

    def fetchFieldsFromSummaryTable(self, sqlconstraint=None, raColName=None, decColName=None,
                                    degreesToRadians=False):
        f = super(OpsimDatabaseV3, self).fetchFieldsFromSummaryTable(sqlconstraint=sqlconstraint,
                                                                     raColName=raColName, decColName=decColName,
                                                                     degreesToRadians=degreesToRadians)
        return f

    def fetchFieldsFromFieldTable(self, propID=None, degreesToRadians=True):
        """
        Fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.

        propID = the proposal ID (default None), if selecting particular proposal - can be a list
        degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians).
        """
        # Note that you can't select any other sql constraints (such as filter).
        # This will select fields which were requested by a particular proposal or proposals,
        #   even if they didn't get any observations.
        if propID is not None:
            query = 'select f.%s, f.%s, f.%s from %s as f' %(self.fieldIdCol, self.raCol, self.decCol,
                                                             'Field')
            query += ', %s as p where (p.Field_%s = f.%s) ' %('Proposal_Field',
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
            fielddata = self.query_arbitrary(query, dtype=list(zip([self.fieldIdCol, self.raCol, self.decCol],
                                                                   ['int', 'float', 'float'])))
            if len(fielddata) == 0:
                fielddata = np.zeros(0, dtype=list(zip([self.fieldIdCol, self.raCol, self.decCol],
                                                  ['int', 'float', 'float'])))
        else:
            fielddata = self.query_columns('Field', colnames=[self.fieldIdCol, self.raCol, self.decCol],
                                           groupBy = self.fieldIdCol)
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
            propData = self.query_columns(self.defaultTable, colnames=[self.propIdCol])
            for propid in propData[self.propIdCol]:
                propIDs[int(propid)] = propid
        else:
            # Query for all propIDs.
            propData = self.query_columns('Proposal', colnames=[self.propIdCol, self.propConfCol,
                                                                self.propNameCol], sqlconstraint=None)
            for propid, propname in zip(propData[self.propIdCol], propData[self.propConfCol]):
                # Strip '.conf', 'Prop', and path info.
                propIDs[int(propid)] = re.sub('Prop','', re.sub('.conf','', re.sub('.*/', '', propname)))
            # Find the 'ScienceType' from the config table, to indicate DD/WFD/Rolling, etc.
            sciencetypes = self.query_columns('Config', colnames=['paramValue', 'nonPropID'],
                                              sqlconstraint="paramName like 'ScienceType'")
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

    def fetchRequestedNvisits(self, propId=None):
        """
        Find the requested number of visits for proposals in propId.
        Returns a dictionary - Nvisits{u/g/r/i/z/y}
        """
        visitDict = {}
        if propId is None:
            # Get all the available propIds.
            propData = self.query_columns('Proposal', colnames=[self.propIdCol, self.propNameCol],
                                          sqlconstraint=None)
        else:
            # Get the propType info to go with the propId(s).
            if hasattr(propId, '__iter__'):
                constraint = '('
                for pi in propId:
                    constraint += '(propId = %d) or ' %(pi)
                constraint = constraint[:-4] + ')'
            else:
                constraint = 'propId = %d' %(propId)
            propData = self.query_columns('Proposal', colnames=[self.propIdCol, self.propNameCol],
                                          sqlconstraint=constraint)
        for pId, propType in zip(propData[self.propIdCol], propData[self.propNameCol]):
            perPropConfig = self.query_columns('Config', colnames=['paramName', 'paramValue'],
                                               sqlconstraint = 'nonPropID = %d and paramName!="userRegion"'
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
        configSummary['Version']['MAFVersion'] = '%s' % (mafversion['__version__'])
        configSummary['Version']['MAFDate'] = '%s' % (mafdate)
        # Opsim date, version and runcomment info from session table
        results = self.query_columns('Session', colnames = [self.versionCol, self.sessionDateCol,
                                                            self.runCommentCol])
        configSummary['Version']['OpsimVersion'] = '%s' % (results[self.versionCol][0])
        configSummary['Version']['OpsimDate'] = '%s' % (results[self.sessionDateCol][0])
        configSummary['RunInfo'] = {}
        configSummary['RunInfo']['RunComment'] = results[self.runCommentCol]
        configSummary['RunInfo']['RunName'] = self.fetchOpsimRunName()
        # Pull out a few special values to put into summary.
        # This section has a number of configuration parameter names hard-coded.
        # I've left these here (rather than adding to self_colNames), because I think schema changes will in the config
        # files will actually be easier to track here (at least until the opsim configs are cleaned up).
        constraint = 'moduleName="Config" and paramName="sha1"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        try:
            configSummary['Version']['Config sha1'] = results['paramValue'][0]
        except IndexError:
            configSummary['Version']['Config sha1'] = 'Unavailable'
        constraint = 'moduleName="Config" and paramName="changedFiles"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        try:
            configSummary['Version']['Config changed files'] = results['paramValue'][0]
        except IndexError:
            configSummary['Version']['Config changed files'] = 'Unavailable'
        constraint = 'moduleName="instrument" and paramName="Telescope_AltMin"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        configSummary['RunInfo']['MinAlt'] = results['paramValue'][0]
        constraint = 'moduleName="instrument" and paramName="Telescope_AltMax"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        configSummary['RunInfo']['MaxAlt'] = results['paramValue'][0]
        constraint = 'moduleName="instrument" and paramName="Filter_MoveTime"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        configSummary['RunInfo']['TimeFilterChange'] = results['paramValue'][0]
        constraint = 'moduleName="instrument" and paramName="Readout_Time"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        configSummary['RunInfo']['TimeReadout'] = results['paramValue'][0]
        constraint = 'moduleName="scheduler" and paramName="MinDistance2Moon"'
        results = self.query_columns('Config', colnames=['paramValue', ], sqlconstraint=constraint)
        configSummary['RunInfo']['MinDist2Moon'] = results['paramValue'][0]
        configSummary['RunInfo']['keyorder'] = ['RunName', 'RunComment', 'MinDist2Moon', 'MinAlt', 'MaxAlt',
                                                'TimeFilterChange', 'TimeReadout']
        # Now build up config dict with 'nice' group names (proposal name and short module name)
        #  Each dict entry is a numpy array with the paramName/paramValue/comment values.
        # Match proposal IDs with names.
        query = 'select %s, %s, %s from Proposal group by %s' %(self.propIdCol, self.propConfCol,
                                                                self.propNameCol, self.propIdCol)
        propdata = self.query_arbitrary(query, dtype=([(self.propIdCol, int),
                                                       (self.propConfCol, np.str, 256),
                                                       (self.propNameCol, np.str, 256)]))
        # Make 'nice' proposal names
        propnames = np.array([os.path.split(x)[1].replace('.conf', '') for x in propdata[self.propConfCol]])
        # Get 'nice' module names
        moduledata = self.query_columns('Config', colnames=['moduleName',], sqlconstraint='nonPropID=0')
        modulenames = np.array([os.path.split(x)[1].replace('.conf', '') for x in moduledata['moduleName']])
        # Grab the config information for each proposal and module.
        cols = ['paramName', 'paramValue', 'comment']
        for longmodname, modname in zip(moduledata['moduleName'], modulenames):
            config[modname] = self.query_columns('Config', colnames=cols,
                                                 sqlconstraint='moduleName="%s"' %(longmodname))
            config[modname] = config[modname][['paramName', 'paramValue', 'comment']]
        for propid, propname in zip(propdata[self.propIdCol], propnames):
            config[propname] = self.query_columns('Config', colnames=cols,
                                                  sqlconstraint='nonPropID="%s" and paramName!="userRegion"'
                                                                %(propid))
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
            configSummary['Proposals']['keyorder'].append(propnames[np.where(propdata[self.propIdCol]
                                                                             == propid)][0])
        for propid, propname in zip(propdata[self.propIdCol], propnames):
            configSummary['Proposals'][propname] = {}
            propdict = configSummary['Proposals'][propname]
            propdict['keyorder'] = [self.propIdCol, self.propNameCol, 'PropType', 'RelPriority',
                                    'NumUserRegions', 'NumFields']
            propdict[self.propNameCol] = propname
            propdict[self.propIdCol] = propid
            propdict['PropType'] = propdata[self.propNameCol][np.where(propnames == propname)]
            propdict['RelPriority'] = self._matchParamNameValue(config[propname], 'RelativeProposalPriority')
            # Get the number of user regions.
            constraint = 'nonPropID="%s" and paramName="userRegion"' %(propid)
            result = self.query_columns('Config', colnames=['paramName',], sqlconstraint=constraint)
            propdict['NumUserRegions'] = result.size
            # Get the number of fields requested in the proposal (all filters).
            propdict['NumFields'] = self.fetchFieldsFromFieldTable(propID=propid).size
            # Find number of visits requested per filter for the proposal, with min/max sky & airmass values.
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
                propdict['SubSeq'], Nvisits = self._parseSequences(config[propname],
                                                                   propdict['PerFilter']['Filters'])
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
                tmp = propdict['PerFilter'][k][idx]
                propdict['PerFilter'][k] = tmp
            propdict['PerFilter']['keyorder'] = ['Filters', 'VisitTime', 'MaxSeeing', 'MinSky',
                                                 'MaxSky', 'NumVisits']
        return configSummary, config

