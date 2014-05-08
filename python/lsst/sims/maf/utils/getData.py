# Utilities for fetching simData and fieldData from the opsim database
#  (these should probably be put into the DB section)

import os
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning) # Ignore db warning
    import lsst.sims.maf.db as db


def getDbAddress(connectionName='SQLITE_OPSIM', dbLoginFile=None):
    """Utility to get the dbAddress info corresponding to 'connectionName' from a dbLogin file.

    connectionName is the name given to a sqlalchemy connection string in the file
        (default 'SQLITE_OPSIM').
    dbLoginFile is the file location (default None will try to use $HOME/dbLogin). """
    ## This is actually replicating functionality already in catalogs.generation, so should go away
    # The dbLogin file is a file containing simple 'names' corresponding to sqlite connection engine
    #  strings.
    # Example:
    # SQLITE_OPSIM sqlite:///opsim.sqlite
    # MYSQL_OPSIM mysql://lsst:lsst@localhost/opsim
    #  More information on sqlalchemy connection strings can be found at
    #  http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html
    if dbLoginFile is None:
        # Try default location in home directory.
        dbLoginFile = os.path.join(os.getenv("HOME"), 'dbLogin')
    f = open(dbLoginFile, 'r')
    for l in f:
        els = l.rstrip().split()
        if els[0] == connectionName:
            dbAddress = els[1]
    return dbAddress
            
def fetchSimData(dbTable, dbAddress, sqlconstraint, colnames, distinctExpMJD=True):
    """Utility to fetch opsim simulation data (colnames). 

    dbTable = the opsim data table, such as the output_* table.
    dbAddress = the sqlalchemy connection string.
    colnames = the columns to fetch from the table.
    distinctExpMJD = group by expMJD to get unique observations only (default True)."""
    table = db.Table(dbTable, 'obsHistID', dbAddress)
    if distinctExpMJD:
        simdata = table.query_columns_RecArray(chunk_size=10000000, 
                                               constraint = sqlconstraint,
                                               colnames = colnames, 
                                               groupByCol = 'expMJD')
    else:
        simdata = table.query_columns_RecArray(chunk_size=10000000, 
                                               constraint = sqlconstraint,
                                               colnames = colnames)
    return simdata


def fetchFieldsFromOutputTable(dbTable, dbAddress, sqlconstraint):
    """Utility to fetch field information (fieldID/RA/Dec) from opsim output_* table. """
    # Fetch field info from the output_* table, by selecting unique fieldID + ra/dec values.
    # This implicitly only selects fields which were actually observed by opsim.
    table = db.Table(dbTable, 'obsHistID', dbAddress)
    fielddata = table.query_columns_RecArray(constraint=sqlconstraint,
                                             colnames=['fieldID', 'fieldRA',  'fieldDec'],
                                             groupByCol='fieldID')
    return fielddata


def fetchFieldsFromFieldTable(dbAddress, fieldTable='Field', proposalFieldTable='Proposal_Field',
                              proposalID=None, sessionID=None, degreesToRadians=True):
    """Utility to fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.

    dbAddress = the sqlalchemy connection string
    dbTable = the Field table
    proposalFieldTable = the Proposal_Field table
    proposalID = the proposal ID (default None), if selecting particular proposal - can be a list
    sessionID = the opsim session ID (generally not actually needed)
    degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians) -- HACK """
    # Fetch field information from the Field table, plus Proposal_Field table if proposalID != None.
    # Note that you can't select any other sql constraints (such as filter). 
    # This will select fields which were requested by a particular proposal or proposals,
    #   even if they didn't get any observations. 
    table = db.Table(fieldTable, 'fieldID', dbAddress)
    if proposalID != None:
        query = 'select f.fieldID, f.fieldRA, f.fieldDec from %s as f' %(fieldTable)
        if proposalID != None:
            query += ', %s as p where (p.Field_fieldID = f.fieldID) ' %(proposalFieldTable)
            if hasattr(proposalID, '__iter__'): # list of propIDs
                query += ' and ('
                for propID in proposalID:
                    query += '(p.Proposal_propID = %d) or ' %(int(propID))
                # Remove the trailing 'or' and add a closing parenthesis.
                query = query[:-3]
                query += ')'
            else: # single proposal ID.
                query += ' and (p.Proposal_propID = %d) ' %(int(proposalID))
        if sessionID != None:
            query += ' and (p.Session_sessionID=%d)' %(int(sessionID))
        results = table.engine.execute(query)
        fielddata = table._postprocess_results(results.fetchall())
    else:
        table = db.Table(fieldTable, 'fieldID', dbAddress)
        fielddata = table.query_columns_RecArray(colnames=['fieldID', 'fieldRA', 'fieldDec'],
                                                 groupByCol = 'fieldID')
    if degreesToRadians:
        fielddata['fieldRA'] = fielddata['fieldRA'] * np.pi / 180.
        fielddata['fieldDec'] = fielddata['fieldDec'] * np.pi / 180.
    return fielddata

    
        
        
