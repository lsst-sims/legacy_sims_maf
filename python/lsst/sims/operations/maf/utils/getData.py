import os
import numpy as np
import lsst.sims.operations.maf.db as db

def getDbAddress(connectionName='SQLITE_OPSIM', dbLoginFile=None):
    """Utility to get the dbAddress info corresponding to 'connectionName' from a dbLogin file.

    connectionName is the name given to a sqlalchemy connection string in the file
        (default 'SQLITE_OPSIM').
    dbLoginFile is the file location (default None will try to use $HOME/dbLogin). """
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


def fetchFieldsFromFieldTable(fieldTable, dbAddress, 
                              sessionID=None, proposalTable='tProposal_Field', proposalID=None,
                              degreesToRadians=True):
    """Utility to fetch field information (fieldID/RA/Dec) from Field (+Proposal_Field) tables.

    dbTable = the Field table
    dbAddress = the sqlalchemy connection string
    sessionID = the opsim session ID, needed if proposalID != None
    proposalTable = the Proposal_Field table
    proposalID = the proposal ID (default None), if selecting particular proposal
    degreesToRadians = RA/Dec values are in degrees in the Field table (so convert to radians) -- HACK """
    # Fetch field information from the Field table, plus Proposal_Field table if proposalID != None.
    # Note that you can't select any other sql constraints (such as filter). 
    # This will select fields which were requested by a particular proposal (or which were part of
    # the simulation), even if they didn't get any observations. 
    table = db.Table(fieldTable, 'fieldID', dbAddress)
    if proposalID != None:
        query = 'select f.fieldID, f.fieldRA, f.fieldDec from %s as f, %s as p' \
        %(fieldTable, proposalTable)
        if sessionID != None:
            query += ' where (p.Field_fieldID=f.fieldID) and (p.Session_sessionID=%d) and (p.Proposal_propID=%d)' \
              %(int(sessionID), int(proposalID))
        else:
            query += ' where (p.Field_fieldID=f.fieldID) and (p.Proposal_propID=%d)' %(proposalID)
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
