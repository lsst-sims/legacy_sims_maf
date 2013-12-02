import numpy
import lsst.sims.operations.maf.utils.astromStack as asstack
import lsst.sims.operations.maf.db as db


bandpass = 'r'
dbTable = 'output_opsim3_61_forLynne' 
#dbTable = 'output_opsim2_145_forLynne'   
dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'  
table = db.Table(dbTable, 'obsHistID', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expMJD',  'night',
                                                 'fieldRA', 'fieldDec',
                                                 '5sigma_modified', 'seeing'], 
                                                 groupByCol='expMJD')

ack = asstack.astroStack(simdata)
