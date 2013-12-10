import numpy
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics
import glob


bandpass = 'r'

dbTable = 'output_opsim3_61'
dbAddress = 'postgres://calibuser:calibuser@ivy.astro.washington.edu:5432/calibDB.05.05.2010'


table = db.Table(dbTable, 'obshistid', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expmjd',  'night',
                                                 'fieldra', 'fielddec', 'airmass',
                                                 '5sigma_modified', 'seeing',
                                                 'skybrightness_modified', 'altitude',
                                                 'hexdithra', 'hexdithdec'], 
                                                 groupByCol='expmjd')

