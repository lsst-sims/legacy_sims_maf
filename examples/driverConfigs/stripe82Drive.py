# Let's try to connect to fatboy and run some metrics on stripe 82

from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict


root.dbAddress={'dbAddress':'mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433'}


import lsst.sims.maf.db as db

stripe = db.Table(u'viewStripe82JoinAll', 'id', 'mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433')

stripe = db.Database('mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433', dbTables={'viewStripe82JoinAll':['viewStripe82JoinAll','id']})

from sqlalchemy import *
ack = create_engine('mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433')





# Here we go, this works

stripe = db.Database('mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433', dbTables={'viewStripe82JoinAll':['viewStripe82JoinAll','id']})

data = stripe.queryDatabase('viewStripe82JoinAll', 'select top 10 filter from clue.dbo.viewStripe82JoinAll  ;')

