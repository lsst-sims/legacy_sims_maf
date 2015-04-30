import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

def runBenchmarks(benchmarks):
    """

    """

    dbcols = benchmark.findReqCols()

    database = db.OpsimDatabase(benchmark.dbAddress)
    simdata = utils.getSimData(database, benchmark.sqlWhere, dbcols)

    for stacker in benchmark.stackers:
        simdata = stacker.run(simdata)
