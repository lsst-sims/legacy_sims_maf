import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

def runBenchmarks(benchmarks):
    """

    """

    database = db.OpsimDatabase(benchmark.dbAddress)

    dbcols = benchmark.findReqCols()
    simdata = utils.getSimData(database, benchmark.sqlWhere, dbcols)

    for stacker in benchmark.stackers:
        simdata = stacker.run(simdata)

    # Set up the slicer
    slicer = benchmark.slicer
    maps = benchmark.maps
    if slicer.slicerName == 'OpsimFieldSlicer':
        fieldData = benchmark.getFieldData()
        slicer.setupSlicer(simdata, fieldData, maps=maps)
    else:
        slicer.setupSlicer(simdata, maps=maps)
