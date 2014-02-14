import numpy as np
import lsst.sims.operations.maf.driver as driver
import sys


if __name__=="__main__":
    filename = sys.argv[1]
    drive = driver.MafDriver(filename)
    drive.run()
    
