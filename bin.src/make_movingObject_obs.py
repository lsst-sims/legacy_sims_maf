#!/usr/bin/env python

from __future__ import print_function, division
import os
import argparse
from lsst.sims.maf.objUtils.moObs import runMoObs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate moving object detections for a particular opsim run.")
    parser.add_argument("opsimDb", type=str, help="Opsim run db file")
    parser.add_argument("--orbitFile", type=str, default='pha20141031.des',
                        help="File containing the moving object orbits.")
    parser.add_argument("--outDir", type=str,
                        default='.', help="Output directory for moving object detections.")
    parser.add_argument("--obsFile", type=str, default=None,
                        help="Output file name for moving object observations. Default will build opsimRun_orbitFile_obs.txt.")
    parser.add_argument("--tStep", type=float, default=2./24.0,
                        help="Timestep between ephemeris generation / linear interpolation steps (in days). Default 2 hours.")
    args = parser.parse_args()

    print('Making moving object observations from %s for opsim run %s' % (args.orbitFile, args.opsimDb))

    orbitbase = '.'.join(os.path.split(args.orbitFile)[-1].split('.')[:-1])
    opsimRun = os.path.split(args.opsimDb)[-1].replace('_sqlite.db', '')
    if args.obsFile is None:
        obsFile = os.path.join(args.outDir, '%s_%s_obs.txt' % (opsimRun, orbitbase))
    else:
        obsFile = args.obsFile

    runMoObs(args.orbitFile, outFile, args.opsimDb, tstep=args.tStep, useCamera=True)
