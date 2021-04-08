#!/usr/bin/env python

import os
import matplotlib
matplotlib.use('Agg')

import argparse
import sqlite3
from sqlite3 import OperationalError, IntegrityError

import numpy as np
import pandas as pd
import healpy as hp

from lsst.sims.maf.metrics import CountMetric
from lsst.sims.maf.slicers import HealpixSlicer
import lsst.sims.maf.metricBundles as mb


__all__ = ['define_ddname', 'get_visits', 'label_visits', 'update_database']


def define_ddname(note):
    field = note.replace('u,', '')
    return field


def get_visits(opsimdb):
    conn = sqlite3.connect(opsimdb)
    query = 'select observationId, observationStartMJD, fieldRA, fieldDec, filter, note from summaryallprops'
    visits = pd.read_sql(query, conn)
    conn.close()
    return(visits)


def label_visits(visits, wfd_footprint, nside=64):
    # Set up DD names.
    d = set()
    for p in visits['note'].unique():
        if p.startswith('DD'):
            d.add(define_ddname(p))
    # Define dictionary of proposal tags.
    propTags = {'Other': 0, 'WFD': 1}
    for i, field in enumerate(d):
        propTags[field] = i + 2
    # Identify Healpixels associated with each visit.
    vec = hp.dir2vec(visits['fieldRA'], visits['fieldDec'], lonlat=True)
    vec = vec.swapaxes(0, 1)
    radius = np.radians(1.75)  # fov radius
    #pointings = []
    propId = np.zeros(len(visits), int)
    for i, (v, note) in enumerate(zip(vec, visits['note'])):
        # Identify the healpixels which would be inside this pointing
        pointing_healpix = hp.query_disc(nside, v, radius, inclusive=False)
        # This can be useful for debugging/plotting
        #pointings.append(pointing_healpix)
        # The wfd_footprint consists of values of 0/1 if out/in WFD footprint
        in_wfd = wfd_footprint[pointing_healpix].sum()
        # So in_wfd = the number of healpixels which were in the WFD footprint
        # .. in the # in / total # > limit (0.4) then "yes" it's in WFD
        propId[i] = np.where(in_wfd / len(pointing_healpix) > 0.4, propTags['WFD'], 0)
        # BUT override - if the visit was taken for DD, use that flag instead.
        if note.startswith('DD'):
            propId[i] = propTags[define_ddname(note)]
    return visits, propTags, propId


def update_database(opsimdb, visits, propTags, propId):
    # Write visits_wfd into a new column in the table.
    conn = sqlite3.connect(opsimdb)
    cursor = conn.cursor()
    # Add indexes on observationStartMJD and observationId and filter
    try:
        indxMJD = "CREATE UNIQUE INDEX idx_observationStartMJD on SummaryAllProps (observationStartMJD);"
        cursor.execute(indxMJD)
    except OperationalError:
        print('Already had observationStartMJD index')
    try:
        indxObsId = "CREATE UNIQUE INDEX idx_observationId on SummaryAllProps (observationId)"
        cursor.execute(indxObsId)
    except OperationalError:
        print('Already had observationId index')
    try:
        indxFilter = "CREATE INDEX idx_filter on SummaryAllProps (filter)"
        cursor.execute(indxFilter)
    except OperationalError:
        print('Already had filter index')
    # Add new table to track proposal information.
    sql = 'CREATE TABLE IF NOT EXISTS "Proposal" ("proposalId" INT PRIMARY KEY, ' \
            '"proposalName" VARCHAR(20), "proposalType" VARCHAR(5))'
    cursor.execute(sql)
    # Add proposal information to Proposal table.
    for pName, pId in propTags.items():
        pType = pName.split(':')[0]
        try:
            sql = f'INSERT INTO Proposal (proposalId, proposalName, proposalType) VALUES ("{pId}", "{pName}", "{pType}")'
            cursor.execute(sql)
        except IntegrityError:
            print(f'This proposal ID is already in the proposal table {pId},{pName} (just reusing it)')
    # Add data to proposalID column.
    # 0 = general, 1 = WFD, 2 = DD.
    for obsid, pId in zip(visits.observationId, propId):
        sql = f'UPDATE SummaryAllProps SET proposalId = {pId} WHERE observationId = {obsid}'
        cursor.execute(sql)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add approximate proposal id labels to FBS visits, '
                                                 'assuming WFD == area with 750 or more visits.')
    parser.add_argument('dbfile', type=str, help='sqlite file of observations (full path).')
    args = parser.parse_args()

    # Pull out the visits. We'll reuse these for the footprint calculation below.
    # Note that visits is a dataframe.
    visits = get_visits(args.dbfile)

    # Instead of always figuring out the WFD footprint from what we expected, let's define it based on
    # what we got .. and define the "WFD" as the area with at least 750 visits per pointing.
    runName = os.path.split(args.dbfile)[-1].replace('.db', '')
    nside = 64
    # Find the WFD footprint
    m = CountMetric(col='observationStartMJD')
    s = HealpixSlicer(nside)
    simdata = visits.query('visitExposureTime > 11')
    simdata = simdata.query('not note.str.startswith("DD")', engine='python').to_records()
    bundle = mb.MetricBundle(m, s, 'long notDD', runName=runName)
    g = mb.MetricBundleGroup({f'{runName}': bundle}, None)
    g.setCurrent('long notDD')
    g.runCurrent('long notDD', simData=simdata)

    wfd_footprint = bundle.metricValues.filled(0)
    wfd_footprint = np.where(wfd_footprint > 750, 1, 0)

    visits, propTags, propId = label_visits(visits, wfd_footprint)
    update_database(args.dbfile, visits, propTags, propId)
