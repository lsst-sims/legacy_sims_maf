import os
import sqlite3
from sqlite3 import OperationalError, IntegrityError

import numpy as np
import pandas as pd

import healpy as hp
import lsst.sims.featureScheduler as sched


def get_standard_wfd():
    # Get the standard survey footprint (in each bandpass)
    nside = 64
    standard_footprint = sched.utils.standard_goals(nside=nside)
    # WFD is where footprint values = max
    wfd_footprint = {}
    for f in standard_footprint:
        threshold = np.max(standard_footprint[f])
        wfd_footprint[f] = np.where(standard_footprint[f] >= threshold, 1, 0)
    return wfd_footprint


def define_ddname(note):
    field = note.replace('u,', '')
    return field


def get_visits(opsimdb):
    conn = sqlite3.connect(opsimdb)
    visits = pd.read_sql('select observationId, fieldRA, fieldDec, filter, note from summaryallprops', conn)
    conn.close()
    return(visits)


def label_visits(visits, wfd_footprint):
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
    pointings = []
    propId = np.zeros(len(visits), int)
    for i, (v, note) in enumerate(zip(vec, visits['note'])):
        pointing_healpix = hp.query_disc(nside, v, radius, inclusive=False)
        pointings.append(pointing_healpix)
        in_wfd = wfd_footprint['r'][pointing_healpix].sum()
        propId[i] = np.where(in_wfd / len(pointing_healpix) > 0.4, propTags['WFD'], 0)
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
            '"proposalName" VARCHAR(10), "proposalType" VARCHAR(5))'
    cursor.execute(sql)
    # Add proposal information to Proposal table.
    for pName, pId in propTags.items():
        pType = pName.split(':')[0]
        try:
            sql = f'INSERT INTO Proposal (proposalId, proposalName, proposalType) VALUES ("{pId}", "{pName}", "{pType}")'
            cursor.execute(sql)
        except IntegrityError:
            print(f'This proposal ID is already in the proposal table! {pId},{pName}')
    # Add data to proposalID column.
    # 0 = general, 1 = WFD, 2 = DD.
    for obsid, pId in zip(visits.observationId, propId):
        sql = f'UPDATE SummaryAllProps SET proposalId = {pId} WHERE observationId = {obsid}'
        cursor.execute(sql)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add approximate proposal id labels to FBS visits, '
                                                 'assuming standard WFD footprint.')
    parser.add_argument('dbfile', type=str, help='sqlite file of observations (full path).')
    args = parseArgs()

    # If you need a non-standard WFD footprint, you must modify the wfd_footprint.
    # Have a look at how the run was set up, and use the footprint specified there.
    wfd_footprint = get_standard_wfd()
    visits = get_visits(parser.dbfile)
    visits, propTags, propId = label_visits(visits, wfd_footprint)
    update_database(parser.dbfile, visits, propTags, propId)
