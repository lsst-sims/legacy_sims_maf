#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import argparse
import sqlite3
from sqlite3 import OperationalError, IntegrityError

import numpy as np
import pandas as pd

import healpy as hp
from lsst.sims.featureScheduler import utils as schedUtils

__all__ = ['define_ddname', 'get_visits', 'label_visits', 'update_database']


def define_ddname(note):
    field = note.replace('u,', '')
    return field


def get_visits(opsimdb):
    conn = sqlite3.connect(opsimdb)
    visits = pd.read_sql('select observationId, fieldRA, fieldDec, filter, note from summaryallprops', conn)
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
            '"proposalName" VARCHAR(10), "proposalType" VARCHAR(5))'
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
                                                 'assuming standard WFD footprint.')
    parser.add_argument('dbfile', type=str, help='sqlite file of observations (full path).')
    parser.add_argument('--wfd', type=str, default='standard',
                        help='Type of wfd footprint [standard, extended, dust]')
    args = parser.parse_args()

    # There are some standard WFD footprints: the standard version, the extendded version (with gal-lat cut),
    # and an extended version with dust extinction cuts.
    # If you're using something else (non-standard), this script will have to be modified.
    # Note these "footprints" are standard (full) length healpix arrays for nside, but with values of 0/1
    wfd_defaults = ['standard', 'extended', 'dust', 'extended with dust']
    nside = 64

    if args.wfd.lower() == 'standard':
        wfd_footprint = schedUtils.WFD_no_gp_healpixels(nside)
    elif args.wfd.lower() == 'extended':
        wfd_footprint = schedUtils.WFD_bigsky_healpixels(nside)
    elif args.wfd.lower() == 'dust' or args.wfd.lower() == 'extended with dust':
        wfd_footprint = schedUtils.WFD_no_dust_healpixels(nside)
    else:
        raise ValueError(f'This script understands wfd footprints of types {wfd_defaults}')

    visits = get_visits(args.dbfile)
    visits, propTags, propId = label_visits(visits, wfd_footprint)
    update_database(args.dbfile, visits, propTags, propId)
