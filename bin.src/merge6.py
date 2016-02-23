#!/usr/bin/env python

import argparse
import subprocess


if __name__ == "__main__":
    """
    Merge u,g,r,i,z,y plots into a 3x2 grid.
    Requires pdfjam:
    http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/firth/software/pdfjam
    Can be used with .pdf or .png files, but the output will always be a pdf

    examples:
    merge6.py ops1_1122_Median_fiveSigmaDepth_i_band_WFD_HEAL_SkyMap.pdf
    merge6.py thumb.ops1_1122_Median_fiveSigmaDepth_u_band_all_props_HEAL_SkyMap.png
    """

    parser = argparse.ArgumentParser(description="Merge 6 plots into a single pdf")
    parser.add_argument("fileBase", type=str, help="filename of one of the files to merge.")
    parser.add_argument("-O", "--outfile", type=str, default=None, help="Output filename")
    args = parser.parse_args()
    fileBase = args.fileBase

    if '/' in fileBase:
        fileBase = fileBase.split('/')
        path = fileBase[:-1]
        path = '/'.join(path) + '/'
        fileBase = fileBase[-1]
    else:
        path = ''

    # Make a list of the 6 files:
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    for f in filters:
        if '_' + f + '_' in fileBase:
            fileBase = fileBase.split('_' + f + '_')

    fileList = [path + fileBase[0] + '_' + f + '_' + fileBase[1] for f in filters]
    if args.outfile is None:
        outfile = fileBase[0] + '_6_' + fileBase[1]
    else:
        outfile = args.outfile

    # can only output pdf files
    if outfile[-3:] == 'png':
        outfile = outfile[:-3] + 'pdf'

    callList = ["pdfjoin", "--nup 3x2 ", "--outfile " + outfile] + fileList
    command = ''
    for item in callList:
        command = command + ' ' + item
    subprocess.call([command], shell=True)
