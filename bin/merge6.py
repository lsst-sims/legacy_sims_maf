#! /usr/bin/env python
import os, argparse
import subprocess


if __name__=="__main__":

    """
    Merge the u,g,r,i,z,y plots into a 3x2 grid.
    Requires pdfjam: http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/firth/software/pdfjam
    """

    parser = argparse.ArgumentParser(description="Merge 6 plots into a single pdf")
    parser.add_argument("fileBase", type=str, help="filename of one of the files to merge.")
    parser.add_argument("-O", "--outfile", type=str, default=None, help="Output filename")
    args = parser.parse_args()
    fileBase = args.fileBase

    # Make a list of the 6 files:
    filters = ['u','g','r','i','z','y']
    for f in filters:
        if '_'+f+'_' in fileBase:
            fileBase = fileBase.split('_'+f+'_')
            fileBase[0] = fileBase[0]

    fileList = [ fileBase[0]+'_'+f+'_'+fileBase[1] for f in filters]
    if args.outfile is None:
        outfile = fileBase[0]+'_6_'+fileBase[1]
    else:
        outfile = args.outfile

    callList = ["pdfjoin", "--nup 3x2 ", "--outfile "+outfile]+fileList
    command=''
    for item in callList: command=command+' '+item
    subprocess.call([command], shell=True)

