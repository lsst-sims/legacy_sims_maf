#!/usr/bin/env python

import argparse
import subprocess
import warnings

if __name__ == "__main__":
    """
    Copy a MAF output directory from one location to another using rsync.
    examples:
    cpMAF.py mafdir /some/path/or/other/here
    cpMAF.py /some/MAF/dir user@computer.name.edu:"~/here/"

    Note that the source directory name will always be the same at the destination.
    """
    parser = argparse.ArgumentParser(description="Copy MAF output directory using rsync")
    parser.add_argument("dirSource", type=str, help="directory to copy")
    parser.add_argument("dirDest", type=str, help="destination to copy to")
    parser.add_argument("--noNpz", dest='noNpz', default=False, action='store_true',
                        help="skip the .npz files")
    parser.add_argument("--noPdf", dest='noPdf', default=False, action='store_true',
                        help="skip the .pdf files")
    args = parser.parse_args()
    source = args.dirSource
    while source[-1] == '/':
        warnings.warn('stripping trailing slash on the source directory')
        source = source[:-1]

    dest = args.dirDest

    callList = ['rsync', '-rav']
    if args.noNpz:
        callList.append("--exclude '*.npz'")
    if args.noPdf:
        callList.append("--exclude '*.pdf'")
    callList = callList + [source, dest]
    command = ' '.join(callList)
    subprocess.call([command], shell=True)
