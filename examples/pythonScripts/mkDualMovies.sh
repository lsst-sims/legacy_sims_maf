#!/bin/csh

## This script is intended to be used as a convenience to help generate commands for making cumulative + differential movies,
## then combining the results into a single movie. Will echo, but not run, the commands.

## usage: mkDualMovies.sh [opsim name]
## (edit this file to change the sql constraint).

set opsimRun = $1
set sqlconstraint = 'filter="r"'
set sqlconstraint = ''

echo "python example_movie.py "$opsimRun".db --movieStepsize 30 --nside 32 --ips 5 --sqlConstraint '"$sqlconstraint"' --outDir "$opsimRun"_cumulative"

echo "python example_movie.py "$opsimRun".db --movieStepsize 30 --nside 32 --ips 5 --sqlConstraint '"$sqlconstraint"' --binned --outDir "$opsimRun"_binned"

echo "ffmpeg -i "$opsimRun"_cumulative/"$opsimRun"_N_Visits_HEAL_SkyMap_5.0_5.0.mp4 -i "$opsimRun"_binned/"$opsimRun"_N_Visits_HEAL_SkyMap_5.0_5.0.mp4 -filter_complex 'nullsrc=size=1152x576 [background]; [0:v] setpts=PTS-STARTPTS [left]; [1:v] setpts=PTS-STARTPTS [right]; [background][left] overlay=shortest=1 [background+left]; [background+left][right] overlay=shortest=1:x=576 [left+right]' -map '[left+right]' -r 5  -pix_fmt yuv420p "$opsimRun"_N_Visits.mp4"

echo "ffmpeg -i "$opsimRun"_cumulative/"$opsimRun"_Coaddm5Metric_HEAL_SkyMap_5.0_5.0.mp4 -i "$opsimRun"_binned/"$opsimRun"_Coaddm5Metric_HEAL_SkyMap_5.0_5.0.mp4 -filter_complex 'nullsrc=size=1224x388 [background]; [0:v] setpts=PTS-STARTPTS [left]; [1:v] setpts=PTS-STARTPTS [right]; [background][left] overlay=shortest=1 [background+left]; [background+left][right] overlay=shortest=1:x=612 [left+right]' -map '[left+right]' -r 5 -pix_fmt yuv420p "$opsimRun"_Coaddm5.mp4"



