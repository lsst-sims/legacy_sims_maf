#!/bin/csh

## This script is intended to be used as a convenience to help generate commands for making cumulative + differential movies,
## then combining the results into a single movie. Will echo, but not run, the commands. 

set opsimRun = $1
set sqlconstraint = 'filter="r"'

echo "python example_movie.py "$opsimRun"_sqlite.db --movieStepsize 30 --nside 64 --ips 5 --sqlConstraint '"$sqlconstraint"' --raCol ditheredRA --decCol ditheredDec --outDir "$opsimRun"_r_cumulative"

echo "python example_movie.py "$opsimRun"_sqlite.db --movieStepsize 30 --nside 64 --ips 5 --sqlConstraint '"$sqlconstraint"' --raCol ditheredRA --decCol ditheredDec --binned --outDir "$opsimRun"_r_binned"

echo "ffmpeg -i "$opsimRun"_r_cumulative/"$opsimRun"_N_Visits_HEAL_SkyMap_5.0_5.0.mp4 -i "$opsimRun"_r_binned/"$opsimRun"_N_Visits_HEAL_SkyMap_5.0_5.0.mp4 -filter_complex 'nullsrc=size=1224x388 [background]; [0:v] setpts=PTS-STARTPTS [left]; [1:v] setpts=PTS-STARTPTS [right]; [background][left] overlay=shortest=1 [background+left]; [background+left][right] overlay=shortest=1:x=612 [left+right]' -map '[left+right]' -r 5  -pix_fmt yuv420p "$opsimRun"_r_N_Visits.mp4"

echo "ffmpeg -i "$opsimRun"_r_cumulative/"$opsimRun"_Coaddm5Metric_HEAL_SkyMap_5.0_5.0.mp4 -i "$opsimRun"_r_binned/"$opsimRun"_Coaddm5Metric_HEAL_SkyMap_5.0_5.0.mp4 -filter_complex 'nullsrc=size=1224x388 [background]; [0:v] setpts=PTS-STARTPTS [left]; [1:v] setpts=PTS-STARTPTS [right]; [background][left] overlay=shortest=1 [background+left]; [background+left][right] overlay=shortest=1:x=612 [left+right]' -map '[left+right]' -r 5 -pix_fmt yuv420p "$opsimRun"_r_Coaddm5.mp4"



