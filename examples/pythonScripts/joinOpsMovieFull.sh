#!/bin/tcsh

## This script will generate but not run the commands to concatenate mp4 files generated as movies
## of each individual opsim night, combining VisitFilters/Nvisits(in each filter and all filters) using
##  'comboCommand' and outputting the list of combo videos to combine in bigMovieList.

## Note that the bigMovieList file consists of a line pointing to each mp4 file, then another pointing to
##  a special "blank.mp4" file, which is part of the git repo. The 'blank.mp4' file places a short
##  blank space between each movie night. Adding more versions of 'blank.mp4' will add a longer pause.
## Note that the bigMovieList file could be edited and the ffmpeg command rerun to change these pauses, etc.

## usage: expects to be run after mkOpsMovie.sh is used to generate movies for a series of individual nights.
## usage: joinOpsMovie.sh [db name] [start night] [end night]

set opsRun = $1
set nightStart = $2
set nightEnd = $3
echo "#Joining movie from " $opsRun " for nights " $nightStart " to " $nightEnd

set nights = `seq $nightStart $nightEnd`

# Set up to make VisitFilters + Nvisits movie.
set combomovieList = 'bigMovieList'
set comboCommand = 'comboCommand'
if (-e $combomovieList) then
   rm $combomovieList
endif
if (-e $comboCommand) then
  rm $comboCommand
endif

foreach night ($nights)
 if (-e $opsRun"_n"$night"/VisitFilters_SkyMap_30.0_30.0.mp4") then
    # combine NVisits and VisitFilters
    # "normal" opsim movie size is 576x432
    echo "ffmpeg -f lavfi -i color=c=white:s=870x502 -f lavfi -i color=c=black:s=2x502 -i "$opsRun"_n"$night"/VisitFilters_SkyMap_30.0_30.0.mp4 -i "$opsRun"_n"$night"/Nvisits_SkyMap_30.0_30.0.mp4  -i "$opsRun"_n"$night"/Nvisits_u_SkyMap_30.0_30.0.mp4 -i "$opsRun"_n"$night"/Nvisits_g_SkyMap_30.0_30.0.mp4 -i "$opsRun"_n"$night"/Nvisits_r_SkyMap_30.0_30.0.mp4 -i "$opsRun"_n"$night"/Nvisits_i_SkyMap_30.0_30.0.mp4 -i "$opsRun"_n"$night"/Nvisits_z_SkyMap_30.0_30.0.mp4 -i "$opsRun"_n"$night"/Nvisits_y_SkyMap_30.0_30.0.mp4 -i lsstLogo.png -filter_complex '[0:v] setpts= [background]; [1:v] setpts=PTS-STARTPTS [stripe]; [2:v] setpts=PTS-STARTPTS [pointings]; [3:v] setpts=PTS-STARTPTS,scale=288:-1 [cumulative]; [4:v] setpts=PTS-STARTPTS,scale=144:-1 [u]; [5:v] setpts=PTS-STARTPTS,scale=144:-1 [g]; [6:v] setpts=PTS-STARTPTS,scale=144:-1 [r]; [7:v] setpts=PTS-STARTPTS,scale=144:-1 [i]; [8:v] setpts=PTS-STARTPTS,scale=144:-1 [z]; [9:v] setpts=PTS-STARTPTS,scale=144:-1 [y]; [10:v] setpts=PTS-STARTPTS,scale=100:-1 [logo];  [background][stripe] overlay=x=581:y=0 [bgd]; [bgd][pointings] overlay=shortest=1:x=0:y=35 [tmp1]; [tmp1][cumulative] overlay=shortest=1:x=583:y=0 [tmp2]; [tmp2][u] overlay=shortest=1:x=583:y=200 [tmp3]; [tmp3][g] overlay=shortest=1:x=727:y=200 [tmp4];  [tmp4][r] overlay=shortest=1:x=583:y=297 [tmp5];  [tmp5][i] overlay=shortest=1:x=727:y=297 [tmp6];  [tmp6][z] overlay=shortest=1:x=583:y=394 [tmp8];  [tmp8][y] overlay=shortest=1:x=727:y=394 [tmp9]; [tmp9][logo] overlay=shortest=0:x=10:y=10' -r 30 -pix_fmt yuv420p -crf 18 -preset slower "$opsRun"_n"$night"/combo_30.0_30.0.mp4" >> $comboCommand
    echo "file "$opsRun"_n"$night"/combo_30.0_30.0.mp4" >> $combomovieList
    echo "file blank.mp4" >> $combomovieList
 endif
end

echo "# To create long movie with both VisitFilters + Nvisits, over multiple nights:"
echo "source "$comboCommand
echo "ffmpeg -f concat -i "$combomovieList" -c copy "$opsRun"_n"$nightStart"_n"$nightEnd".mp4"
