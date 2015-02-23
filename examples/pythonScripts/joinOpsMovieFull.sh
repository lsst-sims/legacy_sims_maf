#!/bin/tcsh

## This script will generate but not run the commands to concatenate mp4 files generated as movies
## of each individual opsim night, combining VisitFilters/Nvisits(in each filter and all filters) using
##  'comboCommand' and outputting the list of combo videos to combine in bigMovieList.

## Note that the bigMovieList file consists of a line pointing to each mp4 file, then another pointing to
##  a special "blank.mp4" file, which is part of the git repo. The 'blank.mp4' file places a short
##  blank space between each movie night. Adding more versions of 'blank.mp4' will add a longer pause.
## Note that the bigMovieList file could be edited and the ffmpeg command rerun to change these pauses, etc.

## The 'blank.mp4' file can be re-generated (if a different size is needed, for example) by combining a blank png
##  from matplotlib with a blank automatically generated frame from ffmpeg (optionally, including the logo).
## ffmpeg -f lavfi -i color=c=white:s=965x648:d=0.7 -i blank_1.png -i lsstLogo.png \
## -filter_complex "[0:v] setpts=PTS-STARTPTS [background]; [1:v] setpts=PTS-STARTPTS [blank]; \
## [2:v] setpts=PTS-STARTPTS,scale=150:-1 [logo]; [background][blank] overlay=shortest=0 [bgd]; \
## [bgd][logo] overlay=shortest=0:x=10:y=10" -r 30 -pix_fmt yuv420p -crf 18 -preset slower blank2.mp4

## usage: expects to be run after mkOpsMovie.sh is used to generate movies for a series of individual nights.
## usage: joinOpsMovieFull.sh [opsim name] [start night] [end night]

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
    echo 'ffmpeg -f lavfi -i color=c=white:s=965x648 \' >> $comboCommand
    echo ' -f lavfi -i color=c=black:s=2x648\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/VisitFilters_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_u_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_g_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_r_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_i_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_z_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i '$opsRun'_n'$night'/Nvisits_y_SkyMap_30.0_30.0.mp4\'  >> $comboCommand
    echo ' -i lsstLogo.png\'  >> $comboCommand
    echo ' -filter_complex "[0:v] setpts=PTS-STARTPTS [background];\'  >> $comboCommand
    echo ' [1:v] setpts=PTS-STARTPTS [stripe];\'  >> $comboCommand
    echo ' [2:v] setpts=PTS-STARTPTS [pointings];\'  >> $comboCommand
    echo ' [3:v] setpts=PTS-STARTPTS,scale=288:-1 [cumulative];\'  >> $comboCommand
    echo ' [4:v] setpts=PTS-STARTPTS,scale=192:-1 [u];\'  >> $comboCommand
    echo ' [5:v] setpts=PTS-STARTPTS,scale=192:-1 [g];\'  >> $comboCommand
    echo ' [6:v] setpts=PTS-STARTPTS,scale=192:-1 [r];\'  >> $comboCommand
    echo ' [7:v] setpts=PTS-STARTPTS,scale=192:-1 [i];\'  >> $comboCommand
    echo ' [8:v] setpts=PTS-STARTPTS,scale=192:-1 [z];\'  >> $comboCommand
    echo ' [9:v] setpts=PTS-STARTPTS,scale=192:-1 [y];\'  >> $comboCommand
    echo ' [10:v] setpts=PTS-STARTPTS,scale=150:-1 [logo];\'  >> $comboCommand
    echo ' [background][stripe] overlay=x=581:y=0 [bgd];\'  >> $comboCommand
    echo ' [bgd][pointings] overlay=shortest=1:x=0:y=100 [tmp1];\'  >> $comboCommand
    echo ' [tmp1][cumulative] overlay=shortest=1:x=630:y=0 [tmp2];\'  >> $comboCommand
    echo ' [tmp2][u] overlay=shortest=1:x=583:y=200 [tmp3];\'  >> $comboCommand
    echo ' [tmp3][g] overlay=shortest=1:x=775:y=200 [tmp4];\'  >> $comboCommand
    echo ' [tmp4][r] overlay=shortest=1:x=583:y=344 [tmp5];\'  >> $comboCommand
    echo ' [tmp5][i] overlay=shortest=1:x=775:y=344 [tmp6];\'  >> $comboCommand
    echo ' [tmp6][z] overlay=shortest=1:x=583:y=488 [tmp8];\'  >> $comboCommand
    echo ' [tmp8][y] overlay=shortest=1:x=775:y=488 [tmp9];\'  >> $comboCommand
    echo ' [tmp9][logo] overlay=shortest=0:x=10:y=10"\'  >> $comboCommand
    echo ' -r 30 -pix_fmt yuv420p -crf 18 -preset slower '$opsRun'_n'$night'/combo_30.0_30.0.mp4' >> $comboCommand
    echo 'file '$opsRun'_n'$night'/combo_30.0_30.0.mp4' >> $combomovieList
    echo 'file blank2.mp4' >> $combomovieList
 endif
end

echo '# To create long movie with both VisitFilters + Nvisits, over multiple nights:'
echo 'source '$comboCommand
echo 'ffmpeg -f concat -i '$combomovieList' -c copy '$opsRun'_n'$nightStart'_n'$nightEnd'.mp4'
