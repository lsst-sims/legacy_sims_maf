set opsRun = $1
set nightStart = $2
set nightEnd = $3
echo "#Joining movie from " $opsRun " for nights " $nightStart " to " $nightEnd

set nights = `seq $nightStart $nightEnd`
set movieList = 'tmpMovieList'
if (-e $movieList) then
   rm $movieList
   endif
foreach night ( $nights )
 if (-e $opsRun"_n"$night"/movieFrame_SkyMap_30.0_30.0.mp4") then
    echo "file "$opsRun"_n"$night"/movieFrame_SkyMap_30.0_30.0.mp4" >> $movieList
    echo "file blank.mp4" >> $movieList
 endif
 end

echo "ffmpeg -f concat -i "$movieList" -c copy "$opsRun"_n"$nightStart"_n"$nightEnd".mp4"

