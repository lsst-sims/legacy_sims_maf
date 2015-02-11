#!/bin/tcsh

set opsRun = $1
set nightStart = $2
set nightEnd = $3
echo "#Making movie from " $opsRun " for nights " $nightStart " to " $nightEnd

if ($4) then
   set sqlconstraint = $4" and"
   echo " using general sqlconstraint " $sqlconstraint
else
   set sqlconstraint = ''
endif


set nights = `seq $nightStart $nightEnd`
foreach night ( $nights )
 set nightconstraint = $sqlconstraint" night="$night
 echo "python opsimMovie.py "$opsRun"_sqlite.db --sqlConstraint "$nightconstraint" --ips 30 --addPreviousObs --outDir "$opsRun"_n"$night
end

