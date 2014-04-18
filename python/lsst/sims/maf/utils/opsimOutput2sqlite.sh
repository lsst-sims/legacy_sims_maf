#!/bin/bash

# a simple script to convert an opsim output file to a sqlite3 file

# $1 = opsim input .dat file
# $2 = outfilename
# $3 = table name

echo 'CREATE TABLE' $3 '(  "obsHistID" INTEGER,"sessionID" INTEGER,"propID" INTEGER, "fieldID" INTEGER,"fieldRA" NUMERIC(16, 6), "fieldDec" NUMERIC(16, 6),filter VARCHAR(1),"expDate" INTEGER,"expMJD" NUMERIC(16, 6),night INTEGER,"visitTime" NUMERIC(16, 6),"visitExpTime" NUMERIC(16, 6),"finRank" NUMERIC(16, 6), finSeeing NUMERIC(16, 6),transparency NUMERIC(16, 6),airmass NUMERIC(16, 6), "VSkyBright" NUMERIC(16, 6),"filtSkyBright" NUMERIC(16, 6), "rotSkyPos" NUMERIC(16, 6), lst NUMERIC(16, 6), altitude NUMERIC(16, 6),azimuth NUMERIC(16, 6), "dist2Moon" NUMERIC(16, 6),"solarElong" NUMERIC(16, 6),"moonRA" NUMERIC(16, 6),"moonDec" NUMERIC(16, 6),"moonAlt" NUMERIC(16, 6),"moonAZ" NUMERIC(16, 6),"moonPhase" NUMERIC(16, 6),"sunAlt" NUMERIC(16, 6),"sunAz" NUMERIC(16, 6),"phaseAngle" NUMERIC(16, 6),"rScatter" NUMERIC(16, 6),"mieScatter" NUMERIC(16, 6),"moonIllum" NUMERIC(16, 6),"moonBright" NUMERIC(16, 6),"darkBright" NUMERIC(16, 6),"rawSeeing" NUMERIC(16, 6), "wind" NUMERIC(16, 6), "humidity" NUMERIC(16, 6), "slewDist" NUMERIC(16, 6),"slewTime" NUMERIC(16, 6),"5sigma" NUMERIC(16, 6),perry_skybrightness NUMERIC(16, 6),"5sigma_ps" NUMERIC(16, 6),skybrightness_modified NUMERIC(16, 6),"5sigma_modified" NUMERIC(16, 6),hexdithra NUMERIC(16, 6),hexdithdec NUMERIC(16, 6),vertex INTEGER);' > dat2lite.sql

#echo '.mode csv' >> dat2lite.sql
echo '.separator "\t"' >> dat2lite.sql
echo '.headers ON' >> dat2lite.sql
echo '.import '$1 $3  >> dat2lite.sql
echo 'DELETE FROM '$3 'WHERE rowid = 1; '>> dat2lite.sql  #delete the header

sqlite3 $2 < dat2lite.sql
rm dat2lite.sql
