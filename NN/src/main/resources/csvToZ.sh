#!/bin/bash
rm -f Z.csv
for f in $(ls [0-9]*.csv) ; do
	filename=$(basename "$file")
	echo filename="${f%.*}"
	extension="${f##*.}"
	cat ${f} | sed 's/0/1/g; s/16777215/0/g' | tr '\r\n' ','  | sed 's/[,]$//g'  >> Z.csv
	echo "" >> Z.csv
done
