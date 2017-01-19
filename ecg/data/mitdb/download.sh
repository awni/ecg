#!/bin/bash
# For more info about the data visit
# https://physionet.org/physiobank/database/mitdb/
# and 
# https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

path=$1
mkdir $path
cd $path
url=https://physionet.org/physiobank/database/mitdb/
for i in {100..234}
do
    for ext in 'hea' 'dat' 'atr'
    do
        wget $url/$i.$ext
    done
done
