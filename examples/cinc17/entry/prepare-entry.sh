#!/bin/bash
#
# file: prepare-entry.sh

set -e
set -o pipefail

mkdir entry && cd entry


## Copy source code files
echo `pwd`
cp ../../../../LICENSE LICENSE.txt
cp ../../config.json .
cp ../setup.sh .
cp ../next.sh .
cp ../AUTHORS.txt .
cp ../dependencies.txt .
cp ../evaler.py .

pip download -r ../requirements.txt -d packages

src_dir='../../../../ecg'
for f in 'util.py' 'load.py' 'network.py'; do
    cp $src_dir/$f .
done

## Copy model files
python ../weights_only.py $1
cp `dirname $1`/preproc.bin preproc.bin 

echo "==== running entry script on validation set ===="
validation=../../data/sample2017/validation

for r in `cat $validation/RECORDS`; do
    echo $r
    ln -sf $validation/$r.hea .
    ln -sf $validation/$r.mat .
    ./next.sh $r
    rm $r.hea $r.mat
done

## Make zip
rm  *.pyc
zip -r entry.zip .
