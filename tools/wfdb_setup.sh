#!/bin/bash

# NB, this script has only been tested on deep machines.
install_dir=/deep/group/med/tools
cd $install_dir
wget https://physionet.org/physiotools/wfdb.tar.gz
tar -xzvf wfdb.tar.gz
rm wfdb.tar.gz
wfdb_dir=`ls | grep wfdb`
cd $wfdb_dir
mkdir build
./configure --static_only --prefix=$install_dir/$wfdb_dir/build
make install
