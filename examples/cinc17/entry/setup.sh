#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail


# challenge system has not enough space in /tmp, apparently
export TMPDIR=$HOME/temp
mkdir -p $TMPDIR

virtualenv -p python2.7 myenv
source myenv/bin/activate
pip install --upgrade packages/setuptools-39.2.0.zip  
pip install --upgrade packages/wheel-0.31.1.tar.gz
pip install --upgrade packages/pip-10.0.1.tar.gz
pip install -r requirements.txt --find-links packages/
