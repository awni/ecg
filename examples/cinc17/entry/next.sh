#! /bin/bash
#
# file: next.sh
#
# This bash script analyzes the record named in its command-line
# argument ($1), and writes the answer to the file 'answers.txt'.
# This script is run once for each record in the Challenge test set.
#
# The program should print the record name, followed by a comma,
# followed by one of the following characters:
#   N   for normal rhythm
#   A   for atrial fibrillation
#   O   for other abnormal rhythms
#   ~   for records too noisy to classify
#
# For example, if invoked as
#    next.sh A00001
# it analyzes record A00001 and (assuming the recording is
# considered to be normal) writes "A00001,N" to answers.txt.

set -e
set -o pipefail

RECORD=$1

# || true so we can run locally
source myenv/bin/activate || true

printf "$RECORD," >> answers.txt
python evaler.py $RECORD >> answers.txt
