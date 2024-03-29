#!/bin/bash

module load python/3.6
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --upgrade pip
pip install --no-index numpy scipy matplotlib pandas
pip freeze > requirements.txt
deactivate
rm -rf $ENVDIR
