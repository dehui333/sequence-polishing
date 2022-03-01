#!/bin/bash

rm -r /scratch/sequence-polishing/build/

python setup.py build_ext
python setup.py install