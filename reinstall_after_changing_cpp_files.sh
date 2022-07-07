#!/bin/bash

rm -r ./build/

python3 setup.py build_ext
python3 setup.py install

rm -r ./build/
