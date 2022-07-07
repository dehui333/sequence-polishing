#!/bin/bash

cd Dependencies/htslib-1.9
chmod +x ./configure ./version.sh

#apt update
#apt install build-essential
#apt-get install zlib1g-dev

./configure CFLAGS=-fpic --disable-bz2 --disable-lzma --without-libdeflate && make

cd ../..

python3 setup.py build_ext
python3 setup.py install

rm -r ./build/
