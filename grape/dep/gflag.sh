#!/bin/bash -x

cd gflags
if [ $? -ne 0 ]
then
    echo "gflags directory not found"	
    git clone https://github.com/gflags/gflags.git
    if [ $? -ne 0 ]
    then
        echo "Unable to perform git clone https://github.com/gflags/gflags.git"
        exit
    fi
fi
cd gflags
mkdir build
cd build/
cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DINSTALL_HEADERS=ON -DINSTALL_SHARED_LIBS=ON -DINSTALL_STATIC_LIBS=ON ..
make
make install

