#!/bin/bash -x

cd glog
if [ $? -ne 0 ]
then
    echo "glog directory not found"	
    git clone -b v0.4.0 https://github.com/google/glog
    if [ $? -ne 0 ]
    then
        echo "Unable to perform git clone https://github.com/google/glog"
        exit
    fi
fi
sudo apt-get install autoconf automake libtool
cd glog
./autogen.sh
./configure
make -j 6
sudo make install

