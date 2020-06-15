#!/bin/bash

# Simply scp the necessary files under thesis_benchmark/ folder on given server
if [ "$#" != "1" ]; then
	echo "Usage: $0 <server_url>"
	exit -1
fi

rsync -ruv {Makefile,benchmark.py,digits.csv,src} "$1:thesis_benchmark/" --delete 

