#!/bin/bash

# Simply scp the necessary files under thesis_benchmark/ folder on given server
if [ "$#" != "1" ] && [ "$#" != "2" ]; then
	echo "Usage: $0 <server_url> [<folder>]"
	exit -1
fi

folder="thesis_benchmark/"
if [ "$#" == "2" ]; then
	folder=$2
fi

if [ "$1" != "localhost" ]; then 
	rsync -ruv {Makefile,benchmark.py,digits.csv,src} "$1:$folder/" --delete
else
	# Copy it to one folder up benchmark/
	rm -rf ../benchmark/{Makefile,benchmark.py,digits.csv,src}
	cp -r {Makefile,benchmark.py,digits.csv,src} ../benchmark/
fi

