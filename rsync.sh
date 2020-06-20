#!/bin/bash

# Simply run the rsync command
rsync -ruv {Makefile,benchmark.py,digits.csv,src} DAS5:thesis/ --delete
