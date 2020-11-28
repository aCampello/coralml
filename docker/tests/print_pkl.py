#!/usr/bin/env python
# -*- coding: utf-8 -*-



import sys
import pickle


with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)


for key in data:
    print(f"{key} = {data[key]}")
