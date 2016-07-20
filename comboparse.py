#!/usr/bin/env python
import pickle

infilename = "combo_summary.txt"
infile = open(infilename, "r")

names = []
percents = []

for line in infile:
    lines = line.split(":")
    digit = lines[1].split(" ")
    numb = float(digit[1])
    if numb > 80:
        names.append(lines[0])
        percents.append(numb)

data=list(zip(names,percents))

pickle.dump(data,open("combo_80.pkl","wb"))
