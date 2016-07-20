#!/usr/bin/env python

#script to convert parsed data into useable format

def amino_dict():
    aminofilename = "amino_acid_letter_code.txt"
    aminofile = open(aminofilename, 'r')
    Amino = {}
    LineNumber = 0
    Codes = []

    for Line in aminofile:
        if(LineNumber>0):
            Codes = Line.split(",")
            code = Codes[2].strip("\n")
            numb = Codes[3].strip("\n")
            Amino[code]=int(numb)
        LineNumber = LineNumber + 1
    aminofile.close()
    return Amino
