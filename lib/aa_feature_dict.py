#!/usr/bin/env python

#Script to add features to amino acids to increase accuracy of network

def feature_dict():
    featurefilename = "lib/aa_data_chart.txt"
    aminofile = open(featurefilename, 'r')
    Feature = {}
    LineNumber = 0
    Codes = []

    for Line in aminofile:
        if(LineNumber>0):
            Codes = Line.split(",")
            numb = int(Codes[0].strip(" "))
            helicode = Codes[3].strip(" ")
            sheetcode = Codes[4].strip(" ")
            heliprop = Codes[5].strip(" ")
            sheetprop = Codes[6].strip(" ")
            aaweight = Codes[7].strip(" ")
            pkacarb = float(Codes[8].strip(" "))
            pkaamine = float(Codes[9].strip(" "))
            pkaR = Codes[10].strip(" ")
            #pka acid, base, neutral approximation
            #0 = neutral, 1 = acidic, 2 = basic
            if pkaR == "":
                pkaR = 0
            else:
                pkaR = float(pkaR)
                if pkaR < 7:
                    pkaR = 1
                else:
                    pkaR = 2
            numcarbs = Codes[11].strip(" ")
            numhyd = Codes[12].strip(" ")
            numnitro = Codes[13].strip(" ")
            numoxy = Codes[14].strip(" ")
            numsulfur = Codes[15].strip(" ")
            if numsulfur == "":
                numsulfur = 0
            area = Codes[16].strip(" ")
            accessarea = Codes[17].strip(" ")
            buriedarea = Codes[18].strip(" ")
            arealoss = Codes[19].strip(" ")
            residmass = Codes[20].strip(" ")
            monoisomass = Codes[21].strip("\n")
            Feature[numb]=(float(heliprop), float(aaweight), float(pkacarb), float(pkaamine),
                            int(pkaR), int(numcarbs), int(numhyd), int(numnitro), int(numoxy), int(numsulfur),
                            float(area), float(accessarea), float(buriedarea), float(arealoss), float(residmass),
                            float(monoisomass))
        LineNumber = LineNumber + 1
    aminofile.close()
    return Feature
