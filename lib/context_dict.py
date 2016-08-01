#!/usr/bin/env python
import numpy as np
import pickle
import os
from .aa_feature_dict import feature_dict

def contextdict(title,choose,rand=True,args=False):
    if rand or args:
        feat_dict = feature_dict()
        context_dict = {}
        for i in range(20):
            context_dict[i+1]=[]
        if rand:
            randnumfeat = np.random.uniform(1,5,size = 1)
            featlist = list(randnumfeat)
            featint = int(featlist[0])
            randdict = np.random.choice(16,4,replace=False)
            randfeats = []

            for i in range(featint):
                randfeats.append(randdict[i-1])
        elif args:
            helpargs = """
            #    Input the following digit to include it in the data:
            #    0: Chou-Fasman helix propensity value
            #    1: Amino acid molecular weight
            #    2: pKa value for free amino acid carboxylate
            #    3: pKa value for free amino acid amine
            #    4: pKa approximation for amino acid side chain
            #    5: Number of carbon atoms in amino acid
            #    6: Number of hydrogen atoms in amino acid zwitterion
            #    7: Number of nitrogen atoms in amino acid
            #    9: Number of sulfur atoms in amino acid
            #   10: Area in the standard state (standard state accessibiity
            #       is defined as the Average surface area that residue
            #       has in an ensemble of Gly-X-Gly tripeptides)
            #   11: Average accessible area in proteins
            #   12: Average area buried upon transfer from the standard
            #       state to the folded proteins
            #   13: Mean fractional area loss, equal to the average area buried
            #       normalized by standard state area
            #   14: Residue mass
            #   15: Monoisotopic mass
            #       If a digit greater than 15 is entered,
            #       an error will occur.
            """
            if choose == "Help" or "":
                print(helpargs)
            if choose == None:
                pass
            else:
                for i in range(len(choose)):
                    if type(choose[i]) is not int:
                        raise Exception('Non-integer value entered as feature')
                    elif choose[i] > 15:
                        raise Exception('Invalid integer entered as feature')
                randfeats = choose
        tempfeatures = []
        randfeats.sort()
        for k in feat_dict.keys():
            for i in range(len(randfeats)):
                tempfeatures.append(feat_dict[k][randfeats[i]])
                if len(tempfeatures) == len(randfeats):
                    context_dict[k]=tempfeatures
                    tempfeatures = []

        features = []
        for i in range(len(randfeats)):
            if randfeats[i]==0:
                features.append("Chou-Fasman helix propensity value")
            elif randfeats[i]==1:
                features.append("Amino acid molecular weight")
            elif randfeats[i]==2:
                features.append("pKa of free amino acid carboxylate")
            elif randfeats[i]==3:
                features.append("pKa of free amino acid amine")
            elif randfeats[i]==4:
                features.append("Acidity/Basicity of side chain")
            elif randfeats[i]==5:
                features.append("Number of carbons in amino acid")
            elif randfeats[i]==6:
                features.append("Number of hydrogens in amino acid (zwitterion)")
            elif randfeats[i]==7:
                features.append("Number of nitrogens in amino acid")
            elif randfeats[i]==8:
                features.append("Number of oxygens in amino acid")
            elif randfeats[i]==9:
                features.append("Number of sulfurs in amino acid")
            elif randfeats[i]==10:
                features.append("Amino acid area in standard state")
            elif randfeats[i]==11:
                features.append("Average accessible area in protein")
            elif randfeats[i]==12:
                features.append("Average area buried in folded protein")
            elif randfeats[i]==13:
                features.append("Mean fractional area lost")
            elif randfeats[i]==14:
                features.append("Residue mass")
            elif randfeats[i]==15:
                features.append("Monoisotopic mass")
            else:
                print("A feature not currently accounted for has been added to your run?!")


        if len(features) == 16:
            Feats = ("This run contains all possible features.")
        elif len(features) == 15:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12],features[13],features[14]))
        elif len(features) == 14:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12],features[13]))
        elif len(features) == 13:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12]))
        elif len(features) == 12:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11]))
        elif len(features) == 11:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10]))
        elif len(features) == 10:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9]))
        elif len(features) == 9:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8]))
        elif len(features) == 8:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7]))
        elif len(features) == 7:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5],features[6]))
        elif len(features) == 6:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4],features[5]))
        elif len(features) == 5:
            Feats = ("This run contains the following features: %s, %s, %s, %s, %s." % (features[0],features[1],features[2],features[3],features[4]))
        elif len(features) == 4:
            Feats = ("%s,%s,%s,%s" % (features[0],features[1],features[2],features[3]))
        elif len(features) == 3:
            Feats = ("%s,%s,%s" % (features[0],features[1],features[2]))
        elif len(features) == 2:
            Feats = ("%s,%s" % (features[0],features[1]))
        else:
            Feats = ("%s" % (features[0]))

    #################################################################################################################################

    else:
        Feats = ("This run will contain no features.")
        context_dict = {}
        for i in range(20):
            context_dict[i+1]=[i+1]

    return context_dict, Feats
