import os
import scipy.stats as stat
import math
import time
import sys
import warnings

warnings.filterwarnings('ignore')


def ssn_score(deta, pcc, nn):
    if pcc == 1:
        pcc = 0.99999999
    if pcc == -1:
        pcc = -0.99999999
    z = deta / ((1 - pcc * pcc) / (nn - 1))
    return z


begin = time.asctime()
print(begin)

param = {}
for i in range(1, len(sys.argv)):
    t = sys.argv[i].split("=")
    param[t[0].lower()] = t[1]

help_msg = """
usage: python construct_single_network.py -pvalue=threshold_p_value | -threshold=threshold_value -background=background_network_file -ref=reference_sample_file  -sample=sample_data_file -out=results_output_fold
Options and arguments:
-pvalue : set the threshold of p-value [0..1], if the -pvalue set 1, all edges will be outputted to the SSN
-threshold : set the threshold value of the absolute value of deltaPCC [0..2]
-background : background network to calculate the deltaPCC of edges based on the network
-comutation : background network comutation score
-ref : the expression profile of reference samples
-sample : the expression profile for the sample to be constructed the SSN
-out : the directory to store the SSN
"""

if "-help" in param.keys() or "-h" in param.keys():
    print(help_msg)

if "-ref" not in param.keys() or "-sample" not in param.keys() or "-background" not in param.keys():
    print("Parameter missing!")
    print(help_msg)
    exit()

disease = {}
normal = {}

name_c = []

reference_file = param["-ref"]
sample_file = param["-sample"]
background = param["-background"]
comutation = param["-comutation"]

if "-pvalue" in param.keys() and "-threshold" in param.keys():
    print("The parameters -pvalue and -threshold cannot be set at the same time!")
    exit()
if "-pvalue" not in param.keys() and "-threshold" not in param.keys():
    print("The parameter -pvalue or -threshold need be set!")
    exit()

threshold = 0
p_value = 1
if "-pvalue" in param.keys():
    p_value = float(param["-pvalue"])
    if p_value < 0 or p_value > 1:
        print("Please set the correct threshold of p-value in -pvalue [0..1]")
        exit()
else:
    threshold = float(param["-threshold"])
    if threshold < 0 or threshold > 2:
        print("Please set the correct threshold value in -threshold [0..2]")
        exit()

if "-out" not in param.keys():
    fold = "."
else:
    fold = param["-out"]

if not os.path.exists(fold):
    os.mkdir(fold)

f = open("data_sample/{}".format(reference_file))
flag = 0
n = 0
for p in f:
    flag += 1
    t = p.split()
    if flag == 1:
        n = len(t)
        name_c = t[1:]
        continue
    normal[t[0]] = [float(t[i]) for i in range(1, n)]
f.close()

name_d = []

f = open("data_sample/{}".format(sample_file))
flag = 0
n = 0
for p in f:
    flag += 1
    t = p.split()
    if flag == 1:
        n = len(t)
        name_d = t[1:]
        continue
    disease[t[0]] = [float(t[i]) for i in range(1, n)]
f.close()

if "-pvalue" in param.keys():
    p_value = float(param["-pvalue"])
    for i in range(len(name_d)):
        print("sample", name_d[i])
        fw = open(fold + os.sep + "ssn_" + name_d[i] + ".txt", "w")
        fw.write("Gene1\tGene2\tdeltaPCC\tp-value\n")
        f = open("data_sample/{}".format(background))
        c = open("data_sample/{}".format(comutation))
        c = str(c).split()
        index = i
        k = 0
        for p in f:
            k += 1
            t = p.split()
            r1 = stat.pearsonr(normal[t[0]], normal[t[1]])[0]
            r2 = stat.pearsonr(normal[t[0]] + [disease[t[0]][i]], normal[t[1]] + [disease[t[1]][i]])[0]

            r = r2 - r1
            z = ssn_score(r, r1, len(normal[t[0]]))
            pvalue = 1 - stat.norm.cdf(abs(z))
            pvalue *= c[k]
            if pvalue < p_value:
                fw.write(t[0] + "\t" + t[1] + "\t" + str(r) + "\t" + str(pvalue) + "\n")

        f.close()
        fw.close()
else:
    threshold = float(param["-threshold"])
    for i in range(len(name_d)):
        print("sample", name_d[i])
        fw = open(fold + os.sep + "ssn_" + name_d[i] + ".txt", "w")
        fw.write("Gene1\tGene2\tdeltaPCC\tp-value\n")
        f = open("data_sample/{}".format(background))
        c = open("data_sample/{}".format(comutation))
        c = str(c).split()
        index = i
        k = 0
        for p in f:
            t = p.split()
            r1 = stat.pearsonr(normal[t[0]], normal[t[1]])[0]
            r2 = stat.pearsonr(normal[t[0]] + [disease[t[0]][i]], normal[t[1]] + [disease[t[1]][i]])[0]

            r = r2 - r1
            z = ssn_score(r, r1, len(normal[t[0]]))
            pvalue = 1 - stat.norm.cdf(abs(z))
            pvalue *= c[k]
            if abs(r) > threshold:
                fw.write(t[0] + "\t" + t[1] + "\t" + str(r) + "\t" + str(pvalue) + "\n")

        f.close()
        fw.close()

print("Begin time: " + begin)
print("End time: " + time.asctime())
