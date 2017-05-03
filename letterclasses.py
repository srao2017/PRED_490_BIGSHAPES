# -*- coding: utf-8 -*-
"""
Created on Tue May 02 13:36:33 2017

@author: admin
"""
import os
import csv
import pandas as pd

os.chdir("C:/EDUCATION/NWU/490 - Deep Learning/Week 5")

f = open(str("ABCD_26_VARS_classes.csv"),"rb")
df = pd.read_csv(f)
f.close()

cl = pd.Series(df["Class"]).unique()
cl.sort()
for c in cl:
    letters = df[df['Class'] == c]
    v = letters["Letter"]
    vl = v.tolist()
    print ("O"+str(c)+str(letters["Letter"]))
