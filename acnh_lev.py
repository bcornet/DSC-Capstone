# scratch file

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from levenshtein import levenshtein_ratio_and_distance as lev
from levenshtein import levmat

# this all takes a bit
# newest version of this spreadsheet here:
# https://docs.google.com/spreadsheets/d/13d_LAJPlxMa_DubPTuirkIV4DERBMXbrWQsmSh8ReK4/edit#gid=667767649
AC = pd.ExcelFile("Data Spreadsheet for Animal Crossing New Horizons.xlsx")

# this all takes a while; should be 605 housewares in ACNH, so 182710 calculations
# comparatively, MHGU skills has 205 items, so 20910 calculations
housewares = pd.read_excel(AC,"Housewares")
house = housewares.Name.unique().tolist()

acnh_levmat_house = levmat(house,output="s")

# heatmap shows many bright spots (not good)
# same set items ("wooden-block ___", "ironwood ___", etc.)
# same shape items ("____ chair", "____ bed", etc.)
# crazy coincidences ("peach chair" and "beach chair" for example)
# closest match was 0.96 for Mr. and Mrs. Flamingo, congrats to the newlyweds
# this indicates a potential accuracy issue when using a large dictionary
# note also that ACNH capitalizes the first letter in each item in menus
# will need to account for list corrections

acnh_levmat_house[acnh_levmat_house < 1].sort_values(ascending=False,inplace=True)


s = len(lst) # length of list
if s < 1:
    return
acnh_levmat = np.eye(s) # identity matrix based on length of list
print("Starting Levenshtein matrix...")
for i in range(0,s):
    for j in range(i+1,s):
        acnh_levmat[i,j] = lev(lst[i],lst[j],True)
print("Done with Levenshtein matrix!\nShowing plot...")

symmat = acnh_levmat + acnh_levmat.T - np.eye(s)
plt.imshow(symmat, cmap='hot', interpolation='nearest')
plt.show()
print("Done!")

