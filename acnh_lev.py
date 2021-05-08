# scratch file

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from levenshtein import levenshtein_ratio_and_distance as lev
from levenshtein import levmat


fileName = 'acnh_house.csv'

try:
    acnh_house = pd.read_csv(fileName)['HOUSE'].tolist()
    print("Loaded data from '%s' successfully."%fileName)
except:
    print("Can't find '%s', downloading data from Google."%fileName)
    # newest version of this spreadsheet here:
    # https://docs.google.com/spreadsheets/d/13d_LAJPlxMa_DubPTuirkIV4DERBMXbrWQsmSh8ReK4/edit#gid=667767649
    #AC = pd.ExcelFile("Data Spreadsheet for Animal Crossing New Horizons.xlsx")
    # haven't tried this; if you get errors, just follow the link or google the filename above
    # or you know what, just get the .csv file from Bri's GitHub unless you really care about Animal Crossing
    AC = pd.ExcelFile("https://docs.google.com/spreadsheets/d/13d_LAJPlxMa_DubPTuirkIV4DERBMXbrWQsmSh8ReK4/edit#gid=667767649")
    # note that New Horizons sees content updates every month or so; this list will probably include more items
    housewares = pd.read_excel(AC,"Housewares")
    acnh_house = housewares.Name.unique().tolist()

# this all takes a while; should be 605 housewares in ACNH, so 182710 calculations
# comparatively, MHGU skills has 205 items, so 20910 calculations
acnh_levmat = levmat(house,output="s")

# heatmap shows many bright spots (not good)
# same set items ("wooden-block ___", "ironwood ___", etc.)
# same shape items ("____ chair", "____ bed", etc.)
# crazy coincidences ("peach chair" and "beach chair" for example)
# closest match was 0.96 for Mr. and Mrs. Flamingo, congrats to the newlyweds
# this indicates a potential accuracy issue when using a large dictionary
# note also that ACNH capitalizes the first letter in each item in menus
# will need to account for list corrections

acnh_levmat_house = acnh_levmat[acnh_levmat < 1].sort_values(ascending=False)

if False: # this should approximate levmat if you don't have it for some reason
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

