# this is largely copy-pasted from the other mhgu.py script
# use this one to look at threshold accuracy on mhgu3.png

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from levenshtein import levenshtein_ratio_and_distance as lev
from levenshtein import levmat

try:
    from PIL import Image, ImageDraw, ImageColor
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


try:
    image = img['inv_mhgu3.png'].copy()
except:
    try:
        image = Image.open('mhgu3.png')
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = Image.fromarray(np.array(image) ^ 255)
    except:
        print("Error loading the image, no good! Try stuff!")

show_levmats = False

mhgu = {0: ["Sheathing","+10","Snowbaron X","+1","oo-"]}

mhgu_minus = [str(x) for x in list(range(-10,0))]
mhgu_plus = ['+'+str(x) for x in list(range(1,14))]
mhgu_pm = mhgu_minus+mhgu_plus

mhgu_slot = ['---','o--','oo-','ooo']

fileName = 'mhgu_skill.csv' #glad I made this a while ago
try:
    mhgu_skill = pd.read_csv('mhgu_skill.csv')['SKILL'].tolist()
    print("Loaded data from '%s' successfully."%fileName)
except:
    
    print("Can't find '%s', downloading data from Kiranico."%fileName)

    import requests
    from bs4 import BeautifulSoup
    
    header = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0",
      "X-Requested-With": "XMLHttpRequest"
    }
    
    r = requests.get("https://mhgu.kiranico.com/skill/", headers=header)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.findAll('table')[10] #assumes the structure of the site hasn't changed; turn this into a loop if it has
    mhgu_skill = []
    for tr in table.findAll("tr"):
        trs = tr.findNext("td")
        if type(trs.get('rowspan')) != type(None):
            print(trs.text.strip())
            mhgu_skill.append(trs.text.strip())
    df = pd.DataFrame(mhgu_skill, columns=["SKILL"])
    df.to_csv(fileName, index=False)
    print("Saved %s to directory."%fileName)


# levenshtein matrix!
# compare each term in mhgu_skill with each other: 205 items, so 20910 pairs
# can also create lists of each WORD by itself:
# one split by spaces
# one split by any non-character

# these are sets, not dicts
mhgu_space = set()
mhgu_punct = set()
for s in mhgu_skill:
    for w in s.split(): #split on space
        mhgu_space.add(w)
    for w in re.split("\W",s): #split on non-alphanumeric/underscore
        mhgu_punct.add(w)
mhgu_punct.remove("") # re.split adds empty strings too
mhgu_space = list(mhgu_space)
mhgu_punct = list(mhgu_punct)

# 205 in original
# 201 in space
# 202 in punct
s1 = len(mhgu_skill)
s2 = len(mhgu_space)
s3 = len(mhgu_punct)

mhgu_levmat_skill = np.eye(s1)
mhgu_levmat_space = np.eye(s2)
mhgu_levmat_punct = np.eye(s3)

if show_levmats:
    print("Starting Levenshtein matrices...")
    for i in range(0,s1):
        for j in range(i+1,s1):
            mhgu_levmat_skill[i,j] = lev(mhgu_skill[i],mhgu_skill[j],True)
    print("Done with full skill names!")

    for i in range(0,s2):
        for j in range(i+1,s2):
            mhgu_levmat_space[i,j] = lev(mhgu_space[i],mhgu_space[j],True)
    print("Done with space-delimited skill words!")

    for i in range(0,s3):
        for j in range(i+1,s3):
            mhgu_levmat_punct[i,j] = lev(mhgu_punct[i],mhgu_punct[j],True)
    print("Done with punctuation-delimited skill words!")



    # view of top items
    lm1 = pd.DataFrame(mhgu_levmat_skill, columns=mhgu_skill, index=mhgu_skill).stack()
    lm2 = pd.DataFrame(mhgu_levmat_space, columns=mhgu_space, index=mhgu_space).stack()
    lm3 = pd.DataFrame(mhgu_levmat_punct, columns=mhgu_punct, index=mhgu_punct).stack()

    lm1[lm1 < 1].sort_values(ascending=False).head(50)
    lm2[lm2 < 1].sort_values(ascending=False).head(50)
    lm3[lm3 < 1].sort_values(ascending=False).head(50)
    #np.max(mhgu_levmat_skill - np.eye(s1))
    # 0.916 or 11/12
    #np.max(mhgu_levmat_space - np.eye(s2))
    # 0.875 or 7/8
    #np.max(mhgu_levmat_punct - np.eye(s3))
    # 0.909 or 10/11

    # symmetric matrices for plotting
    mhgu_levsym_skill = mhgu_levmat_skill + mhgu_levmat_skill.T - np.eye(s1)
    mhgu_levsym_space = mhgu_levmat_space + mhgu_levmat_space.T - np.eye(s2)
    mhgu_levsym_punct = mhgu_levmat_punct + mhgu_levmat_punct.T - np.eye(s3)

    sym1 = pd.DataFrame(mhgu_levsym_skill, columns=mhgu_skill, index=mhgu_skill)
    sym2 = pd.DataFrame(mhgu_levsym_space, columns=mhgu_space, index=mhgu_space)
    sym3 = pd.DataFrame(mhgu_levsym_punct, columns=mhgu_punct, index=mhgu_punct)

    #plt.imshow(sym1, cmap='hot', interpolation='nearest')
    #plt.show()

    sym = [sym1,sym2,sym3]
    names = ['skill','space','punct']


    for i in sym:
        cz = i[i <= 0].count().sum()
        mm = i[i > 0].mean().mean()
        plt.imshow(i, cmap='hot', interpolation='nearest')
        plt.show()


    lens1 = []
    lens2 = []
    lens3 = []
    for i in mhgu_skill:
        for j in mhgu_skill:
            lens1.append(len(i)+len(j))
    for i in mhgu_space:
        for j in mhgu_space:
            lens2.append(len(i)+len(j))
    for i in mhgu_punct:
        for j in mhgu_punct:
            lens3.append(len(i)+len(j))

# super evaluation method starts here!

# split positions for each column
starts=[0,112,150,262,301]
ends=[111,149,261,300,359]
conf_skill = '--psm 7'
conf_pm = '--psm 7' #-c tessedit_char_whitelist=0123456789+-–'
conf_slot = '--psm 7'# -c tessedit_char_whitelist=Oo-–'
configs=[conf_skill,conf_pm,conf_skill,conf_pm,conf_slot]

# results dictionaries; look at these when you're done!
# testX is the OCR output as is
# testY is the OCR output corrected via dictionary
# testdet is a blob of details, have fun with that
testX = {}
testY = {}
testdet = {}
ims = {}

# imageArray is just the image in Numpy array format
#imageArray = cv2.fastNlMeansDenoisingColored(np.array(image),None,10,10,7,21)
imageArray = np.array(image)

for X in range(1,255):
    print("Starting image at threshold of %d..."%X)
    doot = (((imageArray > X) > 0)*255).astype('uint8')
    d = {}
    dy = {}
    for i in range(1):
        print("\tRow %d [ "%i,end='')
        d[i] = {}
        dy[i] = {}
        for j in range(5):
            val = pytesseract.image_to_string(doot[i*32:31+i*32,starts[j]:ends[j],:], config=configs[j])[:-2]
            d[i][j] = val
            print(j,end=' ')
        print(']')
                #print("%s: %s"%((i,j),d[i][j]))
        testX[X] = d
        testY[X] = dy

#S_skills = [mhgu_skill, mhgu_space, mhgu_punct]

# acc is an array of the lev distance between the predicted (after dictionary) results vs. actual values
acc = np.zeros([len(testX),1,5])
xacc = np.zeros([len(testX),1,5])
xk = 0
for k,v in testX.items(): # k is the index, v is the table
    for i in range(1): # row
        for j in range(5): # col
            orig = v[i][j]
            real = mhgu[i][j]
            if orig == '':
                #print("Empty string at %s from %s, moving on."%((i,j),_))
                maxstr = ""
                maxval = 0.0
                truelev = 0.0
            else:

                if j == 4: # slots
                    S = mhgu_slot
                    _ = "slot"
                elif j % 2 == 1: # plus-minus
                    S = mhgu_pm
                    _ = "pm"
                    orig = orig.replace("—","-")
                else: # skill name
                    S = mhgu_skill
                    _ = "skill"
                    orig = orig.replace("—","-")
                #pd.Series({v: lev("Shorpnes",v,True) for v in mhgu_skill})
                #print("Getting %s [%s] from %s."%((i,j),orig,_))
                maxval = 0
                maxstr = ""
                for x in S:
                    newval = lev(x,orig,True)
                    if newval > maxval:
                        maxval = newval
                        maxstr = x
                    #elif newval == maxval and maxval > 0 and x != maxstr:
                        #print("Tie detected for [%s]\t[%s] vs. [%s] (%d match)"%(orig,x,maxstr,maxval))
                    if maxval == 1:
                        break
                if maxstr == '':
                    truelev = 0.0
                else:
                    truelev = lev(real,maxstr,True)
                acc[xk][i][j] = truelev
                xacc[xk][i][j] = maxval
            testY[k][i][j] = maxstr
            testdet[(k,i,j)] = [orig,maxstr,maxval,real,truelev]
            if truelev < 1:
                print("FAILURE at [%d][%d][%d]: %s"%(k,i,j,[orig,maxstr,maxval,real,truelev]))
                            
    xk=xk+1 #I'm so lazy


def printem(i=0,j=0):
    print("Chars at (%d,%d):"%(i,j))
    print("(parsed)\t[%s]\n-----------------------"%mhgu[i][j])
    for k,v in testX.items():
        print("[%s]\t[%s]"%(v[i][j],testY[k][i][j]))
    
for k,vk in testX.items():
    for i,vi in vk.items():
        for j,vj in vi.items():
            if vj == '':
                print("Blank at (%d,%d,%d)"%(k,i,j))

# Now we finally decide which result to go with based on the most commonly returned result!
# lazy method: https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
# returns the "first" most common item entered if there's a tie
# most_frequent() DOES accept blank strings (or really any type), but predict doesn't 
# ex1: predict on a list of ["a","b","c","a","b","","",""] returns "a"
# ex2: predict on a list of ["b","a","c","a","b","","",""] returns "b"
# ex3: most_frequent on either of the above two lists returns ""
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def predict(i=0,j=0):
    List = []
    for k,v in testX.items():
        s = testY[k][i][j]
        if s != '':
            List.append(s)
    if List:
        return most_frequent(List)
    return ""

def predictX(i=0,j=0):
    List = []
    for k,v in testX.items():
        s = testX[k][i][j]
        if s != '':
            List.append(s)
    if List:
        return most_frequent(List)
    return ""

predicted = {} # final result list
predictedX = {} # without
predAcc = np.zeros([1,5]) # accuracy of results
predAccX = np.zeros([1,5]) # without again

for i in range(1): # row
    l = []
    lx = []
    for j in range(5): # col
        p = predict(i,j)
        l.append(p)
        predAcc[i,j] = lev(p,mhgu[i][j],True)
        px = predictX(i,j)
        lx.append(px)
        predAccX[i,j] = lev(px,mhgu[i][j],True)
    predicted[i] = l
    predictedX[i] = lx


x = np.arange(1,255)
y = np.reshape(xacc,(254,5))

y2 = (y > 0)*1


plt.figure(1,figsize=(13, 6))
plt.plot(x,y[:,0],'b-',lw=1,label='"Sheathing"')
plt.plot(x,y[:,1],'g-',lw=1,label='"+10"')
plt.plot(x,y[:,2],'c-',lw=1,label='"Snowbaron X"')
plt.plot(x,y[:,3],'y-',lw=1,label='"+1"')
plt.plot(x,y[:,4],'r-',lw=1,label='"oo-"')
plt.plot(x,y.mean(axis=1),'k-',lw=2,label='Average')

plt.legend(loc='best')
plt.title('"mhgu3.png" Threshold vs. Accuracy Plot')  # LATEX equations!
plt.xlabel("Threshold")
plt.ylabel("Accuracy")

a = 10
#plt.xticks([0,255]+list(range(a,255,a)))
plt.xticks(list(range(a,254,a)))
plt.xlim(1,254)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("mhgu_threshplot.png")

plt.show()


# predAcc.mean() 
# returns 0.9296296296296297 when using whitelist method


# use acc.mean(axis=0) for measuring accuracy
# axis=0: across different cutoffs
# axis=1: across different rows (talismans in MHGU's case)
# axis=2: across different columns


# RESULTS!
# Total acc: 0.8788359788359787
'''
array([[1.        , 0.57142857, 1.        , 1.        , 0.42857143],
       [0.71428571, 0.85714286, 1.        , 1.        , 1.        ],
       [1.        , 0.71428571, 1.        , 1.        , 0.66666667],
       [1.        , 1.        , 1.        , 1.        , 1.        ],
       [1.        , 0.5       , 1.        , 1.        , 0.66666667],
       [0.42857143, 1.        , 1.        , 1.        , 0.        ],
       [1.        , 1.        , 1.        , 1.        , 0.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ]])
'''

# skill names:
# 2 cases where "Blunt" (2nd row, 1st col) had an empty string: 175, 176
# 4 cases where "Sheathing" (1st row, 6 col) had an empty string: 170, 171, 172, 173
# one case where "B" is detected for "Blunt" but was correct because "Blunt" comes before "Brawn" in list

# numeric values:
# 6 cases where "+5" (1st row, 2nd col) returned -5 from 45: all but 170
# 1 case where "+1" (2nd row, 2nd col) returned -1 from #1: 173
# 4 cases where "+5" (3rd row, 2nd col) returned -5 from $5: 173,

# slot values:
# all "oo-" were correct, everything else was wrong
# didn't test "---"

# cols 3 and 4 were perfect!
# rows 8 and 9 were perfect!

# biggest issue seems to be identifying symbols!

# PLUS-MINUS:
# # should be +
# 4 should be +
# $ should be +
# a missed sign that should've been a + (row 5 col 2)
# a missed sign that should've been a - (row 3 col 4)
# an H that should've been a 1 (row 4 col 4)
# a “ (left quote) that should've been a - (row 5 col 4)

# SLOTS:
# "d" and "e" consistently recognized in place of circles
# "<" and "—" for "-" in "oo-", with "<" being waay more common; all correct
# single character return in cases of "o--": "a", "i", "—"
# "---" was completely untested
# may need to assume a length match may be as accurate, possibly more! 
# this is true for single slots and "eee", as well as one of the doubles as "oo"



# Denoising showed improvements at some locations, but worse in others

# Total acc: 0.8687389029685947
'''
accDn = np.array([
       [0.75963719, 1.        , 0.21456583, 1.        , 0.        ],
       [1.        , 0.28571429, 1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        , 1.        , 0.66666667],
       [1.        , 1.        , 1.        , 1.        , 1.        ],
       [1.        , 0.5       , 1.        , 1.        , 0.66666667],
       [1.        , 1.        , 1.        , 1.        , 0.        ],
       [1.        , 1.        , 1.        , 1.        , 0.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ],
       [1.        , 1.        , 1.        , 1.        , 1.        ]])
'''
# This is likely due to the noise process effectively blurring the results
# of an already relatively clean image (though the original is off due to jpg compression).
# We can use the denoise function on blank strings though! This could help


# CHANGE BY POSITION:
# positive: normal better; negative: denoise better
'''
array([[ 0.24, -0.43,  0.79,  0.  ,  0.43],
       [-0.29,  0.57,  0.  ,  0.  ,  0.  ],
       [ 0.  , -0.29,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [-0.57,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ]])
'''
# in most cases, no difference
# gains on first column (for blank strings), second column (handling +- in some instances)
# losses on first row in general (aside from second column) and row2 col2

# suggested use is to apply only on a blank string
# for slots, similar results: < and d appearing regularly in place of - and o
# a single "a" for single slots
# no "e" ever came up

# plus/minus:
# perfect score at 0,1
# interpreted + as 4 at 1,1; lead to -4 return
# no sign picked up at 4,1; lead to -9 instead of +9
# interpreted 1 as blank or H in 4,4

# based on each dictionary, there should be a minimum length for ANY result

# SO!
# we could re-examine a word or even a single character based on the following:
# -- length is shorter than minimum length possible in dictionary
# -- multiple results had the same lev distance

# we can also speed up the dictionary process for pre-matched values
# HOWEVER! we'd still want distinct OCR outputs from distinct real values
# for example: if we ONLY see "ood" from "ooo", that's fine
# but if we're seeing "ood" generated on both "oo-" and "ooo", then we've got a problem

# reminder of the general idea: CONSISTENCY!
# we don't care if the results are perfect if we can consistently match them via dictionary
# we also need to assume that there's no one single perfect method for any image
#    let alone ALL images




#mhgu_unique = pd.Series(pd.DataFrame(mhgu).values.flatten()).unique()

# 45 items normally, 24 unique

# SAMPLE FOR TESTING DENOISE WITH STAR WARS BACKGROUND:
#k = Image.open("esb.png")
#newk = np.array(k)
#newk3 = cv2.fastNlMeansDenoisingColored(newk,None,10,10,7,21)
#newk3 = ((newk3.std(axis=2) > newk3.std(axis=2).mean())*255).astype('uint8')
#Image.fromarray(newk3).show()
#newk2 = ((newk.std(axis=2) > newk.std(axis=2).mean())*255).astype('uint8')
#newk2 = np.array(Image.fromarray(newk2).convert('RGB'))
#newk2 = cv2.fastNlMeansDenoisingColored(newk2,None,10,10,7,21)
#Image.fromarray(newk2).show()

# can see how correcting BEFORE applying the threshold removes fewer stars
# but it also has less bleeding between adjacent characters

# I'll include the download link for this when I clean it up


