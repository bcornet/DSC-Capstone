import pandas as pd
import re
import matplotlib.pyplot as plt
from levenshtein import levenshtein_ratio_and_distance as lev

mhgu = {
0: ["Sharpness","+5","Critical Up","+4","ooo"],
1: ["Blunt", "+2", "Critical Up", "+5","oo-"],
2: ["Crit Draw", "+5", "Chain Crit", "-1", "ooo"],
3: ["Sheathing","+10","Snowbaron X","+1","oo-"],
4: ["Sheathing","+9","Sharpness","-4","ooo"],
5: ["Sheathing","+5","Expert","+10","o--"],
6: ["Sheathing","+5","Tenderizer","+4","o--"],
7: ["Sheathe Sharpen","+10","Dreadqueen X","+2","oo-"],
8: ["Sheathe Sharpen","+7","Expert","+8","oo-"]
}

mhgu_minus = [str(x) for x in list(range(-10,0))]
mhgu_plus = ['+'+str(x) for x in list(range(1,14))]
mhgu_pm = mhgu_minus+mhgu_plus

mhgu_slot = ['---','o--','oo-','ooo']

fileName = 'mhgu_skill.csv'
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

starts=[0,112,150,262,301]
ends=[111,149,261,300,359]

testX = {}
testY = {}
testdet = {}
for X in range(170,177):
    doot = (((np.array(img['inv_mhgu2.png'].copy()) > X) > 0)*255).astype('uint8')
    d = {}
    dy = {}
    for i in range(9):
        d[i] = {}
        dy[i] = {}
        for j in range(5):
            d[i][j] = pytesseract.image_to_string(doot[i*32:31+i*32,starts[j]:ends[j],:], config='--psm 7')[:-2]
            print("%s: %s"%((i,j),d[i][j]))
    testX[X] = d
    testY[X] = dy

#S_skills = [mhgu_skill, mhgu_space, mhgu_punct]

acc = np.zeros([len(testX),9,5])
xk = 0
for k,v in testX.items(): # k is the index, v is the table
    for i in range(9): # row
        for j in range(5): # col
            orig = v[i][j]
            real = mhgu[i][j]
            if orig == '':
                #print("Empty string at %s from %s, moving on."%((i,j),_))
                maxstr = 0.0
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
    


# RESULTS!
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
