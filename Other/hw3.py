# stupid method of doing a CIS 490 problem

import pandas as pd
import numpy as np

# INITIALIZE DATASET
try: 
    df = pd.read_csv('hw3_table.csv',index_col=0)
except: # this requires Tesseract! if you don't have it, you ain't runnin this
    print("Can't find 'hw3_table.csv', gonna try and make it! Ask Bri for help probably\n.")
    tess = False
    import pytesseract
    tess = True
    try:
        from PIL import Image, ImageDraw, ImageColor
    except ImportError:
        import Image

    hw = np.array(Image.open('hw3_table.png'))
    hw = hw.mean(axis=2) # lazy method of converting to grayscale
    rows = np.where(hw[:,1] == 0) # determine rows based on solid black lines
    cols = np.where(hw[1,:] == 0) # determine columns based on solid black lines

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #tesseract file path here
    results = []
    for x,i in enumerate(rows[:-1]):
        newList = []
        for y,j in enumerate(cols[:-1]):
            img = Image.fromarray(hw[i+1:rows[x+1],j+1:cols[y+1]]).convert('RGB') # tesseract isn't playing nice with grayscale
            s = pytesseract.image_to_string(img,config='--psm 7').strip()
            newList.append(s)
        results.append(newList)
    df = pd.DataFrame(results)
    df.columns = ("Name","GiveBirth","CanFly","LiveInWater","HaveLegs","Class")
    df.Class = df.Class.str.replace("[.:]","") # Tesseract picked up some mistakes on this column, this fixes em
    df.set_index("Name",inplace=True)
    df.to_csv('hw3_table.csv')


# GiveBirth, CanFly, LiveInWater, HaveLegs
#x = ["no","yes","yes","no"] # for homework 3
x = ["yes","no","no","yes"] # for exam 2
Ycol = 'Class' # variable to determine


# ACTUAL BAYES METHOD STARTS HERE
# could have this whole block be its own function
Xcol = df.columns[df.columns != Ycol].tolist()
Yvals = df[Ycol].unique().tolist()
N = len(df)
results = {}
# assumes len(x) == len(Xcol)
for y in Yvals:
    Xvals = df.loc[df[Ycol].str.match(y),Xcol]
    Yn = len(Xvals) 
    P = [len(Xvals.loc[Xvals[ci].str.match(xi)])/Yn for ci,xi in zip(Xcol,x)]
    P.append(Yn/N)
    results[y] = P
bayes = pd.DataFrame.from_dict(results,orient='index')
bayes.columns = ["P(%s=%s|y)"%(ci,xi) for ci,xi in zip(Xcol,x)] + ["P(y)"]
bayes['P(y|x)'] = bayes.product(axis=1) # product of all P(xi|y) values and P(y)
bayes.sort_values(by='P(y|x)',ascending=False,inplace=True)
likely = bayes.index[0]


# RESULTS HERE
'''
Naive Bayes:
P(yi | x1, x2, …, xn) = P(x1, x2, …, xn | yi) * P(yi)
P(yi | x) = P(x1|yi) * P(x2|yi) * … P(xn|yi) * P(yi)

Compare P(x|M)*P(M) vs. P(x|N)*P(N):

P(M) = 7/20
P(x|M) = 1/7 * 1/7 * 2/7 * 2/7
P(M|x) = 28/48020 = 1/1715 ~ 0.00058

P(N) = 13/20
P(x|N) = 12/13 * 3/13 * 3/13 * 4/13
P(N|x) = 5616/571220 = 108/10985 ~ 0.00983

Since P(N|x) > P(M|x), assume N (non-mammal)
'''
print(bayes)
print("\nMost likely %s from Naive Bayes: %s"%(Ycol,likely))


# binary or no? quick looksies
for col in df:
    if len(df[col].unique()) == 2:
        print('BINARY! we got two: %s\t%s'%(col,df[col].unique()))
    else:
        print('not binary (%d):\t%s\t%s'%(len(df[col].unique()),col,df[col].unique()))