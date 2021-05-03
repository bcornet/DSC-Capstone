import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import re
import cv2
import levenshtein

try:
    from PIL import Image, ImageDraw, ImageColor
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#k = Image.open('test.png')
#s = pytesseract.image_to_string(k)
#print(s)

L = ['test.png','fft.png','acnh.png','acnh.jpg','mhgu.jpg','acnh2.png','acnh3.png','mhgu2.png','mhgu3.png',
'mhgu3-1.png','mhgu3-2.png','mhgu3-3.png','mhgu3-4.png','mhgu3-5.png',"donghui.jpg","donghui_base.png"]
invL = []
img = {}
text = {}

for i in L:
    print('========%s======='%i)
    try:
        img[i] = Image.open(i)
        if img[i].mode == 'RGBA':
            img[i] = img[i].convert('RGB')
    except:
        print("ERROR: Can't find file '%s' probably"%i)
        print('========done!========')
        L.remove(i)
        continue
    j = 'inv_'+i
    invL.append(j)
    img[j] = Image.fromarray(np.array(img[i]) ^ 255)
    text[i] = pytesseract.image_to_string(img[i])
    text[j] = pytesseract.image_to_string(img[j])
    print(text[i])
    print('========%s======='%j)
    print(text[j])
    print('========done!========')
print()
L = L + invL

chars = {}
for i in L:
    chars[i] = pytesseract.image_to_boxes(img[i])
    print('========%s======='%i)
    print(chars[i])
    print('========done!========')
print()

words = {}
for i in L:
    words[i] = pytesseract.image_to_data(img[i])
    print('========%s======='%i)
    print(words[i])
    print('========done!========')
print()


pages = {}
for i in L:
    print('========%s======='%i)
    try: 
        pages[i] = pytesseract.image_to_osd(img[i])
        print(pages[i])
    except:
        print("no good!")
    print('========done!========\n')
print()
   



df = pd.concat( [pd.DataFrame.from_dict(text,orient='index'),
                pd.DataFrame.from_dict(chars,orient='index'),
                pd.DataFrame.from_dict(words,orient='index'),
                pd.DataFrame.from_dict(pages,orient='index')],axis=1)
df.columns = ['string','boxes','data','osd']

n = 'acnh2.png'

# wH = words[n].split('\n')[0].split('\t')
wH = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
 'left', 'top', 'width', 'height', 'conf', 'text']

W = pd.DataFrame(data=[i.split('\t') for i in words[n].split('\n')[1:-1]])
W.columns=wH

def showChars(n = 'test.png',color = 'blue'):
    global chars,img
    if n in L:
        k = img[n].copy()
    elif type(n) == Image:
        k = Image.open(n)
    else:
        print("I dunno")
        return
    test = ImageDraw.Draw(k)
    c = ImageColor.getcolor(color,'RGB')
    h = k.height
    S = [i.split(' ') for i in chars[n].split('\n')[0:-1]]
    for i in S:
        t = [int(j) for j in i[1:-1]]
        print(i,"pos at %d,%d,%d,%d"%(t[0],h-t[1],t[2],h-t[3]))
        test.rectangle((t[0],h-t[1],t[2],h-t[3]),outline=c)
    k.show()

    
def showWords(n = 'test.png',color = 'blue'):
    global words
    if n in L:
        k = img[n].copy()
    elif type(n) == Image:
        k = Image.open(n)
    else:
        print("I dunno")
        return
    test = ImageDraw.Draw(k)
    c = ImageColor.getcolor(color,'RGB')
    h = k.height
    S = data=[i.split('\t') for i in words[n].split('\n')[1:-1]]
    for i in S:
        t = [int(j) for j in i[6:10]]
        print(i)
        #test.rectangle((t[0],h-t[1],t[2],h-t[3]),outline=c)
        test.rectangle((t[0],t[1],t[0]+t[2],t[1]+t[3]),outline=c)
    k.show()

#2**np.array(range(8))

def bitplanes(n = 'test.png', csplit = True):
    if csplit:
        p = colorsplit(n)
    else:
        p = np.array(img[n])
    bp = []
    for i in range(8):
        bp.append(((p & 2**i) > 0)*255)
    return np.concatenate(bp).astype('uint8')
    
def colorsplit(n = 'test.png'):
    p = np.array(img[n])
    if p.shape[2] == 4:
        p = p[:,:,0:3]
    return p.reshape((p.shape[0],p.shape[1]*3),order='F').astype('uint8')

def gray(n = 'test.png'):
    p = np.array(img[n])
    return np.dot(p[...,:3], [0.299, 0.587, 0.144])
    
def bitplanes1(n = 'test.png'):
    p = bitplanes(n)
    k = Image.fromarray(p)
    s = pytesseract.image_to_string(k)
    S = [p]
    fig = plt.figure()
    for i, a in enumerate(S):
        ax = fig.add_subplot(1, 1, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(n+' Combined Bitplane', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    
def bitplanes3(n = 'test.png'):
    p = bitplanes(n)
    S = np.hsplit(p,3)
    #titles = ['R__ 11111111','_G_ 11111111','__B 11111111']
    titles = ['Red','Green','Blue']
    #colors = ['Red','Green','Blue']
    fig = plt.figure()
    for i, a in enumerate(S):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        #ax.set_xlabel(colors[i],fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()

def bitplanes8(n = 'test.png'):
    p = bitplanes(n)
    S = np.vsplit(p,8)
    titles = ['1st','2nd','3rd','4th','5th','6th','7th','8th']
    colors = ['Red','Green','Blue']
    bits = ['00000001','00000010','00000100','00001000','00010000','00100000','01000000','10000000']
    hex = ['0x01','0x02','0x04','0x08','0x10','0x20','0x40','0x80']
    fig = plt.figure()
    for i, a in enumerate(S):
        ax = fig.add_subplot(8, 1, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        #ax.set_title(titles[i], fontsize=10)
        ax.set_ylabel(hex[i],fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()



def bitplanes24(n = 'test.png', csplit=True):
    p = bitplanes(n,csplit)
    if csplit:
        numsplit = 3
    else:
        numsplit = 1
    S = np.vsplit(p,8) # S is 8 bitplanes for the three colors stacked
    titles = ['1st','2nd','3rd','4th','5th','6th','7th','8th']
    colors = ['Red','Green','Blue']
    bits = ['00000001','00000010','00000100','00001000','00010000','00100000','01000000','10000000']
    hex = ['0x01','0x02','0x04','0x08','0x10','0x20','0x40','0x80']
    fig = plt.figure(figsize=([12,4.5]))
    dat = {}
    #checkem = n == 'acnh2.png'
    checkem = 'mhgu3-3' in n
    if checkem:
        #compare = 'Acanthostega\nAmber\nAmmonite\nAnkylo skull\nAnkylo tail\nAnkylo torso\nAnomalocaris\nArchelon skull\nArchelon tail'.split("\n")
        compare = ['Snowbaron X']
    for i, a in enumerate(S):
        V = np.hsplit(a,numsplit) # V is one of the 24 individual bitplanes
        dat[i] = {}
        for j, b in enumerate(V): # bitplane order: R[0...7],G[0...7],B[0...7]
            ind = i + j*8
            ax = fig.add_subplot(numsplit, 8, ind + 1)
            ax.imshow(b, interpolation="nearest", cmap=plt.cm.gray)
            if i == 0:
                ax.set_ylabel(colors[j], fontsize=10)
            if j == 0:
                ax.set_title(hex[i],fontsize=10)
            s = pytesseract.image_to_string(Image.fromarray(b))
            
            #ax.set_xlabel('%s: [%s]'%(format(2**ind,'024b'),s),fontsize=6)
            
            print('%s:\n%s\n========='%((i,j),s))
            dat[i][j] = s
            if checkem:
                s = re.sub('\n\x0c','',s)
                v = 0
                if len(re.sub('\s','',s)) == 0:
                    print('0 lines (blank)\n')
                else:
                    words = s.split('\n')
                    if words == compare:
                       v = 1
                    else:
                        WORDS = {}
                        print(len(words),"lines,",end=' ')
                        for w in words:
                            w = w.strip()
                            if len(w) == 0:
                                continue
                            WORDS[w] = {}
                            for z in compare:
                                WORDS[w][z] = levenshtein_ratio_and_distance(z,w,True)
                            
                        print(len(WORDS),"words")
                        #W = np.array(W)
                        #v = W.sum()/9
                        df = pd.DataFrame.from_dict(data=WORDS).stack()
                        COMP = compare.copy()
                        WORDS = list(WORDS)
                        print(df)
                        print("LET'S TRY!")
                        while COMP and WORDS:
                            print(COMP)
                            print(WORDS)
                            p = df.loc[(COMP,WORDS)].sort_values(ascending=False).head(1)
                            print('p:\n',p)
                            pcomp,pwords = p.index[0]
                            COMP.remove(pcomp)
                            WORDS.remove(pwords)
                            v = v + p[0]
                        v = v / len(compare)    
                            #v = levenshtein_ratio_and_distance(compare,s,True)
                s = '{:.3%}'.format(v)
            else:
                s = re.sub('\s','',s)
                if len(s) == 0:
                    s = '(no text)'
                else:
                    s = 'TEXT FOUND!'
            ax.set_xlabel(s,fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    return dat


def highbits(n = 'test.png'):
    p = bitplanes(n)
    S = np.vsplit(p,8)
    S = np.hsplit(S[-1],3)
    #titles = ['1st','2nd','3rd','4th','5th','6th','7th','8th']
    colors = ['Red','Green','Blue']
    #bits = ['00000001','00000010','00000100','00001000','00010000','00100000','01000000','10000000']
    hex = ['0x01','0x02','0x04','0x08','0x10','0x20','0x40','0x80']
    fig = plt.figure()
    for i, a in enumerate(S):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title('%s %s'%(colors[i],hex[-1]), fontsize=10)
        s = pytesseract.image_to_string(Image.fromarray(a))
        #ax.set_xlabel('%s: [%s]'%(format(2**ind,'024b'),s),fontsize=6)
        print('%5s:\n%s\n========='%(colors[i],s))
        s = re.sub('\s','',s)
        if len(s) == 0:
            s = '(no text)'
        ax.set_xlabel(s,fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()

import pywt.data
def dwt_ex():
    # Load image
    original = pywt.data.camera()

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', 'Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 8))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()



def dwtgray(n = 'test.png'):
    dwt(gray(n))

def dwtbit(n = 'test.png'):
    dwt(bitplanes(n))
    
def dwtsplit(n = 'test.png'):
    dwt(colorsplit(n))

def dwtsplitB(n = 'test.png',i=0):
    i = max(0,min(i,2))
    p = colorsplit(n)
    r = p.shape[1]/3
    print(p.shape)
    c = (np.arange(r)+i*r).astype(int)
    print(i*r,i*r+r)
    dwt(colorsplit(n)[:,c])

#k = Image.fromarray(bitplanes('mhgu3.png'))
#s = pytesseract.image_to_string(k)
def dwt(p):
    titles = ['Approximation', 'Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(p, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure()
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(2, 2, i + 1 )
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()



# bitplane method works great for resolving ACNH table


# METHODS:
# bitplane for RGB
# bitplane for grayscale

# monochrome

# dwt for RBG
# dwt for grayscale






def denoise(n):
    global img
    if n in L+invL:
        k = img[n].copy()
    elif type(n) == str:
        k = Image.open(n)
    else:
        print("I dunno")
        return
    #k = np.array(k)
    newk = cv2.fastNlMeansDenoisingColored(np.array(k),None,10,10,7,21)
    # Doc from Image Denoising in OpenCV:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
    # h : parameter deciding filter strength. Higher h value removes noise better, but removes details of image also. (10 is ok)
    # hForColorComponents : same as h, but for color images only. (normally same as h)
    # templateWindowSize : should be odd. (recommended 7)
    # searchWindowSize : should be odd. (recommended 21)
    Image.fromarray(newk).show()
    return newk









def threshold(img):
    
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean = img.mean()
    if mean < 100:
        img = 255-img
        
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 101, 21)
                                 



