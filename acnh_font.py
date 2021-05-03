import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt


import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

acfont = "FOT-Seurat Pro B.otf"


if False: # this doesn't work with .otf files, but here it is
    import pathlib
    from matplotlib import font_manager
    font_dirs = [str(pathlib.Path().absolute())]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = acfont

# set font
font = ImageFont.truetype(acfont,size=24)

if False: # method for looking at font printout
    img = Image.new('RGB', (300, 100), (255, 255, 255))
    #font = ImageFont.load("FOT-Seurat Pro B.otf")
    # sizes in the font file: 12,18,24,36,48,60,72
    s = "Acanthostega"
    d = ImageDraw.Draw(img)
    d.text((0,0), s, fill=(0, 0, 0), font=font)

    text_width, text_height = d.textsize(s,font=font)
    print(text_width,text_height)
    img = img.crop((0,0,text_width,text_height))
    img.show()
    
# ACNH resolution is 1920x1080 docked [16:9], 1280x720 portable [4:3]
# 12MP camera resolution is 4032x3024 [4:3], about 3.15 (or 20/63) times larger than portable
# can add 360 blackspace to a docked image (180 on top and bottom) to match 4:3 resolution (widescreen)
#   Nintendo Switch firmware update version 11.0.0 (Nov 30, 2020): 
#   Users can now transfer screenshots and videos from Album to their smart devices.
# (accessibility for clean images is way way way up! kinda puts this whole thing in perspective)
# (but if we're aiming for maximum convenience, we still want a platform entirely on smartphones)

# the sample image acnh.jpg is 1280x720 (portable)
# docked extends 50% larger (1920/1280 = 1080/720 = 1.5), or portable is 2/3 the size, whatever
# in MSPaint, the original text seems to closest match when set to font size 20


# for a character set, look at 0x21 ("!") to 0x7e ("~")

# need to do two things:
# one, find out the max dimensions for all printed characters
charset = []
charrange = range(0x21,0x7f) # 94 characters
w,h = 0,0
img = Image.new('L', (1,1), 255)
d = ImageDraw.Draw(img)
doot = []
for i in charrange:
    d.text((0,0), chr(i), fill=0, font=font)
    nw,nh = d.textsize(chr(i),font=font)
    w,h = max(nw,w),max(nh,h)
    #doot.append(np.array([
    charset.append(chr(i))
print("Dimensions: Width=%d,Height=%d"%(w,h))
imgset = []
charlen = len(charset)
characc = np.zeros([charlen,charlen])
for s in charset:
    img = Image.new('L', (w, h), 255)
    d = ImageDraw.Draw(img)
    d.text((0,0), s, fill=(0), font=font)
    charimage = np.array(img).astype(int)
    j = len(imgset)
    imgset.append(charimage)
    for i,v in enumerate(imgset[0:-1]):
        newacc = 1 - np.abs(charimage - v).mean()/255
        characc[i,j] = newacc
        characc[j,i] = newacc
    characc[j,j] = 1
    
df = pd.DataFrame(characc, columns=charset, index=charset)
#display(df.mean().sort_values().head(47))
#display(df.mean().sort_values().tail(47))
meandist = df.mean().mean()
print("Mean distance between basic character set: %f"%meandist)
lm = df.stack()
lm = lm[lm < 1].sort_values(ascending=False)


# basic heat map
# plt.imshow(characc, cmap='hot', interpolation='nearest')
# plt.show()

# better heatmap
fig, ax = plt.subplots()
im = ax.imshow(characc, cmap='hot', interpolation='nearest')

tickrange = np.arange(charlen)
ax.set_xticks(tickrange)
ax.set_yticks(tickrange)

ax.set_xticklabels(charset)
ax.set_yticklabels(charset)

plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

csfont = {'fontname':'FOT-Seurat Pro B'}
ax.set_title("Character Distance",**csfont)
fig.tight_layout()
plt.show()



# can pull index of df.loc[charlist] for possible dictionary characters
# then get max value for each based on what's present
# for MHGU: df.loc[['o','-'],['a','d','e','>']]
# can also weigh with other fonts to estimate best match for any given char:
# probably rely on most common fonts
# HOWEVER! must consider variations in character shapes:
#  lowercase a, curved lowercase t, 0 with or without line/dot, open or closed 4, etc.
#  also serifs, italics, cursive patterns, and accents like umlauts and what have you
#  also also fonts with borders and fills
#  may be helpful to have a way for users to define general font attributes if unable to provide font object
#  would be helpful to understand spacing/kerning between adjacent characters
#  should also be able to provide simplified character sets, IE replace dash with hyphen, remove accents, non-directional quotes
