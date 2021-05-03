# Yanlai notes py code (be sure to thank him!):

import numpy as np
from levenshtein import levenshtein_ratio_and_distance as lev

# OCR interpretation of Yanlai's MTH 361 notes
a = ["Chord secant and Navon's adeterods",
"Idea usemoreinformationthan just the sign ofthefaaetimf",
"ego fexch f exam",
"ReviewTaylor Serves Expansive",
"approximating afractionfun around some fixed point x"]

# human interpretation
b = ["② Chord, Secant, and Newton’s methods",
"• Idea: use more information than just the sign of the function f",
"e.g. f(x(k)), f’(x(k)) …",
"Review Taylor Series Expansion:",
" approximating a function f(x) “around” some fixed point x"]

acc1 = np.zeros(5)
lens = np.zeros(5)
acc2 = np.zeros(5)
for i in range(5):
    acc1[i] = lev(a[i],b[i],True)
    lens[i] = len(a[i])+len(b[i])

print("comparing two strings with levenshtein distance:\n------")
for s in a:
    print(s)
print("------")
for s in b:
    print(s)

print("------\nindependent accuracies:")
print(acc1)
print(acc1.mean())

tlen = lens.sum()
for i in range(5):
    acc2[i] = acc1[i]*lens[i]/tlen

print("------\nweighted accuracies:")
print(acc2)
print(acc2.mean())