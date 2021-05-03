import cv2
import numpy as np

# works well with border only, but any image can be used
img = cv2.imread('Other/bri_acnh_photo_cplus.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

img_copy1 = img.copy()
img_copy2 = img.copy()
lines1 = cv2.HoughLines(edges,1,np.pi/180,200)
for i in lines1:
	for rho,theta in i:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		cv2.line(img_copy1,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('acnh_hough1.jpg',img_copy1)



minLineLength = 100
maxLineGap = 10
lines2 = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for i in lines2:
	for x1,y1,x2,y2 in i:
		cv2.line(img_copy2,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('acnh_hough2.jpg',img_copy2)