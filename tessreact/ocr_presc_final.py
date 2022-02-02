import cv2
import pytesseract
import os
import numpy as np

imgQ = cv2.imread('new_image/template.jpeg')
h, w, c = imgQ.shape
imgQ = cv2.resize(imgQ, (w//3, h//3))
per = 25

roi = [[20,60,70,80]]

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)

# imgkp1 = cv2.drawKeypoints(imgQ, kp1, None)
# cv2.imshow("KeyPoint", imgkp1)

path ='new_image'
myPicList = os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path+"/"+y)
    img = cv2.resize(img, (w//3, h//3))
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches = list(matches)
    matches.sort(key=lambda x:x.distance)
    matches = bf.match(des2, des1)
    good = matches[:int(len(matches)*(per/100))]
    print(len(matches))
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:10],None,flags=2)
    cv2.imshow(y,imgMatch )



    


cv2.imshow("Output", imgQ)
cv2.waitKey(0)