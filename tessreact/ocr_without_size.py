import cv2
import pytesseract
import numpy as np

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image, convert to HSV, color threshold to get mask
image = cv2.imread('images/presc5.jpg')
# fromCenter = False
# showCrosshair = False
# r = cv2.selectROI(image, fromCenter, showCrosshair)
# imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0])
upper = np.array([200, 200, 200])
mask = cv2.inRange(hsv, lower, upper)


# Invert image and OCR
invert = 255 - mask
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
print(data)
cv2.imshow('mask', mask)
cv2.imshow('invert', invert)
cv2.waitKey()
cv2.destroyAllWindows()