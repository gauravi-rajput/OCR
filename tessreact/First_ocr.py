import cv2
import pytesseract

def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

# img = cv2.imread('download.png')
img = cv2.imread('images/presc1.jpg')

def get_grayscale(image):
    color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(color)
    return  color

def remove_noise(image):
    return cv2.medianBlur(image, 3)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img =  cv2.resize(img, (1000,1000), interpolation=cv2.INTER_LINEAR)
img = get_grayscale(img)
img = thresholding(img)
img = remove_noise(img)

cv2.imshow('result', img)
cv2.waitKey(0)

print(ocr_core(img))