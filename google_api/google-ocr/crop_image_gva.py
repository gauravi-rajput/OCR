from http import client
import os, io
from google.cloud import vision
# from google.cloud.vision import types
import pandas as pd
import cv2

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'serviceAcoount.json'
client = vision.ImageAnnotatorClient()

FOLDER_PATH = r'/home/gauravirajput/TPN/train-ocr/google-ocr/images/'
IMAGE_FILE = 'presc43.jpg'
FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)
image = cv2.imread(FILE_PATH)
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
image_resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
fromCenter = False
showCrosshair = False
r = cv2.selectROI(image_resize, fromCenter, showCrosshair)
imCrop = image_resize[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# scale_percent = 220 # percent of original size
# width = int(imCrop.shape[1] * scale_percent / 100)
# height = int(imCrop.shape[0] * scale_percent / 100)
# dim = (width, height)
  
# # resize image
# resized = cv2.resize(imCrop, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite(f"/home/gauravirajput/TPN/train-ocr/google-ocr/cropped_image/{IMAGE_FILE}",imCrop)
image_path = f"/home/gauravirajput/TPN/train-ocr/google-ocr/cropped_image/{IMAGE_FILE}"
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content = content)
response = client.document_text_detection(image=image)
docText = response.full_text_annotation.text
print(docText)
pages = response.full_text_annotation.pages
for page in pages:
    for block in page.blocks:
        print("block confidence:",block.confidence)

        for paragraph in block.paragraphs:
            print("paragraph confidence:",paragraph.confidence)

            for word in paragraph.words:
                word_text = "".join([symbol.text for symbol in word.symbols])
                print("word_text: {0} , confidence: {1}".format(word_text,word.confidence))
