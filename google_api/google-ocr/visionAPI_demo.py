from http import client
import os, io
from google.cloud import vision
# from google.cloud.vision import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'serviceAcoount.json'
print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
client = vision.ImageAnnotatorClient()

# def crophint(file_path, asepct_ratio):
#     with io.open(file_path, 'rb') as image_file:
#         content = image_file.read()
#     image = vision.types.Image(content = content)
#     crop_hint_params = vision.types.CropHintsparams(asepct_ratio=asepct_ratio)
#     image_context = vision.types.ImageContent(
#         crop_hint_params=crop_hint_params
#     )
#     response = client.crop_hints(
#         image = image,
#         image_context = image_context
#     )

FOLDER_PATH = r'/home/gauravirajput/TPN/train-ocr/google-ocr/demo/'

IMAGE_FILE = 'presc2.jpg'
FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)

with io.open(FILE_PATH, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content = content)
response = client.document_text_detection(image=image)
docText = response.full_text_annotation.text
print(docText)

