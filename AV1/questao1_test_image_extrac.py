import pytesseract
import cv2 as cv
import pandas as pd

image = cv.imread("img.png")
image = cv.resize(image,
                         (int(image.shape[1] + (image.shape[1] * .1)),
                          int(image.shape[0] + (image.shape[0] * .25))),
                         interpolation=cv.INTER_AREA)

gray = cv.cvtColor(image,cv.COLOR_RGBA2GRAY)
custom_config = r'--oem 3 --psm 6'  # Configuração personalizada para o pytesseract
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
text = pytesseract.image_to_string(thresh, config=custom_config,lang="eng")
print(text)

lines = text.split('\n')
data = [line.split() for line in lines if line.strip()]

df = pd.DataFrame(data)

print(df)
#result = pytesseract.image_to_s(thresh)

