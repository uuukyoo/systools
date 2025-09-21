import cv2
img = cv2.imread("dive.jpg")
imgCropped = img[50:283,25:190]
shape = imgCropped.shape
print(shape[0])
imgCropped = cv2.resize(imgCropped,(shape[0]*12//10,shape[1]*2))
cv2.imshow("Image cropped",imgCropped)
cv2.imshow("Image",img)
cv2.waitKey(0)
