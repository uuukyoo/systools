from PIL import Image,ImageEnhance
img_original = Image.open("dive.jpg")
img_original.show("Original Image")
img = img_original.resize((50, int(img_original.size[1] * 50 / img_original.size[0])))
img.show("Image changed")

