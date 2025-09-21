from PIL import Image,ImageEnhance
img_original = Image.open("dark.png")
img_original.show("Original Image")
img = ImageEnhance.Contrast(img_original)
img.enhance(10).show("Image With More Contrast")
