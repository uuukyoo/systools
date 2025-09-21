from PIL import Image

img = Image.open("tsubami.jpg")
img.show("original")
# 旋转图片
rotated_img = img.rotate(90)  # 顺时针旋转90度
rotated_img.show("Rotated 90°")

# 翻转图片
flipped_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
flipped_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)    # 垂直翻转

flipped_horizontal.show("Flipped Horizontal")
flipped_vertical.show("Flipped Vertical")
