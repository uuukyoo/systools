from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

img = Image.open("dive.jpg").convert("L") 

img_array = np.array(img)

result = gaussian_filter(img_array, sigma=5)

fig = plt.figure()
plt.gray()  

ax1 = fig.add_subplot(121)
ax1.set_title("Original")
ax1.imshow(img_array)

ax2 = fig.add_subplot(122)
ax2.set_title("Gaussian Blur")
ax2.imshow(result)

plt.show()

