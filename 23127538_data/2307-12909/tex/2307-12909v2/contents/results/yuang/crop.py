import imageio
import numpy as np
# read image 
import glob 
import matplotlib.pyplot as plt
img = imageio.imread('000019.jpg')
img = img[200:200+512, 650:650+512]
imageio.imwrite('000019.jpg', img)
# plt.imshow(img)
# plt.show()
# imgs_path = glob.glob('rgb_map_*.jpg')
# for img_path in imgs_path:
#     img = imageio.imread(img_path)
#     img = img[200:200+512, 550:550+512]
#     imageio.imwrite(img_path, img)
    # plt.imshow(img)
    # plt.show()
    # exit()