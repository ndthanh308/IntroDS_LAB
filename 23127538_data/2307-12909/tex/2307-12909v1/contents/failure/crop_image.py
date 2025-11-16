import numpy as np
import cv2
import imageio
orig = imageio.imread('/raid/zsz/nips2023/dyedit_sig/contents/failure/000.png')/255.
layer = imageio.imread('/raid/zsz/nips2023/dyedit_sig/contents/failure/005.png')/255.
h = 0
w = 100
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/failure/edit_crop.png',layer[h:h+200,w:w+355])
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/failure/orig_crop.png',orig[h:h+200,w:w+355])

