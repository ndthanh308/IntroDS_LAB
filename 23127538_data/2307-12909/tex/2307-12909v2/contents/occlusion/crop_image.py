import numpy as np
import cv2
import imageio
orig = imageio.imread('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/or.jpg')/255.
layer = imageio.imread('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/edit.png')/255.
h = 300
w = 650
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/edit_crop.png',layer[h:h+300,w:w+512])
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/orig_crop.png',orig[h:h+300,w:w+512])

oc = imageio.imread('oc.jpg')/255.
wooc = imageio.imread('wooc.jpg')/255.
h = 200
w = 800
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/oc_crop.png',oc[h:h+300,w:w+512])
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/wooc_crop.png',wooc[h:h+300,w:w+512])