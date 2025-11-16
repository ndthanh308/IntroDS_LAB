import numpy as np
import cv2
import imageio
orig = imageio.imread('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/000019.jpg')/255.
layer = imageio.imread('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/layer_d.png')/255.
edit =  layer[...,:3] * layer[...,-1:] +(1 - layer[...,-1:]) * orig
imageio.imwrite('/raid/zsz/nips2023/dyedit_sig/contents/occlusion/edit.png', edit)