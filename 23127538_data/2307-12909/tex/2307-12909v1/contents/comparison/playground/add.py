import imageio
import numpy as np
#read image 
orig_img = imageio.imread('000.png')/255.
layer = imageio.imread('layer_d.png')/255.
mask = 1 - layer[:,:,3:]
orig_img = orig_img * mask + (1-mask) * layer[:,:,:3]
imageio.imwrite('orig.png',orig_img)