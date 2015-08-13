from scipy import misc, ndimage
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from pylab import *
from numpy import *
import PIL



def pil2array(im,alpha=0):
    if im.mode=="L":
        a = fromstring(im.tostring(),'B')
        a.shape = im.size[1],im.size[0]
        return a
    if im.mode=="RGB":
        a = fromstring(im.tostring(),'B')
        a.shape = im.size[1],im.size[0],3
        return a
    if im.mode=="RGBA":
        a = fromstring(im.tostring(),'B')
        a.shape = im.size[1],im.size[0],4
        if not alpha: a = a[:,:,:3]
        return a
    return pil2array(im.convert("L"))
    
im = PIL.Image.open('bq01_006-1.png')
im = pil2array(im)

im = im / 255.0

print(im)


h = im.shape[0]
w = im.shape[1]

smooth = filters.gaussian_filter(im, (h * 0.5, h * 1.0), mode='constant')

smooth += 0.001*filters.uniform_filter(smooth, (h*0.5, w), mode='constant')

print(smooth.shape)

a = argmax(smooth, axis=0)
a = filters.gaussian_filter(a, h * 0.3)

center = array(a,'i')
# print(center)
deltas = abs(arange(h)[:, newaxis] - center[newaxis, :])
mad = mean(deltas[im != 0])
r = int(1 + 4 * mad)

plt.imshow(smooth, cmap=cm.gray)
plot(center)
plt.show()