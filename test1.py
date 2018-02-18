from __future__ import print_function
import numpy as np
import tensor
from time import clock
from mnistread import training_images, training_labels, test_images, test_labels, getpim, disp

kernel=np.array([
  [[-3,0,3],[-10,0,10],[-3,0,3]], # Vertical edge detector
  [[-3,-10,-3],[0,0,0],[3,10,3]], # Horizontal edge detector
],dtype=np.float32)/20
kernel=np.moveaxis(kernel,0,2).reshape((3,3,1,2))

def uint8tofloat(a): return a.astype(np.float32)/255-0.5
def floattouint8(a): return np.clip(np.around(((a+0.5)*255)),0,255).astype(np.uint8)

training_images=uint8tofloat(training_images[:,:,:,None])

imagenode=tensor.PlaceHolder("images")
kernelnode=tensor.Constant(kernel)
edgesnode=tensor.Conv2d(imagenode,kernelnode)

t0=clock()
edges=edgesnode.evaluate(results={imagenode:training_images})
print("Time taken to apply convolutions to",len(edges),"images over",kernel.shape[3],"channels =",clock()-t0,"seconds")

disp(floattouint8(edges[2,:,:,0]))
disp(floattouint8(edges[2,:,:,1]))
