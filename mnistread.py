import numpy as np
from PIL import Image, ImageOps

# b=str of 4 bytes
def bigend4(b):
  t=0
  for x in b: t=(t<<8)+ord(x)
  return t

def readmnistimages(fn):
  with open(fn,'r') as fp:
    magic=bigend4(fp.read(4));assert magic==2051
    n=bigend4(fp.read(4))
    rows=bigend4(fp.read(4))
    cols=bigend4(fp.read(4))
    return np.reshape(np.fromstring(fp.read(), dtype=np.uint8), (n, rows, cols))

def readmnistlabels(fn):
  with open(fn,'r') as fp:
    magic=bigend4(fp.read(4));assert magic==2049
    n=bigend4(fp.read(4))
    return np.fromstring(fp.read(), dtype=np.uint8)

def getpim(im):
  assert im.dtype==np.uint8
  assert len(im.shape)==2
  pim=Image.fromarray(im,mode='L')
  return pim

def disp(im):
  ImageOps.fit(getpim(im),(120,120)).show()

training_images=readmnistimages("train-images-idx3-ubyte")
training_labels=readmnistlabels("train-labels-idx1-ubyte")
test_images=readmnistimages("t10k-images-idx3-ubyte")
test_labels=readmnistlabels("t10k-labels-idx1-ubyte")
