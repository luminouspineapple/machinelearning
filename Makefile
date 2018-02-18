# Download MNIST dataset

all:: t10k-labels-idx1-ubyte t10k-images-idx3-ubyte train-labels-idx1-ubyte train-images-idx3-ubyte

MNISTADDR:=http://yann.lecun.com/exdb/mnist/

t10k-labels-idx1-ubyte:
	wget $(MNISTADDR)/t10k-labels-idx1-ubyte.gz -O - |gunzip -c > t10k-labels-idx1-ubyte

t10k-images-idx3-ubyte:
	wget $(MNISTADDR)/t10k-images-idx3-ubyte.gz -O - |gunzip -c > t10k-images-idx3-ubyte

train-labels-idx1-ubyte:
	wget $(MNISTADDR)/train-labels-idx1-ubyte.gz -O - |gunzip -c > train-labels-idx1-ubyte

train-images-idx3-ubyte:
	wget $(MNISTADDR)/train-images-idx3-ubyte.gz -O - |gunzip -c > train-images-idx3-ubyte
