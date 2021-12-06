
import numpy
# x is your dataset
x = numpy.random.rand(100, 5)
indices = numpy.random.permutation(x.shape[0])
training_idx, test_idx = (indices[:50], indices[50:]), (indices[:50], indices[50:])
training, test = x[training_idx,:], x[test_idx,:]