import numpy as np

def isinteger(x):
  return type(x) in [int, long]# Just [int] in Python 3

class TensorOp(object):
  '''
  Represents a function from zero or more tensors to a tensor.
   - inputs - a list of TensorOps.
  '''
  def __init__(self, inputs, name=None):
    assert type(inputs) is list
    for i in inputs:
      assert isinstance(i, TensorOp)
    self.inputs = inputs
    self.name = name

  def raw_evaluate(self, tensors):
    '''
    Evaluates this function, given the values of its inputs.
    Subclasses must override this method to define the function.
     - tensors - a list of np.ndarrays.
    '''
    raise NotImplementedError('Abstract method')

  def evaluate(self, results={}):
    '''
    Ensures that all inputs have already been evaluated, then calls `raw_evaluate()`.
     - results - a dict from TensorOp to tensor representing results already computed.
    Sets `results[self]` to the result.
    '''
    if self not in results:
      results[self] = self.raw_evaluate([i.evaluate(results=results) for i in self.inputs])
    return results[self]

  def raw_gradient_op(self, g_self):
    '''
    Returns an iterable yielding pairs `(i, G(i))`, where:
     - `i` must be one of `self.inputs`. The inputs can be returned in any order, zero or more times each.
     - `G(i)` is the product of the matrix `d(self)/d(i)` with the vector `g_self`.
    '''
    raise NotImplementedError('Abstract method')
  
  def gradient_op(self, g_self):
    '''
    Construct a TensorOp G(t) for each TensorOp `t` that `self` depends on, representing the
    derivative of an objective function with respect to `t`.
     - g_self - G(self), i.e. the derivative of the objective function with respect to `self`.
    Returns a dict mapping each TensorOp `t` to `G(t)`.
    '''
    # Compute reverse post-order of the graph.
    seen = set()
    order = []
    def walk(t):
      if t in seen: return
      seen.add(t)
      for i in t.inputs:
        walk(i)
      order.append(t)
    walk(self)
    order.reverse()
    # Iterate over the dependencies.
    g = {self: g_self}
    for t in order:
      for (i, g_i) in t.raw_gradient_op(g[t]):
        if i not in g:
          g[i] = g_i
        else:
          g[i] = Add(g[i], g_i)
    return g
  
  def __str__(self):
    return "%s(%s)" % (self.__class__.__name__, self.name)
               
class PlaceHolder(TensorOp):
  def __init__(self, name):
    super(PlaceHolder, self).__init__([], name=name)
    self.name = name

  def raw_evaluate(self, tensors):
    raise ValueError("Please specify a value for %s" % self)

  def raw_gradient_op(self, g_self):
    return []
  
class Constant(TensorOp):
  def __init__(self, tensor, name=None):
    '''
     - tensor - an np.ndarray.
    '''
    super(Constant, self).__init__([], name=name)
    self.tensor = tensor

  def raw_evaluate(self, tensors):
    assert len(tensors) == 0
    return self.tensor

  def raw_gradient_op(self, g_self):
    return []

ZERO=Constant(0.0)

class IfGreater(TensorOp):
  def __init__(self, greater, lesser, if_true, if_false, name=None):
    super(IfGreater, self).__init__([greater, lesser, if_true, if_false], name=name)

  def raw_evaluate(self, tensors):
    (greater, lesser, if_true, if_false) = tensors
    return np.where(greater > lesser, if_true, if_false)

  def raw_gradient_op(self, g_self):
    (greater, lesser, if_true, if_false) = self.inputs
    return [
      (if_true, IfGreater(greater, lesser, g_self, ZERO)),
      (if_false, IfGreater(greater, lesser, ZERO, g_self)),
    ]

class Maximum(IfGreater):
  def __init__(self, input1, input2, name=None):
    super(Maximum, self).__init__(input1, input2, input1, input2, name=name)

class Minimum(IfGreater):
  def __init__(self, input1, input2, name=None):
    super(Minimum, self).__init__(input1, input2, input2, input1, name=name)

class Relu(Maximum):
  def __init__(self, input, name=None):
    super(Relu, self).__init__(input, ZERO, name=name)

class Add(TensorOp):
  def __init__(self, input1, input2, name=None):
    super(Add, self).__init__([input1, input2], name=name)

  def raw_evaluate(self, tensors):
    (t1, t2) = tensors
    return np.add(t1, t2)

  def raw_gradient_op(self, g_self):
    (input1, input2) = self.inputs
    return [
      (input1, g_self),
      (input2, g_self),
    ]

class Mul(TensorOp):
  def __init__(self, input1, input2, name=None):
    super(Mul, self).__init__([input1, input2], name=name)

  def raw_evaluate(self, tensors):
    (t1, t2) = tensors
    return np.multiply(t1, t2)

  def raw_gradient_op(self, g_self):
    (input1, input2) = self.inputs
    return [
      (input1, Mul(g_self, input2)),
      (input2, Mul(input1, g_self)),
    ]

class Matmul(TensorOp):
  '''
  Matrix multiplication. Use numpy's matmul broadcast rules.
  This will work as expected for matrix-matrix multiplication if one of the arguments has rank 2.
  It will work as expected for matrix-vector multiplication if the second argument has rank 1,
  so any batch dimension has to be in the first argument in this case.
  Examples of sane cases:
   - (X, Y, Z, A, B) @ (B, C) -> (X, Y, Z, A, C)
   - (A, B) @ (X, Y, Z, B, C) -> (X, Y, Z, A, C)
   - (X, Y, Z, A, B) @ (B,) -> (X, Y, Z, A,)
  '''
  def __init__(self, input1, input2, name=None):
    super(Matmul, self).__init__([input1, input2], name=name)

  def raw_evaluate(self, tensors):
    (t1, t2) = tensors
    return np.matmul(t1, t2)

  # input1 = (X, Y, Z, A, B) 
  # input2 = (B, C) 
  # g_self = (X, Y, Z, A, C)  (same shape as output)
  # d/dinput1 = (X, Y, Z, A, C) @ (B, C) (contracted over C)
  # d/dinput2 = (X, Y, Z, A, B) @ (X, Y, Z, A, C)  (contracted over X, Y, Z, A)
  #
  # Only works in some cases
  def raw_gradient_op(self, g_self):
    (input1, input2) = self.inputs
    return [
      (input1, Matmul(g_self, input2.T)),
      (input2, Matmul(input1.T, g_self)),
    ]

class MoveAxis(TensorOp):
  def __init__(self, input, sources, destinations, name=None):
    super(MoveAxis, self).__init__([input], name=name)
    self.sources = sources
    self.destinations = destinations

  def raw_evaluate(self, tensors):
    (t,) = tensors
    return np.moveaxis(t, self.sources, self.destinations)

  def raw_gradient_op(self, g_self):
    (input,) = self.inputs
    return [
      (input, MoveAxis(g_self, destinations, sources)),
    ]
  
class TensorDot(TensorOp):
  def __init__(self, input1, input2, rank1=2, rank2=2, axes=2, name=None):
    super(TensorDot, self).__init__([input1, input2], name=name)
    self.rank1=rank1
    self.rank2=rank2
    self.axes=axes

  def raw_evaluate(self, tensors):
    (t1, t2) = tensors
    assert len(t1.shape)==self.rank1
    assert len(t2.shape)==self.rank2
    return np.tensordot(t1, t2, self.axes)
  
  def raw_gradient_op(self, g_self):
    (input1, input2) = self.inputs
    if isinteger(self.axes):
      # As per documentation of `np.tensordot()`.
      (ax1, ax2) = (range(self.rank1-self.axes, self.axes), range(self.axes))
    else:
      (ax1, ax2) = self.axes
    nc = len(ax1); assert nc==len(ax2)
    # Complements of `ax1` and `ax2`.
    cax1 = [a for a in range(self.rank1) if a not in ax1]
    cax2 = [a for a in range(self.rank2) if a not in ax2]
    n_self = len(cax1) + len(cax2)
    # Which axes of `self` come from which input.
    from1 = range(len(cax1)); from2 = range(len(cax1), n_self)
    return [
      (input1, MoveAxis(
        TensorDot(g_self, input2, n_self, self.rank2, axes=(from2, cax2)),
        range(self.rank1-nc, self.rank1),
        [ax1[i] for i in np.argsort(ax2)]
      )),
      (input2, MoveAxis(
        TensorDot(input1, g_self, self.rank1, n_self, axes=(cax1, from1)),
        range(nc),
        [ax2[i] for i in np.argsort(ax1)]
      )),
    ]

def Contract(TensorOp):
  '''
  Represents a general contraction combined with a permutation of the axes.
   - s1     - a string with one element for each axis of `input1`. "F" means the axis is contracted with `input2` and "B" means it appears in `self`.
   - s2     - a string with one element for each axis of `input2`. "F" means the axis appears in `self` and "B" means it is contracted with `input1`.
   - s_self - a string with one element for each axis of `self`. "F" means the axis comes from `input1` "B" means it comes from `input2`.
  '''
  def __init__(self, input1, input2, s1, s2, s_self, name=None):
    super(Contract, self).__init__([input1, input2], name=name)
    for (x, y) in [(s1, s2), (s2, s_self), (s_self, s1)]:
      assert sum(c=='F' for c in x) == sum(c=='B' for c in y)
    self.s1 = s1; self.s2 = s2; self.s_self = s_self

def raw_evaluate(self, tensors):
  (t1, t2) = tensors
  assert len(t1.shape)==len(self.s1); assert len(t2.shape)==len(self.s2)
  product = np.tensordot(t1, t2, axes=(
    [i for (i, c) in enumerate(self.s1) if c=='F'],
    [i for (i, c) in enumerate(self.s2) if c=='B'],
  ))
  from1 = [i for (i, c) in enumerate(self.s_self) if c=='F']
  return np.moveaxis(product, range(len(from1)), from1)

def raw_gradient_op(self, tensors):
  (input1, input2) = self.inputs
  return [
    (input1, Contract(input2, g_self, self.s2, self.s_self, self.s1)),
    (input2, Contract(g_self, input1, self.s_self, self.s1, self.s2)),
  ]

class Conv2d(TensorOp):
  '''
  - Input shape (n,h,w,c)
  - Kernel shape (kh,kw,c,c')
  - Output shape (n,h-kh+1,w-kw+1,c')
  '''
  def __init__(self, input, kernel, name=None):
    super(Conv2d, self).__init__([input, kernel], name=name)

  def raw_evaluate(self, tensors):
    (input, kernel) = tensors
    (n,h,w,c)=input.shape
    (kh,kw,kcin,kcout)=kernel.shape
    assert kcin==c
    (ns,hs,ws,cs)=input.strides
    newinput=np.ndarray(buffer=input.data, dtype=input.dtype, shape=(n,h-kh+1,w-kw+1,kh,kw,c),
                        strides=(ns,hs,ws,hs,ws,cs))
    return np.tensordot(newinput,kernel,axes=3)
