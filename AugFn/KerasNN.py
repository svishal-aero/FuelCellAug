import numpy as np
import keras as K

from .AugFn import AugFn

class KerasNN(AugFn):

  def __init__(self, name, structure=[], actFn=[], learning_rate=1e-3, batchSize=10, nTrainIters=1000):

    super().__init__(name)
    
    # Check that there is only one output
    assert(structure[-1]==1)

    # Check that there is at least one hidden layer
    assert(len(structure)>2)

    # Create neural network
    self.nn = K.models.Sequential()

    # Add input layer
    self.nn.add(
      K.layers.Dense(
        structure[1],
        input_dim=structure[0],
        activation=actFn[1]
      )
    )

    # Add hidden layers and output layers
    for i in range(2,len(structure)):
      self.nn.add(
        K.layers.Dense(
          structure[i],
          activation=actFn[i]
        )
      )
    
    # Set loss function, optimizer and convergence metrics
    self.nn.compile(
      loss=K.losses.MeanSquaredError(),
      optimizer=K.optimizers.Adam(lr=learning_rate),
      metrics=[]
    )

    self.batchSize = batchSize
    self.nTrainIters = nTrainIters

    self.weightsToParams()

  def weightsToParams(self):

    self.paramStruct = self.nn.get_weights()
    self.params = np.hstack([arr.flatten() for arr in self.paramStruct])

  def paramsToWeights(self):

    # Transfer data from 'params' to 'paramStruct'
    ind=0
    for arr in self.paramStruct:
      arrFlat = arr.ravel()
      indNext = ind + arrFlat.size
      arrFlat[:] = self.params[ind:indNext]
      ind = indNext

    # Set params as NN weights
    self.nn.set_weights(self.paramStruct)

  def train(self, features, beta, **kwargs):

    if 'verbose' in kwargs.keys():
      verbose = kwargs['verbose']
    else:
      verbose = 1
    indices = np.random.permutation(beta.size)
    features = features[indices]
    beta = beta[indices]
    self.nn.fit(
      features,
      beta,
      epochs     = self.nTrainIters,
      batch_size = self.batchSize,
      verbose    = verbose
    )
    self.weightsToParams()
    self.paramsToWeights()

  def predict(self, features, **kwargs):

    return self.nn.predict(features)

  def save(self, identifier=None):
    
    if identifier is None:
      self.nn.save(self.name)
    else:
      self.nn.save(self.name + '_' + str(identifier))

  def load(self, identifier=None):
    
    if identifier is None:
      self.nn = K.models.load_model(self.name)
    else:
      self.nn = K.models.load_model(self.name + '_' + str(identifier))
    self.weightsToParams()
    self.paramsToWeights()
