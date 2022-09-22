class AugFn:
  
  def __init__(self, name):
    
    self.name = name
    self.params = None

  def train(self, features, beta, **kwargs):
    pass

  def predict(self, features, **kwargs):
    pass

  def save(self, identifier=None):
    pass

  def load(self, identifier=None):
    pass
