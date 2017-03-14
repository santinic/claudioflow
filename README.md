## ClaudioFlow
ClaudioFlow is a didactic full-featured Deep Learning framework implemented only using NumPy,
with a focus on simplicity
and code readability (rather than performance).

Most modules in the code link to a web resource or to pages of the
Deep Learning Book by Goodfellow et al. that explains the theory behind them.

ClaudioFlow architecture draws a lot of inspiration from Torch 7's clean approach, as well
as Keras interfaces.

## Example usage
```python
model = SequentialModel([
    LinearLayer(2, 5),
    TanhLayer(),
    LinearLayer(5, 1),
    TanhLayer(),
])
```


## Defining new layers
Similarly to Torch, you just need to implement a forward() and a backward() function, and you get a functioning layer.
```python

class MyNewLayer:
    def forward(self, x):
        self.x = x
        return x**2
    def backward(self, delta):
        return delta * 2 * self.x
        
model = SequentialModel([
   LinearLayer(2, 5),
   MyNewLayer(),
])
 
model.learn(...)
```

## Why
The only good way to understand modern neural network is to try to implement one from scratch.
After that you can use TensorFlow.

## Implementend features
- Minibatch learning with train/validation/test sets and "patience"
- Tests with numerical gradient checks
- Softmax Negative Log Likelihood (Cross Entropy)
- Working examples like Multi-layer Perceptron for Mnist classification
- SGD Momentum / AdaGrad / RMSProp
- Sigmoid / Relu / Tanh and more common activation functions

## Work in progress
- Recurrent Networks
- LSTM
- Convolutions
- Reinforcement learning

## Will never have
- Automatic differentiation: Theano is perfect if you need that in python.

