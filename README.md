# cond_rnn_optics
An implementation of [cond_rnn](https://github.com/philipperemy/cond_rnn) for the prediction of pulse propagation in nonlinear waveguides.

This project is a companion to an [Optics Express publication](http://dx.doi.org/10.1364/OE.506519), and more in general to my PhD work in pulse propagation modelling with deep learning techniques. 

## Background
Pulse propagation modelling in nonlinear optical materials is traditionally performed via numerical solutions of equations (NLSE, NEE, UPPE). Recently however, data-driven recurrent neural networks have shown potential in reducing the long running times required by legacy methods. This project builds on the state of the art to provide a more generalised ML-bsed approach. Combining existing RNNs for pulse propagation with the `ConditionalRecurrent()` wrapper for Keras, the prediction is informed by auxiliary data, which allows for a single neural network capale of predicting the evolution of optical pulses through different structures. 

For more background on the research and the theory, please refer to the [paper](http://dx.doi.org/10.1364/OE.506519). 

## Usage
Firstly, you will need to install both Tensorflow and [Conditional Recurrent](https://github.com/philipperemy/cond_rnn).
The model can then be built by using the Keras Functional API, as shown in the Jupyter Notebooks in the `CODE` folder.
Two examples for creating models are provided here, for uses with one or two conditions.
In short, the idea is to define the inputs (sequential and conditional) as nodes, and pass them as inputs to the `LSTM` layers, that are wrapped with the `ConditionalRecurrent`. 

```python
# Sequence
i = Input(shape=[sequence_length, n_features])

# Conditional parameter
c = Input(shape=[cond_features])

# Combine into the first conditional LSTM layer
x = ConditionalRecurrent(LSTM(n_features, return_sequences=True))([i, c])

# Combine again into the second conditional LSTM layer
x = ConditionalRecurrent(LSTM(n_features, return_sequences=False))([x, c])

# Finally, the output Dense layer
x = Dense(units=n_features, activation=a_func)(x)

# Build the model from the tensors
model = Model(inputs=[i, c], outputs=[x])

# Compile the model
model.compile()
```
The model takes as inputs a sequence, like any LSTM model, and a condition. This means that for each individual prediction, the sequence is fed together with a conditon, which allows for different values of the condition to be used within a single structure. For instance, this is how non-uniformly poled structures can be modelled with this architecture (see [paper](http://dx.doi.org/10.1364/OE.506519)).

### Training and testing
Similarly to any non-sequential Keras model, the training and testing dataset includes multiple inputs, as shown here:
```python
history = model.fit(
        verbose=1,
        x=[train_x, train_c], y=train_y,
    batch_size=140,
        validation_data=([test_x, test_c], test_y),
        epochs= 200
    )
```
Since each individual sequence is fed with a condition, the `train_` and `test_` arrays have the follwing shapes:
```python
x = [None, sequence_length, n_features]
c = [None, cond_features]
y = [None, n_features]
```
See `CODE/TrainModel.ipynb` for a more detailed guide on how to train the model.




### Tips
Although each use case is different, and the hyperparameters should be determined heuristically, there are a few things I have found that could be useful.
#### Normalization
As suggested by @philipperemy [here](https://github.com/philipperemy/cond_rnn/issues/41), complex conditions should be normalised to a range centered at 0 and with a varinace of 1.
#### Initialization of hidden states
When modelling complex dynamics like the evolution of optical pulses, I have found that feeding each condition twice works best, as you can see in the example notebooks. 


## Examples and performance
Here is a couple of examples of what the trained networks can do. In these examples, the networks have been trained with the simulations computed via a numerical solution of the [UPPE](http://dx.doi.org/10.1103/PhysRevA.105.043511). 
### Single conditional network
This network was built with a sequence lenght of 10, for the prediction of a spectrum made of 500 points along the freuency axis (`n_features=500`). The condition fed with each sequence is the poling period of the second-order nonlinear coefficient at that longitudinal step $\Lambda(z)$. The initial input is therefore 10 sequential spectra at propagation distances $z=z-10 \Delta z$, $z=z-9\Delta z$, ..., $z=z-\Delta z$, and the condition $\Lambda(z)$, and the output is the spectrum at $z=z$. In this example, $\Delta z$ is set at 30 $\mu$ m.
<strong>N.B.</strong>: The condition is normalized as discussed above.

This is a visual comparison of the RNN compared to the numerical solution.
![githubExample1](https://github.com/Simo-JDev/cond_rnn_optics/assets/44927443/71a92f2d-c629-49c3-8bc9-625378da2ddd)

This model produced the above result in 5 seconds.

<strong>For more results and a detailed discussion on the performance, pelase see the [paper](http://dx.doi.org/10.1364/OE.506519)</strong>

## Possible future improvements
Since the `ConditionalRecurrent` is compatible with all recurrent layers in Keras, it is possible to use `SimpleRNN` and `GRU` layers instead of the `LSTM`. Specifically, `GRU` layers could be used for achieving a more lean and quick network. 

It should also be possible to extend this work, to include three or more conditional parameters.

I am also working on a more generalized version of this repo, since this method can ideally be applied to other nonlinear systems. 


## References
- [Conditional Recurrent for Tensorflow/Keras](https://github.com/philipperemy/cond_rnn)
- Lauria and Saleh, [Conditional recurrent neural networks for broad applications in nonlinear optics](http://dx.doi.org/10.1364/OE.506519)
- Lauria and Saleh, [Mixing second- and third-order nonlinear interactions in nanophotonic lithium-niobate waveguides](http://dx.doi.org/10.1103/PhysRevA.105.043511)
