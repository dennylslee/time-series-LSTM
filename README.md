# Introduction

This project uses the popular Long Short Term Memory (LSTM) Recurrent Neural Network for performing simple time series forcasting (prediction).  Sensitivity to training set size is analyzed.

Two main programs are contained in this project:
1. cosine-datagen.py for generating the time series for analysis
2. cosine-RNN-LSTM.py for constructing the LSTM 

The LSTM is built on Keras framework. For some of best LSTM tutorial I have encontered, goes to [Colah's Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [Shi Yan blog on understanding LSTM](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)

## Time series generation

The raw time series is created 4 raw components (effectively modulation):
1. a so-called carrier cosine wave 
2. a low frequncy sine wave modulation
3. a slow rising parabola curve 
4. guassian noise added on top at each timestep

The main code construct:

```python
TS_list_size = 2000			# the length of the time series
norm_mu = 0 				# center the guassian mean at zero
norm_signma = 2				# sigma determine the spread
noise_amp_ctl = 10			# a denominator value for amplitude control	
low_freq_cycle = 4			# number of low frequncy cycle in the time series
TS_noise_list = []
TS_list_quant = []

# write the list into a csv
for i in range(TS_list_size):
	a = random.gauss(norm_mu,norm_signma)
	TS_noise_list.append(a)

# put the time series waveform together
cossample_list = []
coscycle = 20
coscyclelong = TS_list_size/low_freq_cycle						# generate low freq cycle to module
for i in range(TS_list_size):									# generate a cosine wave
	cossample = np.cos(np.pi *2 * (i % coscycle) / coscycle) 	# main carrier frequency
	cossample += TS_noise_list[i] / noise_amp_ctl		     	# add high freq noise to the cosine
	cossample += np.square(i/1000)								# put the time series on a square function
	cossample += np.sin(np.pi *2 * (i % coscyclelong) / coscyclelong) # add low freq cosine for modulation
	cossample_list.append(cossample)
```

The raw waveform (time series) looks like this:

![image of raw ts](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-rawTS.png)

## LSTM prediction

A single layer LSTM is used for performing the prediction.  The internal state (vector) size of the cell and hidden state is set as 10. The look_back variable controls the size of the input vector into the RNN(LSTM).  

Sensitivity analysis options:
1. The training size proportion
2. Look back (i.e. the timestep of the input vector) 
3. LSTM unit which is the internal vector size of the cell (memory) and hidden state 

```python
# split into train and test sets
# control the proportion of training set here
train_size = int(len(dataset) * 0.02)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
# look_back dictates the time steps and the hidden layer; can cause overfitting error when it's too large
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
 
# reshape input to be [samples, time steps, features]
# NOTE: time steps and features are reversed from example given
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1],1))

# create and fit the LSTM network
# single hidden layer (timestep is one) and cell states
# The "unit" value in this case is the size of the cell state and the size of the hidden state
model = Sequential()
model.add(LSTM(10, input_shape=(look_back,1))) 	# NOTE: time steps and features are reversed from example given
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
```

## Results from varying training size

The predition results are overlayed on top of the raw time series.  The orange colored line is the training set and the green colored line is the testing set. The presence of a larger amount of training data minimizes the error (root mean square error) as the LSTM can fit the model from a more representative sequence. 

20 percent for training:

![image of 20pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-20pct-training.png)

10 precent for training:

![image of 10pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-10pct-training.png)

2 percent for training:

![image of 2pct training](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-2pct-training.png)

## Result from adding an extra stacked layer of LSTM blocks

Keras provided a simple way to stack LSTM layer by simply evoking the model.add method.  The main change is to ensure the attribute of return_sequences is set to True in the previous layer such that all the LSTM blocks are from the lower layer to the upper layer. 

```python
# create and fit the LSTM network
# single hidden layer (timestep is one) and cell states
# The "unit" value in this case is the size of the cell state and the size of the hidden state
model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(look_back,1))) 	# NOTE: time steps and features are reversed from example given
model.add(LSTM(10))														# optional stack layer of LSTM blocks
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
```

Result shown for the 2 percentage training set case below.  The RMSE seems to have worsen and the predict does not follow the slow rise trend (i.e. created by the parabolic component in the original time series).  

The lack of training is posting a greater challenage to the LSTM and possibly the extra layer with no dropout  inserted causes an overfitting situation leading to the worsen result. 

![image stack layer](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-testresult-2pct-training-w-stack.png)

# Acknowledgment

Thanks to Dr. Jason Brownlee from [machinelearningmastery.com](https://machinelearningmastery.com/) for providing the base code. 