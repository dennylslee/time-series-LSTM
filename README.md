# Introduction

This project uses the popular Long Short Term Memory (LSTM) Recurrent Neural Network for performing simple time series forcasting (prediction).  Sensitivity to training set size is analyzed.

Two main programs are contained in this project:
1. cosine-datagen.py for generating the time series for analysis
2. cosine-RNN-LSTM.py for constructing the LSTM 

The LSTM is built on Keras framework.

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

The raw waveform (time seres) looks like this:

![image of raw ts](https://github.com/dennylslee/time-series-LSTM/blob/master/cos-rawTS.png)

## LSTM prediction

# Acknowledgment