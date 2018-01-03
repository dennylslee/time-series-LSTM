# this file generates the cosine-based time series data for LSTM testin

import random
import numpy as np
import pandas

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import time
import csv

#start = time.time()
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

with open("cosine-timeseries.csv","w") as g :
	f_writer = csv.writer(g)
	for row in cossample_list:					# write as single column
		f_writer.writerow([row])   				

# plot the probability distributed function
x = np.array(TS_noise_list)
nbins = 20
n, bins = np.histogram(x, nbins, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]
plt.plot(pdfx, pdfy)		
# plt.show()  									# optional plot to visualize the shape of the pdf distribution

# plot the final time series
dataset = pandas.read_csv('cosine-timeseries.csv', usecols=[0], engine='python', skipfooter=0)
plt.plot(dataset)
plt.show()