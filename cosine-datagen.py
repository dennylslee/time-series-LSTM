# this file generate the rock-paper-sessiors data
# code is done in python 2.7

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import time
import csv

#start = time.time()
player1_list_size = 2000	# the length of the series
norm_mu = 0 				# center the guassian mean at zero
norm_signma = 2				# sigma determine the spread
player1_list = []
player1_list_quant = []

# write the list into a csv

for i in range(player1_list_size):
#	a = random.randint(1,3)
	a = random.gauss(norm_mu,norm_signma)
	if a < -1:				# cutoff point for value below -1 
		a_quant = 1			# value 1 = ROCK
	elif a > 1:				# cutoff point for value above +1
		a_quant = 3			# value 3 = SESSIORS
	else:					# everything else in between
		a_quant = 2			# value 2 =  PAPER
	player1_list.append(a)
	player1_list_quant.append(a_quant)
print (player1_list[:10])  # this print takes a lot of time
print (player1_list_quant[:50])
print (player1_list_quant.count(1))	# count the num of ones in list (rock)
print (player1_list_quant.count(2))	# count the num of twos in list (paper)
print (player1_list_quant.count(3))	# count the num of threes in list (sessiors)

cossample_list = []
coscycle = 20
coscyclelong = player1_list_size/4				# generate low freq cycle to module
for i in range(player1_list_size):				# generate a cosine wave
	cossample = np.cos(np.pi *2 * (i % coscycle) / coscycle)
	cossample += player1_list[i] / 10			# add high freq noise to the cosine, play with denominator here
	cossample += np.square(i/1000)
	cossample += np.sin(np.pi *2 * (i % coscyclelong) / coscyclelong) # add low freq cosine for modulation
	cossample_list.append(cossample)

with open("cosine-timeseries.csv","w") as g :	# in python 2, csv should always open in binary mode to avoid extra line
	f_writer = csv.writer(g)
	for row in cossample_list:					# write as single column
		f_writer.writerow([row])   				# writerows and writerow are different!

x = np.array(player1_list)
nbins = 20
n, bins = np.histogram(x, nbins, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]
#done = time.time()
#print (done-start)
plt.plot(player1_list)
plt.show()
plt.plot(pdfx, pdfy)		# plot the probability distributed function
plt.show()
