import pandas
import matplotlib.pyplot as plt
#dataset = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataset = pandas.read_csv('sample-timeseries.csv', usecols=[0], engine='python', skipfooter=0)
dataset = pandas.read_csv('cosine-timeseries.csv', usecols=[0], engine='python', skipfooter=0)
plt.plot(dataset)
plt.show()