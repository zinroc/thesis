from matplotlib import pyplot as plt
import csv
import math
import numpy as np
import random
import sys

"""
These are the methods specific to the function being optimized
"""

################################################################################################
	
def get_saturation_gradient (t, parameters):
	"""Saturation gradient is a vector in the following order:
		* df_dK0
		* df_dK1
		* df_dN0
		* df_dN1
		* df_dN2
		
	"""

	K0 = parameters[0]
	K1 = parameters[1]
	N0 = parameters[2]
	N1 = parameters[3]
	N2 = parameters[4]
	T  = N0*(K0+K1)/(K0*K1)
	dt = parameters[5]
	C3 = parameters[6]
	Cvert = parameters[7]
	r = parameters[8]

	#old: df_dt = ((1 / (N0 * (K0 + K1) **2 )) * (K0 ** 2) * np.exp(K0 * K1 * (dt - t) / (N0 * (K0+K1)))+ 1 / N1) * (-1 / (C3*np.exp(r * (dt - t)) + 1)) + (C3 * r * (np.exp(r*(dt - t)))*((1/K1)*((K0/(K0+K1))*np.exp(K0 * K1 * (dt - t)/(N0 * (K0 + K1)))-1))+(dt-t)/N1) / ((C3*np.exp(r*(dr-t))+1)**2) + 1/N2
	df_dt = ((1 / (N0 * (K0 + K1) **2 )) * (K0 ** 2) * np.exp(K0 * K1 * (dt - t) / (N0 * (K0+K1)))+ 1 / N1) * (-1 / (C3*np.exp(r * (dt - t)) + 1)) + (C3 * r * (np.exp(r*(dt - t)))*((1/K1)*((K0/(K0+K1))*np.exp(K0 * K1 * (dt - t)/(N0 * (K0 + K1)))-1))+(dt-t)/N1) / ((C3*np.exp(r*(dt-t))+1)**2) + 1/N2

	df_dk0 = np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))*(dt*K0*K1+K0*(N0-K1*t)+K1*N0)/(((K0+K1)**3)*(C3*N0*np.exp(r*(dt-t))+N0))
	df_dk1 = (-1/((K1**2)*(C3*np.exp(r*(dt-t))+1)))*(K0*K1*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))*(-dt*(K0**2)+(K0**2)*t+K0*N0+K1*N0)/(N0*(K0+K1)**3)+((K0/(K0+K1))*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))-1))
	df_dN0 = -(K0**2)*(dt-t)*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))/((N0**2)*((K0+K1)**2)*(C3*np.exp(r*(dt-t))+1))
	df_dN1 = (t-dt)/((N1**2)*(C3*np.exp(r*(dt-t))+1))
	df_dN2 = (dt-t)/(N2**2)
	#OPTIONAL - finding dt
	#df_ddt = ((1/(N0*(K0+K1)^2))*(K0^2)*np.exp((K0*K1*(dt-t))/(N0(K0+K1)))+1/N1)*(-1/(C3*np.exp(r*(dt-t))+1))+(C3*r*(np.exp(r*(dt-t)))*((1/K1)*((K0/(K0+K1))*np.exp((K0*K1*(dt-t)/(N0*(K0+K1))))-1))+(dt-t)/N1)/((C3*np.exp(r*(dr-t))+1)^2)-1/N2
	
	return np.array([df_dk0, df_dk1, df_dN0, df_dN1, df_dN2])
	
def mod_maxwell(time, parameters):

	K0 = parameters[0]
	K1 = parameters[1]
	N0 = parameters[2]
	N1 = parameters[3]
	N2 = parameters[4]
	T  = N0*(K0+K1)/(K0*K1)
	dt = parameters[5]
	C3 = parameters[6]
	Cvert = parameters[7]
	r = parameters[8]
	return -1*((1/K1)*(1-(K0/(K0+K1))*np.exp(-1*(time-dt)/(T)))+(time-dt)/N1)
	
def s_curve (time, parameters):
	K0 = parameters[0]
	K1 = parameters[1]
	N0 = parameters[2]
	N1 = parameters[3]
	N2 = parameters[4]
	T  = N0*(K0+K1)/(K0*K1)
	dt = parameters[5]
	C3 = parameters[6]
	Cvert = parameters[7]
	r = parameters[8]
	return 1/(1+C3*np.exp((-time + dt)*r))
	
def steady(time, parameters):
	K0 = parameters[0]
	K1 = parameters[1]
	N0 = parameters[2]
	N1 = parameters[3]
	N2 = parameters[4]
	T  = N0*(K0+K1)/(K0*K1)
	dt = parameters[5]
	C3 = parameters[6]
	Cvert = parameters[7]
	r = parameters[8]
	
	# note that in Sergei's initial calculation, this is just time / N2 + Cvert
	return (time-dt)/N2 + Cvert
	
def saturation_function(time, parameters):
	"""Parameters is a vector in the following order:
		* K0
		* K1
		* N0
		* N1
		* N2
		* dt
		* C3 
		* Cvert
		* r
		
	"""
	
	return mod_maxwell(time, parameters) * s_curve(time, parameters) + steady(time, parameters)

##################################################################

"""
Function-agnostic methods
"""

def get_gradient(X, Y, parameters):
	"""Return gradient over all parameters.
	Will return 0 for all parameters other than first 5, because not optimizing those."""

	obj_grad = np.zeros(len(parameters))

	# compute gradients for all data points
	grad = get_saturation_gradient(X, parameters)

	for i in range(5):
		obj_grad[i] = np.sum(-2 * Y * grad[i] + 2 * saturation_function(X, parameters) * grad[i])

	return obj_grad

def get_loss(X, Y, parameters, left_cutoff=None, right_cutoff=None):
	"""Return float that gives loss of function being optimized compared to ideal scenario.
	"""

	# used to zero out the guys that are outside our range
	factor = np.ones(len(X))
	if left_cutoff is not None:
		low_value_indeces = X < left_cutoff
		factor[low_value_indeces] = 0

	if right_cutoff is not None:
		high_value_indeces = X > right_cutoff
		factor[high_value_indeces] = 0

	return np.sum(factor * (Y - saturation_function(X, parameters)) ** 2)
	
def optimize(X, Y, left_cutoff=None, right_cutoff=None):
	"""Tune parameters for the given function."""

	MAX_ITERATIONS = 10000
	LAMBDA = 0.2
	EPSILON = 0.05

	# placeholders
	parameters = np.zeros(9) * 1.0
	# fill in the last several, as they are user-generated
	parameters[5] = 5000.0
	parameters[6] = 1000.0
	parameters[7] = -11.1
	parameters[8] = 0.1

	# come up with random values for the rest
	#parameters[0] = math.exp(random.randint(1, 15)) * 1.0
	parameters[0] = math.exp(10) * 1.0
	parameters[1] = math.exp(random.randint(1, 15)) * 1.0
	parameters[2] = math.exp(random.randint(1, 15)) * 1.0
	parameters[3] = -1.0 * math.exp(random.randint(1, 15))
	parameters[4] = -1.0 * math.exp(random.randint(1, 15))

	for it in range(MAX_ITERATIONS):
		loss = get_loss(X, Y, parameters, left_cutoff, right_cutoff)
		if loss < EPSILON:
			print "*** close enough"
			break
		elif loss == float("+inf") or math.isnan(loss):
			print "*** [ERROR] OVERFLOW!!!"
			print "*** Consider setting learning rate to smaller value"
			return None

		#sys.stdout.write("\rIteration %d\n" % (it))
		#print "\rIteration%d" % it
		if it % 500 == 0:
			print "[%d] Loss is %.2f" % (it, loss)

		#TODO calculate gradients
		#set new parameters
		parameters -= LAMBDA * get_gradient(X, Y, parameters)

	return parameters
	
def load_data():
	"""Return X, Y as np.array tuple."""

	time = []
	f5 = []
	header = True

	with open("data.csv", "rb") as fp:
		reader = csv.reader(fp, delimiter=",")
		for row in reader:
			if header:
				header = False
			else:
				#print row
				t = float(row[0])
				time.append(t)
				#Y_fitted.append(saturation_function())
				f5.append(float(row[1]))

	return (np.array(time), np.array(f5))

def show_data(X, Y, parameters=None, left_cutoff=None, right_cutoff=None):
	"""Show pretty plot of F5 vs. Time to make sure data imported correctly."""
	
	Y_fitted = []
	if parameters is None:
		parameters = [999999999999999, 0.189, 300, -290, -293, 5000, 1000, -11.1, 0.1]
	
	Y_fitted = saturation_function(X, parameters)

	print "Total loss is %.2f" % (get_loss(X, Y, parameters, left_cutoff, right_cutoff))
	plt.plot(X, Y, 'bo', X, Y_fitted, 'r+')
	plt.show()
	
if __name__ == "__main__":
	

	# data and user-defined parameters
	X, Y = load_data()
	left_cutoff = 1700
	right_cutoff = 9000

	old_X = X
	old_Y = Y

	# simply remove the data which should not be optimized for
	# make sure that Y is also trimmed
	Y = Y[np.where(X >= left_cutoff)]
	X = X[np.where(X >= left_cutoff)]

	Y = Y[np.where(X <= right_cutoff)]
	X = X[np.where(X <= right_cutoff)]
	
	#show_data(X, Y)

	#optimize(X, Y, left_cutoff, right_cutoff)
	new_params = optimize(X, Y)
	if new_params is not None:
		# showing optimization over the new data
		show_data(X, Y, new_params)

