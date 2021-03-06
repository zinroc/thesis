#!/usr/bin/python

from __future__ import print_function
from matplotlib import pyplot as plt
import csv
import math
import numpy as np
import random
import sys
import scipy.optimize
import warnings

TUNED_PARAMS = ["k0", "k1", "n0", "n1", "n2"]
USER_DEFINED_PARAMS = set(["dt", "c3", "cvert", "r", "graph_left_cutoff", "graph_right_cutoff", "optimization_left_cutoff", "optimization_right_cutoff", "cneg"])
INPUT_FILE = "parameters.csv"
OUTPUT_FILE = "new_parameters.csv"
#INPUT_FILE = "new_parameters.csv"
#OUTPUT_FILE = "super_new_parameters.csv"
DATA_FILE = "data.csv"
MAX_ITER = 5000

"""
These are the methods specific to the function being optimized
"""

################################################################################################
	
def get_saturation_gradient (t, parameters, aux_parameters):
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
	dt = aux_parameters["dt"]
	T = N0 / K1 + N0 / K0
	C3 = aux_parameters["c3"]
	Cvert = aux_parameters["cvert"]
	r = aux_parameters["r"]
	Cneg = aux_parameters["cneg"]

	df_dk0 = Cneg*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))*(dt*K0*K1+K0*(N0-K1*t)+K1*N0)/(((K0+K1)**3)*(C3*N0*np.exp(r*(dt-t))+N0))
	df_dk1 = (1/((K1**2)*(C3*np.exp(r*(dt-t))+1)))*(Cneg)*(K0*K1*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))*(-dt*(K0**2)+(K0**2)*t+K0*N0+K1*N0)/(N0*(K0+K1)**3)+((K0/(K0+K1))*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))-1))
	df_dN0 = (Cneg)*(K0**2)*(dt-t)*np.exp((K0*K1*(dt-t))/(N0*(K0+K1)))/((N0**2)*((K0+K1)**2)*(C3*np.exp(r*(dt-t))+1))
	df_dN1 = (Cneg)*(t-dt)/((N1**2)*(C3*np.exp(r*(dt-t))+1))
	df_dN2 = (dt-t)/(N2**2)
	#OPTIONAL - finding dt
	df_ddt = ((1/(N0*(K0+K1) ** 2))*(K0 ** 2)*np.exp((K0*K1*(dt-t))/(N0 * (K0+K1)))+1/N1)*(-1/(C3*np.exp(r*(dt-t))+1))+(C3*r*(np.exp(r*(dt-t)))*((1/K1)*((K0/(K0+K1))*np.exp((K0*K1*(dt-t)/(N0*(K0+K1))))-1))+(dt-t)/N1)/((C3*np.exp(r*(dt-t))+1) ** 2)-1/N2
	
	return np.array([df_dk0, df_dk1, df_dN0, df_dN1, df_dN2, df_ddt])
	
def mod_maxwell(time, parameters, aux_parameters):

	K0 = parameters[0]	
	K1 = parameters[1]
	N0 = parameters[2]
	N1 = parameters[3]
	N2 = parameters[4]
	dt = aux_parameters["dt"]
	T = N0 / K1 + N0 / K0  # very large, when K1 very small; and this is a positive #
	C3 = aux_parameters["c3"]
	Cvert = aux_parameters["cvert"]
	r = aux_parameters["r"]
	Cneg = aux_parameters["cneg"]
	adjusted_t = time - dt # never larger than about 20, 000

	#assert (np.all(T > 0))
	#assert (np.all(time - dt > 0))

	return Cneg * ((1/K1) * (1 - (K0/(K0+K1)) * np.exp(-1 * (adjusted_t) / T)) + (adjusted_t)/N1)
	#return -1*((1/K1)*(1-(K0/(K0+K1)) * np.exp((dt - time) / T)) + (time - dt) / N1)
	
def s_curve (time, parameters, aux_parameters):
	#K0 = parameters[0]
	#K1 = parameters[1]
	#N0 = parameters[2]
	#N1 = parameters[3]
	#N2 = parameters[4]
	dt = aux_parameters["dt"]
	#T = N0 / K1 + N0 / K0
	C3 = aux_parameters["c3"]
	Cvert = aux_parameters["cvert"]
	r = aux_parameters["r"]
	
	return 1/(1+C3*np.exp((-time + dt)*r))
	
def steady(time, parameters, aux_parameters):
	#K0 = parameters[0]
	#K1 = parameters[1]
	#N0 = parameters[2]
	#N1 = parameters[3]
	N2 = parameters[4]
	dt = aux_parameters["dt"]
	#T = N0 / K1 + N0 / K0
	C3 = aux_parameters["c3"]
	Cvert = aux_parameters["cvert"]
	r = aux_parameters["r"]
	
	# note that in Sergei's initial calculation, this is just time / N2 + Cvert
	return Cneg * ((time-dt)/N2 + Cvert)
	
def saturation_function(time, parameters, aux_parameters):
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
		* Cneg
		
	"""
	
	return mod_maxwell(time, parameters, aux_parameters) * s_curve(time, parameters, aux_parameters) + steady(time, parameters, aux_parameters)

##################################################################

def get_obj_function(X, Y, aux_parameters):
	"""Returnt the function which must be minimized."""

	return lambda parameters: np.sum((Y - saturation_function(X, list(parameters), aux_parameters))** 2)

def get_gradient_function(X, Y, aux_parameters):
	"""Return the function which evaluates gradient."""

	return lambda parameters: get_gradient(X, Y, parameters, aux_parameters)


"""
Function-agnostic methods.
This is the meat of the program
"""

def get_gradient(X, Y, parameters, aux_parameters):
	"""Return gradient over all parameters.
	Will return 0 for all parameters other than first 5, because not optimizing those."""

	obj_grad = np.zeros(len(parameters), dtype=np.float64)

	# compute gradients for all data points
	grad = get_saturation_gradient(X, parameters, aux_parameters)

	for i in range(len(obj_grad)):
		obj_grad[i] = np.sum(-2 * Y * grad[i] + 2 * saturation_function(X, parameters, aux_parameters) * grad[i])

	return obj_grad

def get_loss(X, Y, parameters, aux_parameters):
	"""Return float that gives loss of function being optimized compared to ideal scenario.
	"""

	return np.sum((Y - saturation_function(X, parameters, aux_parameters)) ** 2)

def randomize_parameters():
	# placeholders
	parameters = np.zeros(5, dtype=np.float64) 
	# fill in the last several, as they are user-generated
	#parameters[5] = 5000.0
	#parameters[6] = 1000.0
	#parameters[7] = -11.1
	#parameters[8] = 0.1

	# come up with random values for the rest
	#parameters[0] = math.exp(random.randint(1, 15)) * 1.0
	parameters[0] = math.exp(10) * 1.0
	parameters[1] = math.exp(random.randint(1, 15)) * 1.0
	parameters[2] = math.exp(random.randint(1, 15)) * 1.0
	parameters[3] = -1.0 * math.exp(random.randint(1, 15))
	parameters[4] = -1.0 * math.exp(random.randint(1, 15))
	return parameters
	
def read_parameters_from_file():
	"""Read user-defined parameters from a file."""

	#placeholders
	parameters = np.zeros(len(TUNED_PARAMS)) * 1.0
	d = {}

	with open(INPUT_FILE, "rb") as fp:
		for line in fp:
			if line.startswith("#") or len(line.strip()) == 0:
				continue
			row = [item.strip() for item in line.split("=")]
			if row[0].lower().strip() in TUNED_PARAMS:
				i = TUNED_PARAMS.index(row[0].strip().lower())
				parameters[i] = np.float64(row[1].strip())
			elif row[0].lower().strip() in USER_DEFINED_PARAMS:
				d[row[0].lower().strip()] = np.float64(row[1].strip())
			else:
				print("[ERROR] Read invalid item %s" % row[0])
				sys.exit(1)

	missing_keys = USER_DEFINED_PARAMS.difference(set(d.keys()))
	if len(missing_keys) > 0:
		print("[ERROR] User must specify the following keys:")
		print(", ".join( missing_keys))
		sys.exit(1)
				
	# show parameters for tuning
	for i in range(len(TUNED_PARAMS)):
		print("%s = %s" % (TUNED_PARAMS[i], str(parameters[i])))

	# show user-defined fixed parameters
	for k in sorted(d.keys()):
		v = d[k]
		print("%s = %s" % (k, str(v)))
		
	# get confirmation
	user_in = "x"
	
	while user_in not in ["y", "n"]:
		user_in = raw_input("This is OK? [y/n] ")
		
	if user_in == "y":
		return parameters, d
	else:
		sys.exit(1)

def set_parameters():
	return read_parameters_from_file()
	
def scipy_optimize(X, Y, aux_parameters, parameters):
	print("[intial parameters] %s" % str(parameters))
	print("[initial loss] %.1f" % get_loss(X, Y, parameters, aux_parameters))
	obj_fn = get_obj_function(X, Y, aux_parameters)
	grad_obj_fn = get_gradient_function(X, Y, aux_parameters)
	new_params = scipy.optimize.fmin_bfgs(obj_fn, parameters, fprime=grad_obj_fn, maxiter=MAX_ITER)

	print("[DEBUG] Finished BFGS optimization")
	print("[DEBUG] Loss here is %.2f" % get_loss(X, Y, new_params, aux_parameters))

	# now will try to do this again with a different algo
	new_fn = lambda p: (Y - saturation_function (X, p, aux_parameters))
	optimized_params = scipy.optimize.leastsq (new_fn, new_params)[0]
	print("[DEBUG] Finished least squares optimization")
	print("[DEBUG] parameters = %s" % str(optimized_params))
	print("[DEBUG] final loss = %.2f" % get_loss(X, Y, optimized_params, aux_parameters))
	return optimized_params
	
def load_data():
	"""Return X, Y as np.array tuple."""

	time = []
	f5 = []
	header = True

	with open(DATA_FILE, "rb") as fp:
		reader = csv.reader(fp, delimiter=",")
		for row in reader:
			if header:
				header = False
			else:
				#print row
				t = np.float64(row[0])
				time.append(t)
				#Y_fitted.append(saturation_function())
				f5.append(np.float64(row[1]))

	return (np.array(time), np.array(f5))

def show_data(X, Y, parameter_guess, optimized_params, aux_parameters):
	"""Show pretty plot of F5 vs. Time to make sure data imported correctly."""
	
	guess_fitted = saturation_function(X, parameter_guess, aux_params)
	optimized_fitted = saturation_function(X, optimized_params, aux_params)

	#print aux_params
	#print "Total loss is %.2f" % (get_loss(X, Y, parameters, aux_params))
	plt.plot(X, Y, 'bo', X, guess_fitted, 'r+', X, optimized_fitted, 'r-')
	plt.show()

	plt.plot(X, Y, 'bo', X, optimized_fitted, 'r-')
	plt.show()
	
def trim_data (X, Y, left_cutoff, right_cutoff):
	new_Y = Y[np.where(X >= left_cutoff)]
	new_X = X[np.where(X >= left_cutoff)]
	
	new_Y = new_Y[np.where(new_X <= right_cutoff)]
	new_X = new_X[np.where(new_X <= right_cutoff)]
	return new_X, new_Y
	
if __name__ == "__main__":
	# load data from file
	raw_X, raw_Y = load_data()
	# load user-defined parameters from file
	parameter_guess, aux_params = set_parameters()

	optimization_left_cutoff = 1200
	optimization_right_cutoff = 9000
	
	graph_left_cutoff = 0
	graph_right_cutoff = 10000

	# simply remove the data which should not be optimized for
	# make sure that Y is also trimmed
	optimization_X, optimization_Y = trim_data(raw_X, raw_Y, aux_params["optimization_left_cutoff"], aux_params["optimization_right_cutoff"])
	display_X, display_Y = trim_data(raw_X, raw_Y, aux_params["graph_left_cutoff"], aux_params["graph_right_cutoff"])
	
	new_params = scipy_optimize(optimization_X, optimization_Y, aux_params, parameter_guess)
	
	if new_params is not None:
		# write the parameters to disk
		with open (OUTPUT_FILE, "w") as fp:
			for i in range(len(TUNED_PARAMS)):
				fp.write("%s = %s\n" % (TUNED_PARAMS[i].upper(), str(new_params[i])))
				print("%s = %s" % (TUNED_PARAMS[i].upper(), str(new_params[i])))
			for k in sorted(USER_DEFINED_PARAMS):
				fp.write("%s = %s\n" % (k, str(aux_params[k])))
	
		show_data(display_X, display_Y, parameter_guess, new_params, aux_params)
			
