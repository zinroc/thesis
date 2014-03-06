from matplotlib import pyplot as plt
import csv
import math

def get_hypothesis(X, parameters):
	return [saturation_function(parameters, time) for time in X]
	
def get_loss (X, Y, parameters, left_cutoff, right_cutoff):
	loss = 0
	c = 0

	for time in sorted(X):
		if time > right_cutoff:
			break
			
		if time >= left_cutoff:
			loss += (Y[c] - saturation_function(parameters, time)) ** 2
			
		c += 1
			
	return loss
	
def get_gradient (parameters):
	"""
	df_dt = ((1/(N0*(K0+K1)^2))*(K0^2)*math.exp((K0*K1*(dt-t))/(N0(K0+K1)))+1/N1)*(-1/(C3*math.exp(r*(dt-t))+1))+(C3*r*(math.exp(r*(dt-t)))*((1/K1)*((K0/(K0+K1))*math.exp((K0*K1*(dt-t)/(N0*(K0+K1))))-1))+(dt-t)/N1)/((C3*math.exp(r*(dr-t))+1)^2)+1/N2
	df_dk0 = math.exp((K0*K1*(dt-t))/(N0*(K0+K1)))*(dt*K0*K1+K0*(N0-K1*t)+K1*N0)/(((K0+K1)^3)*(C3*N0*math.exp(r*(dt-t))+N0))
	df_dk1 = (-1/((K1^2)*(C3*math.exp(r*(dt-t))+1)))*(K0*K1*math.exp((K0*K1*(dt-t))/(N0*(K0+K1)))*(-dt*(K0^2)+(K0^2)*t+K0*N0+K1*N0)/(N0*(K0+K1)^3)+((K0/(K0+K1))*math.exp((K0*K1*(dt-t))/(N0*(K0+K1)))-1))
	df_dN0 = -(K0^2)*(dt-t)*math.exp((K0*K1*(dt-t))/(N0*(K0+K1)))/((N0^2)*((K0+K1)^2)*(C3*math.exp(r*(dt-t))+1))
	df_dN1 = (t-dt)/((N1^2)*(C3*math.exp(r*(dt-t))+1))
	df_dN2 = (dt-t)/(N2^2)
	OPTIONAL - finding dt
	df_ddt = ((1/(N0*(K0+K1)^2))*(K0^2)*math.exp((K0*K1*(dt-t))/(N0(K0+K1)))+1/N1)*(-1/(C3*math.exp(r*(dt-t))+1))+(C3*r*(math.exp(r*(dt-t)))*((1/K1)*((K0/(K0+K1))*math.exp((K0*K1*(dt-t)/(N0*(K0+K1))))-1))+(dt-t)/N1)/((C3*math.exp(r*(dr-t))+1)^2)-1/N2
	
	return [df_dk0, df_dk1, ]"""
	
	pass
	
def mod_maxwell(parameters, time):

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
	return -1*((1/K1)*(1-(K0/(K0+K1))*math.exp(-1*(time-dt)/(T)))+(time-dt)/N1)
	
def s_curve (parameters, time):
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
	return 1/(1+C3*math.exp((-time + dt)*r))
	
def steady(parameters, time):
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
	
def saturation_function(parameters, time):
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
	
	return mod_maxwell(parameters,time)*s_curve(parameters,time) + steady(parameters,time)
	
def main():
	# set max # iterations

	# load data

	# get initial guess for parameters
	# get hypothesis
	# get loss
	
	# while # iterations < max # iterations
	# 	get gradient 	
	# 	update parameters
	# 	get hypothesis
	# 	get loss
	#	if loss very close to 0, break
	
	# return parameters
	
	pass
	
def show_data():
	"""Show pretty plot of F5 vs. Time to make sure data imported correctly."""

	time = []
	f5 = []
	y_fitted = []
	header = True
	parameters = [999999999999999, 0.189, 300, -290, -293, 5000, 1000, -11.1, 0.1]

	with open("data.csv", "rb") as fp:
		reader = csv.reader(fp, delimiter=",")
		for row in reader:
			if header:
				header = False
			else:
				#print row
				t = float(row[0])
				time.append(t)
				f5.append(float(row[1]))
				y_fitted.append (saturation_function (parameters, t))

	print "Total loss is %.2f" % (get_loss(time, f5, parameters, 1700, 9000))
	plt.plot(time, f5, 'bo', time, y_fitted, 'r+')
	plt.show()
	
if __name__ == "__main__":
	show_data()
