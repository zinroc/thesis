from matplotlib import pyplot as plt
import csv

def get_hypothesis(X, fn, parameters):
	pass
	
def get_loss (Y, hypothesis):
	pass
	
def get_gradient (parameters):
	pass
	
def mod_maxwell(parameters, data):
	pass
	
def s_curve (parameters, data):
	pass
	
def c_vert(parameters, data):
	pass
	
def saturation_function(parameters, data):
	"""Parameters is a vector in the following order:
		* K0
		* K1
		* N0
		* N1
		* N2
		
		data is vector in the order:
			* time
			* f-5
			* Tbuffer
			* Ftime
		
	"""
	
	pass
	
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
	header = True

	with open("data.csv", "rb") as fp:
		reader = csv.reader(fp, delimiter=",")
		for row in reader:
			if header:
				header = False
			else:
				print row
				time.append(float(row[0]))
				f5.append(float(row[1]))
			
	plt.plot(time, f5)
	plt.show()
	
if __name__ == "__main__":
	show_data()
