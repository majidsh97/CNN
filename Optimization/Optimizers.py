import numpy as np

class SgdWithMomentum():

	def __init__(self,learning_rate,momentum_rate) -> None:
		self.v=0	
		self.momentum_rate = momentum_rate
		self.learning_rate=learning_rate
		pass

	def calculate_update(self,weight_tensor, gradient_tensor):
		self.v = self.momentum_rate*self.v - self.learning_rate * gradient_tensor
		new_weight = weight_tensor + self.v
		return new_weight

class Adam():
	
	def __init__(self,learning_rate,mu,rho) -> None:
		self.learning_rate=learning_rate
		self.mu=mu
		self.rho=rho
		self.v=0
		self.r=0
		self.t=1
		
		pass

	def calculate_update(self,weight_tensor, gradient_tensor):
		self.v = self.mu * self.v + (1-self.mu) * gradient_tensor
		self.r = self.rho * self.r + (1-self.rho)*gradient_tensor **2
		vhat = self.v/(1- self.mu**self.t)
		rhat=self.r/(1-self.rho**self.t)
		self.t+=1
		new_weight = weight_tensor - self.learning_rate * vhat/(np.sqrt(rhat) + 1e-8) #2.22044604925e-16)
		return new_weight
	pass


class Sgd():

    def __init__(self,learning_rate:float) -> None:
        self.learning_rate = learning_rate
        pass

    #calculate update(weight tensor, gradient tensor) that returns the updated weights
    #according to the basic gradient descent update scheme.
    def calculate_update(self,weight_tensor, gradient_tensor):
        new_weight = weight_tensor - gradient_tensor * self.learning_rate
        return new_weight

    