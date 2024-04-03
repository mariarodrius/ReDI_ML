import numpy as np
import matplotlib.pyplot as plt

class MLRegression:
    def __init__(self):
        self.m = 0
        self.c = 0
        self.N = 1
        
    def train(self, Xtrain, Ytrain):
        self.X = Xtrain
        self.Y = Ytrain
        SumX = np.sum(self.X)
        SumY = np.sum(self.Y)
        SumXX = np.sum(self.X*self.X)
        SumXY = np.sum(self.X*self.Y)
        self.N = self.X.size
        self.m = (self.N*SumXY - SumX * SumY)/(self.N*SumXX - SumX*SumX)
        self.c = (SumY - self.m*SumX)/self.N
        print("ML Regression Model has been trained")
        print("m=",self.m," c=",self.c)
        plt.scatter(self.X, self.Y)
        yhat = self.m*self.X + self.c
        plt.plot(self.X,yhat, lw=4, c='orange')
        
    def trainPoly(self, Xtrain, Ytrain, koeff=3):
        self.X = Xtrain
        self.Y = Ytrain
        Z = np.polyfit(self.X, self.Y,koeff)
        print("ML Polynomial Regression Model has been trained")
        plt.scatter(self.X, self.Y)
        #print polynomial
        self.X = np.sort(self.X)
        yhat = 0
        #flip to reverse the order of the values of the polynomial coefficient to start with the highest coefficient instead of the lowest. 
        Z = np.flip(Z)
        for i in range(koeff+1):
            yhat += Z[i]*self.X**i
        plt.plot(self.X,yhat, lw=4, c='orange')

        
        
    def predict(self, Xtest):
        print("ML Regression Prediction")
        Y = self.m*Xtest + self.c
        print("Input = ", Xtest," Prediction = ",Y)
