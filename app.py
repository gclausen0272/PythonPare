from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd

app = Flask(__name__)
api = Api(app)
import matplotlib.pyplot as plt
import numpy as np 

def f(X,theta):
  #stops nans 
  h = 1/(1+np.exp(-(np.dot(theta, X.T)))) - 0.0000001
  h = np.array(h,dtype=float)
  return h 
def j(y_pred, y, X):
  return -(1/len(X)) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))- 0.0000001
def kecks_gradient(X,y,theta,alpha,maxEp,tol):
  losses = []
  deltacosts = []
  h = f(X,theta)
  jOld = j(h,y,X)
  for i in range(maxEp):
    theta = theta -(alpha/len(X))*((X.T).dot(h-y))
    h = f(X,theta)
    jNew = j(h,y,X)
    losses.append(jNew)
    err = abs((jNew-jOld)/jOld)
    deltacosts.append(err)
    if(abs(err)<tol):
      print("reached")
      print(err)
      return theta, losses, deltacosts
    jOld = jNew
  return theta, losses, deltacosts


# weights,losses,deltas = kecks_gradient(np.array(X_train), np.array(y_train),[1,1,1,1,1,1,1,1,1,1], .006,500,0.0000000101)   



class Users(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('j', required=True)  # add args
        
        args = parser.parse_args()  # parse arguments to dictionary
        return {'data': args['j']}, 200  # return data with 200 OK
    # methods go here
    pass
class Gradient(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('weights', required=True)  # add args
        parser.add_argument('X_train', required=True)  # add args
        parser.add_argument('y_train', required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary

        weights,losses,deltas = kecks_gradient(np.array(args['X_train'].split(','),dtype=float)), np.array(args['y_train'].split(','),dtype=float)),np.array(args['weights'].split(','),dtype=float)), .006,500,0.0000000101)   

        return {'data':list(weights)}, 200  # return data with 200 OK
    # methods go here
    pass
class Tune(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        
        parser.add_argument('weights', required=True)  # add args
        parser.add_argument('rest', required=True)  # add args
        parser.add_argument('outcome', required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary
        h = f(np.array(args['rest'].split(','),dtype=float),np.array(args['weights'].split(','),dtype=float))
        X = np.array(args['rest'].split(','),dtype=float)
        w = np.array(args['weights'].split(','),dtype=float)
        k = float(args['outcome'])
        theOut = np.array(h)-k
        theta = w -np.array((.05/len(X))*((X.T).dot(theOut)),dtype=float)
        return {'data':list(theta)}, 200  # return data with 200 OK
    # methods go here
    pass      
class Locations(Resource):
    # methods go here
    pass
    
api.add_resource(Users, '/users')  # '/users' is our entry point for Users
api.add_resource(Tune, '/tune')  # '/users' is our entry point for Users
api.add_resource(Gradient, '/gradient')  # '/users' is our entry point for Users

api.add_resource(Locations, '/locations')  # and '/locations' is our entry point for Locations
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = '2012',debug=True)  # run our Flask app