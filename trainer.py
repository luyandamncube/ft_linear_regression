import sys
import pandas as pd
# from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
import math as math

def readArgs(df):
    df = df
    if (len(sys.argv) != 2):
        print("USAGE: python .\trainer.py dataset.csv")
    elif (not sys.argv[1].endswith('.csv')):
        print("ERROR: file should have .csv extension")
    else:
        df = pd.read_csv(sys.argv[1])
        leastSquares(df,N,theta0,theta1)
def calcThetas(df, N):
    # Calculate sums of rows
    colSums = df.sum(axis=0)
    print(colSums)
    # Calculate gradient
    theta1 = (N*colSums[3] - (colSums[0] * colSums[1])) / (N*colSums[2] - math.pow(colSums[0],2))
    # Calculate intersept
    theta0 = (colSums[1] - (theta1*colSums[0]))/N
    return (theta0,theta1)

def plotData(df):
    fig = go.Figure()
    print(df)
    fig.add_trace(
        go.Scatter(
            x = df["mileage"], 
            y = df["price"],
            name='Data',
            mode='markers'
        ))
    fig.add_trace(
        go.Scatter(
            x = df["mileage"], 
            y = df["yhat"],
            name='Line of best fit'
        ))
    fig.show()

def leastSquares(df,N,theta0,theta1):
    for index, row in df.iterrows():
        # Add x2 column
        df.at[index, 'x2'] = math.pow(df.at[index, 'mileage'],2)
        # Add xy column
        df.at[index, 'xy'] = df.at[index, 'mileage'] * df.at[index, 'price'] 
    #Number of rows/sample size
    N = df.shape[0] 
    # Calculate thetas 
    theta0, theta1 = calcThetas(df, N)
    for index, row in df.iterrows():
        # Add yhat column
        df.at[index, 'yhat'] =  (theta1*df.at[index, 'mileage'])+theta0
    plotData(df)
    

df = pd.DataFrame()
theta0 = 0
theta1 = 0
N = 0

if __name__== "__main__":
  readArgs(df)


  