import sys
import time
import pandas as pd
import plotly.graph_objects as go
import math as math
from tqdm import tqdm

# Python terminal colors
CRED = '\033[91m'
CEND = '\033[0m'
CGREEN = '\33[92m'

df = pd.DataFrame()
fig = go.Figure()
# hyperparameters
theta0 = 0
theta1 = 0
ybar = 0
N = 0
SSmean = 0
learningRate = 0.5
iterations =200
def runTrainer():
    global df,theta0, theta1
    if (len(sys.argv) != 2):
        print(CRED+'USAGE: python trainer.py dataset.csv'+CEND)
    elif (not sys.argv[1].endswith('.csv')):
        print(CRED+'ERROR: file should have .csv extension'+CEND)
    else:
        try:
            df = pd.read_csv(sys.argv[1])
            print(CGREEN+'SUCCESS: Read .csv data'+CEND)
            leastSquares()
            print(CGREEN+'SUCCESS: Computed least squares'+CEND)
            plotData(df['mileage'],df['price'],'markers','Data','blue')
            print(CGREEN+'SUCCESS: Plotted Data'+CEND)
            plotData(df['mileage'],df['yhat'],'lines','Line of best fit','red')
            normalize()
            print(CGREEN+'SUCCESS: Plotted line of best fit'+CEND)
            print('Applying model to data...')
            gradientDescent()

            fig.show()
            print(CGREEN+'SUCCESS: Displayed graphs'+CEND)

        except Exception as e:
            print(CRED+'ERROR: Failed to process. Exiting program'+CEND)
            exit(e)
        
def computeVariables():
    global theta0, theta1, ybar
    # Calculate sums of rows
    colSums = df.sum(axis=0)
    # Calculate gradient
    theta1 = (N*colSums[3] - (colSums[0] * colSums[1])) / (N*colSums[2] - math.pow(colSums[0],2))
    # Calculate intersept
    theta0 = (colSums[1] - (theta1*colSums[0]))/N
    # Calulate ybar
    # ybar = colSums[1]/N

def plotData(x,y,mode,name, color):
    global fig
    fig.add_trace(
        go.Scatter(
            x = x, 
            y = y,
            name=name,
            mode=mode,
            line=dict(color=color, width=1)
        ))
    fig.update_layout(
        title=df.columns[1] + " based on " + df.columns[0] ,
        xaxis_title=df.columns[0],
        yaxis_title=df.columns[1],
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ))

def leastSquares():
    global theta0, theta1, ybar, N
    for index, row in df.iterrows():
        # Add x2 column
        df.at[index, 'x2'] = math.pow(df.at[index, 'mileage'],2)
        # Add xy column
        df.at[index, 'xy'] = df.at[index, 'mileage'] * df.at[index, 'price'] 
    #Number of rows/sample size
    N = df.shape[0] 
    # Calculate thetas 
    computeVariables()
    for index, row in df.iterrows():
        # Add yhat column
        df.at[index, 'yhat'] =  (theta1*df.at[index, 'mileage'])+theta0

def normalize():
    global df
    maxX = df['mileage'].max()
    minX = df['mileage'].min()
    maxY = df['price'].max()
    minY = df['price'].min()
    for index, row in df.iterrows():
        # Adjust yhat column
        df.at[index, 'normalized_x'] =  (df.at[index, 'mileage']-minX)/(maxX-minX)
        df.at[index, 'normalized_y'] =  (df.at[index, 'price']-minY)/(maxY-minY)

def computeDerivatives(iteration):
    global theta0, theta1
    maxX = df['mileage'].max()
    minX = df['mileage'].min()
    maxY = df['price'].max()
    minY = df['price'].min()
    for index, row in df.iterrows():
        # Adjust yhat with normalized data
        df.at[index, 'yhat'] = (theta1 * df.at[index, 'normalized_x']) + theta0
        # Add dfdb column
        df.at[index, 'dfdb'] = (df.at[index, 'yhat'] - df.at[index, 'normalized_y'])
        # Add dfdm column
        df.at[index, 'dfdm'] = ((df.at[index, 'yhat'] - df.at[index, 'normalized_y']) * df.at[index, 'normalized_x'])
        # Adjust yhat with denormalized data
        df.at[index, 'denormalized_y'] =  (df.at[index, 'yhat']*(maxY-minY))+minY
        # Add (y-yhat)^2 column
        # df.at[index, '(y-yhat)^2'] = math.pow(df.at[index, 'yhat'],2)

    # Plot current graph
    plotData(df['mileage'],df['denormalized_y'],'lines',f'iteration: {iteration}','green') 
   
    # Calculate sums of rows
    colSums = df.sum(axis=0)

    dfdb = colSums[7]
    dfdm = colSums[8]
    # MSE =  colSums[10] *(2/(2*N))
    return(dfdb,dfdm)

def gradientDescent():
    global theta0, theta1
    theta0 = theta1 = 0
    # print(N)
    for i in tqdm(range(iterations)):
        dfdb,dfdm = computeDerivatives(i)
        theta0 = theta0 - (learningRate * (1/N) *dfdb)
        theta1 = theta1 - (learningRate * (1/N) *dfdm)
        # print(f'{i} | theta0 {theta0} | theta1 {theta1} | dfdm {dfdm} | dfdb {dfdb} | MSE {MSE}')
    print(CGREEN+'SUCCESS: Applied model to data'+CEND)
    print('RESULTS:')
    print(f'theta0 {theta0} | theta1 {theta1}')
if __name__ == "__main__":
    runTrainer()


  