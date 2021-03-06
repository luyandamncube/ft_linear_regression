# ft_linear_regression
Data Science: Linear Regression

<del>Really hope I ACTUALLY finish this project :(</del>. I did it! :)

Don't really know what I'm doing with this one but the plan is to accelerate my data science learning using the `ft_linear_regression` and hopefully the `DSLR` (Datascience X Logistic Regression) modules at 42. Found one other [person](https://github.com/SpenderJ/Linear_Regression) doing the project on github. This is the first project in the deep learning branch at 42. 

## Linear Regression
* [Linear Regression](https://www.statisticssolutions.com/what-is-linear-regression/) is a topic in statistics that is used as a predictive analysis tool.
* It aims to form relationships between data points and:
1. determining the strength of predictors, 
2. forecasting an effect,
3. trend forecasting.

## Objective
This project aims to predict the price of a car based on it's mileage.

## Topics covered
* Statistics
* Predictive Analysis
* Regression Modelling

## Python Dependencies
* `pandas`, `plotly`, `math`, 
* `tqdm` (for progress bar in terminal)
![tqdm](progress.gif)

## Usage

1. Start the trainer. Run `python .\trainer.py .\dataset_00.csv`, where `.\dataset_00.csv` is the training dataset in csv format:

| mileage       | price         |
| ------------- |:-------------:|
| 240000        | 3650          |
| 139800        | 3800          |
| 150500        | 4400          |

2. Start the predictor. Run `python .\predictor.py <SAMPLE_MILEAGE>`, where `<SAMPLE_MILEAGE>`  is the mileage is a number (unsigned).

## Result

This is a sample data set where learning rate = 0.5, iterations = 200

<p align="center">
  <img src="model.png"/>
</p>

## Resources
[DataSci](https://github.com/luyandamncube/DataSci), My repo for all python and data science learning
