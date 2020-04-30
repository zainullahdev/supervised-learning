# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import Lasso,LinearRegression

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv("data.csv",header=None)
X = train_data.iloc[:,:6]
y = train_data.iloc[:,-1]
#print(X)
# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X,y)

lm = LinearRegression()
lm.fit(X,y)
print("normal one",lm.coef_)
# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)