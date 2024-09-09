# Algorithm for gradient descent
# Step 1: Compute the derivative of the loss function
# Step 2: Update the parameters
# Step 3: Compute the step size: Step size = learning rate * slope
# Step 4: Calculate the parameters: New Parameter = Old Parameter - Step Size
# Step 5: Repeat until convergence ( Step size < 0.0001)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
import tensorflow as tf

df = pd.read_csv('train.csv')

def plot_data(df):
    for label in df.columns[:-1]:
        plt.scatter(df[label], df['SalePrice'])
        plt.xlabel(label)
        plt.ylabel('SalePrice')
        plt.title(label)
        plt.legend()
        plt.show()

def plotHistogram(df):
    for col in df.columns:
        plt.hist(df[df["SalePrice"]>159664.5][col], label='High', alpha=0.5, density=False)
        plt.hist(df[df["SalePrice"]<159664.5][col], color='red', label='Low', alpha=0.5, density=False)
        plt.title(col)
        plt.ylabel("Probability")
        plt.xlabel(col)
        plt.legend()
        plt.show()


def get_xy(dataframe, y_label, x_label=None):
    dataframe = copy.deepcopy(dataframe)
    if x_label is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values
    else:
        if len(x_label) == 1:
            X = dataframe[x_label[0]].values.reshape(-1, 1)
        else:
            X = dataframe[x_label].values
    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))
    return data, X, y

train, val = np.split(df.sample(frac=1), [int(.6*len(df))])

data, X_GrLivArea, y_GrLivArea = get_xy(train, 'SalePrice', x_label=["GrLivArea"])
data, X_val, y_val = get_xy(val, 'SalePrice', x_label=["GrLivArea"])

data, X_all, y_all = get_xy(df, 'SalePrice', x_label=df.columns[:-1])

def plotAgainstOne(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    plt.scatter(X, y, label="Data")
    plt.plot(X, reg.predict(X), color='red', label="Fit", linewidth=3)
    plt.xlabel("GrLivArea")
    plt.ylabel("SalePrice")
    plt.title("GrLivArea vs SalePrice")
    plt.legend()
    plt.show()

    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, y))

plotAgainstOne(X_GrLivArea, y_GrLivArea)

def plotAgainstAll():
    temp_reg = LinearRegression()
    temp_reg.fit(X_all, y_all)

    print(temp_reg.coef_)
    print(temp_reg.intercept_)
    print(temp_reg.score(X_all, y_all))



# Using Nueral Network
# temp_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
# temp_normalizer.adapt(X_GrLivArea.reshape(-1))

# temp_nn_model = tf.keras.Sequential([
#     temp_normalizer,
#     tf.keras.layers.Dense(1)
# ])

# temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

