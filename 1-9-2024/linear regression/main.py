import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
# from sklearn.linear_model import LinearRegression as SklearnLR

df = pd.read_csv('test.csv')

print(df[df["SalePrice"]<159664.5].count())

# for col in df.columns:
#     plt.hist(df[df["SalePrice"]>159664.5][col], label='High', alpha=0.5, density=False)
#     plt.hist(df[df["SalePrice"]<159664.5][col], color='red', label='Low', alpha=0.5, density=False)
#     plt.title(col)
#     plt.ylabel("Probability")
#     plt.xlabel(col)
#     plt.legend()
#     plt.show()

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    X = preprocessing.StandardScaler().fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    
    return data, X, y

train, X_train, y_train = scale_dataset(df, oversample=False)
print(sum(y_train > 159664.5))
print(sum(y_train < 159664.5))
