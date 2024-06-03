#------------------------------------------------------------------------------
# Hands-On Lab: Data Engineering with Snowpark
# Script:       05_fahrenheit_to_celsius_udf/app.py
# Author:       Jeremiah Hansen, Caleb Baechtold
# Last Updated: 1/9/2023
#------------------------------------------------------------------------------

# SNOWFLAKE ADVANTAGE: Snowpark Python programmability
# SNOWFLAKE ADVANTAGE: Python UDFs (with third-party packages)
# SNOWFLAKE ADVANTAGE: SnowCLI (PuPr)

import pandas as pd
import sys
from scipy.constants import convert_temperature
# Import the necessary libraries
from sklearn import datasets
from sklearn.linear_model import LinearRegression

def sklearn_example():
    # Load the iris dataset
    iris = datasets.load_iris()

    # Split the data into features and target
    X = iris.data
    y = iris.target

    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X, y)

    # Predict the label of a new data point
    new_data = [3, 5, 4, 2]
    prediction = knn.predict([new_data])

    # Print the prediction
    print(prediction)


def panda_example():
    df = pd.DataFrame({'Name': ['John', 'Mary', 'Bob'], 'Age': [20, 25, 30]})

    # Print the DataFrame
    print(df)

    # # Select a column
    # age = df['Age']

    # # Print the column
    # print(age)

    # # Calculate the mean of the column
    # mean_age = age.mean()

    # # Print the mean age
    # print(mean_age)


def main(temp_f: float) -> float:
    return convert_temperature(float(temp_f), 'F', 'C')


# For local debugging
# Be aware you may need to type-convert arguments if you add input parameters
if __name__ == '__main__':
    # panda_example()
    # sklearn_example()
    if len(sys.argv) > 1:
        print(main(*sys.argv[1:]))  # type: ignore
    else:
        print(main())  # type: ignore
