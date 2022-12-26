# AAPLE stock market forecasting using Machine Learning

*In this project, we use Python and machine learning algorithms to analyze and forecast the stock prices of Apple Inc. (AAPL). The project is divided into several steps:*

## 1- Preparing historic pricing data: 

We clean and prepare the stock price data for analysis. This process, also known as preprocessing, is crucial for ensuring that the data is ready for analysis and modeling.

To perform this step, we will need to import the following libraries:

* NumPy: A library for working with large, multi-dimensional arrays and matrices of numerical data.
* pandas: A library for fast, flexible, and expressive data manipulation and analysis.

To install these libraries, you can use the following command:

```bash
pip install numpy pandas
```

Here is an example of how we might load and prepare the data using these libraries:

```python
import numpy as np
import pandas as pd
```

```python
# Load stock price data into a Pandas DataFrame
df = pd.read_csv('data/aapl_stock_prices.csv')

# Handle missing values
df = df.dropna()

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create new column 'Year'
df['Year'] = df['Date'].dt.year
```


## 2- Exploratory data analysis: 

We identify trends, patterns, or seasonalities in the data. This step is important for gaining insights into the data and understanding the underlying relationships within it.

To perform this step, we will need to import the following library:

* matplotlib: A library for creating static, animated, and interactive visualizations.
To install this library, you can use the following command:

```bash
pip install matplotlib
```

Here is an example of how we might use these libraries to perform exploratory data analysis:

```python
# Calculate summary statistics
df.describe()

# Group data by year and calculate mean
df.groupby('Year').mean()

# Plot stock price over time
import matplotlib.pyplot as plt
plt.plot(df['Date'], df['Close'])
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.show()
```

## 3- Data visualization: 

We use various techniques to better understand the data and identify patterns or trends. This step is important for communicating the insights and findings from the data analysis.

To perform this step, we will need to import the following library:

seaborn: A library for creating statistical graphics and visualizations.
To install this library, you can use the following command:

```bash
pip install seaborn
```

Here is an example of how we might use these libraries to create data visualizations:

```python
import seaborn as sns

# Create box plot to compare stock prices by year
sns.boxplot(x='Year', y='Close', data=df)
plt.show()

# Create heatmap to show correlations between features
corr = df.corr()
sns.heatmap(corr)
plt.show()
```
## 4-Model development: 

We develop four different forecasting models using the following algorithms: Long Short-Term Memory (LSTM), Linear Regression with pandas_ta, Support Vector Machines (SVM), and Autoregressive Integrated Moving Average (ARIMA).

To perform this step, we will need to import the following libraries:

* scikit-learn: A library for machine learning in Python.
* pandas_ta: A library for technical analysis and visualization of financial data in Pandas.
* ta-lib: A library for technical analysis of financial market data.
* Keras (with TensorFlow backend): A library for building and training neural networks.

To install these libraries, you can use the following commands:

```bash
pip install scikit-learn pandas_ta ta-lib
pip install keras tensorflow
```

Here is an example of how we might use these libraries to develop a machine learning model:

```python
# Select features
X = df[['Year', 'Open', 'High', 'Low', 'Close']]
y = df['Close']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

## 5- Model performance: 

We compare the performance of the different models and discuss the implications of our findings. This step is important for evaluating the accuracy and effectiveness of the models and deciding which one to use for forecasting.

To perform this step, we will need to import the following library:

* scikit-learn: A library for machine learning in Python.

To install this library, you can use the following command:

```bash
pip install scikit-learn
```

Here is an example of how we might use these libraries to compare model performance:

```python
# Calculate evaluation metric for each model
from sklearn.metrics import mean_squared_error
mse1 = mean_squared_error(y_test, model1.predict(X_test))
mse2 = mean_squared_error(y_test, model2.predict(X_test))
mse3 = mean_squared_error(y_test, model3.predict(X_test))
mse4 = mean_squared_error(y_test, model4.predict(X_test))

# Create bar plot to compare model performance
import matplotlib.pyplot as plt
plt.bar(['Model 1', 'Model 2', 'Model 3', 'Model 4'], [mse1, mse2, mse3, mse4])
plt.ylabel('Mean Squared Error')
plt.show()
```

Once you have the required libraries installed, you can clone this repository and navigate to the project directory. From there, you can run the Jupyter notebook "Final document - Technical Analysis of AAPL Stocks.ipynb" to see the code and results of the project.

The project is organized into the following directories:

data/: Contains the stock price data.
notebooks/: Contains the Jupyter notebook with the code and results of the project.

This project was created by [Mohamed Isbaine](https://www.linkedin.com/in/mohamed-isbaine/)

Feel free to contact me if anything is wrong or if anything needs to be changed: isbainemouhamed@gmail.com
