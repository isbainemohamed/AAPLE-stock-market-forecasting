Welcome to the AAPLE stock market forecasting project!

This project aims to predict the future price of AAPLE stock using two machine learning models: Long Short-Term Memory (LSTM) and Random Forest Regressor.

## Prerequisites
Before you begin, make sure you have the following installed on your machine:

Python 3.6 or higher
NumPy
pandas
scikit-learn
Keras (with TensorFlow backend)
You will also need to have access to historical stock price data for AAPLE. You can obtain this data from a financial database such as Yahoo Finance or Google Finance.

## Getting Started
Clone this repository to your local machine:

git clone https://github.com/isbainemohamed/AAPLE-stock-market-forecasting
Navigate to the root directory of the repository:

cd aapl-stock-forecasting
Download the stock price data and place it in the data directory. Make sure to name the file aapl.csv.

Run the following command to install the required packages:


pip install -r requirements.txt

Run the following command to train the LSTM model:

python train_lstm.py
Run the following command to train the Random Forest Regressor model:

python train_random_forest.py
Run the following command to make predictions using the trained models:

python predict.py


## Evaluation
The performance of the models will be evaluated using mean squared error (MSE). You can find the MSE for each model in the output of the training scripts (train_lstm.py and train_random_forest.py).

## Further Work
There are many ways you could extend this project, including:

Testing other machine learning models (e.g. support vector regression, gradient boosting)
Incorporating additional features (e.g. news articles, technical indicators)
Implementing a neural network model from scratch (instead of using Keras)
I hope you find this project useful! If you have any questions or suggestions, feel free to open an issue on GitHub.
