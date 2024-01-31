import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SwimwearSalesData:
    def __init__(self, file_path):
        self.sales_data = pd.read_csv(file_path)
        self.sales_data["Year"] = pd.to_datetime(self.sales_data["Years"], format='%m/%d/%Y')
        self.sales_data['Month'] = self.sales_data['Year'].dt.month
        self.sales_data = pd.get_dummies(self.sales_data, columns=['Month'], drop_first=True)
        self.split_data()

    def split_data(self):
        self.sales_data['measure_index'] = np.arange(len(self.sales_data))

        train_size = int(len(self.sales_data) * 0.75)
        self.train_data = self.sales_data.iloc[:train_size]
        self.test_data = self.sales_data.iloc[train_size:]

        self.X_train = self.train_data.drop(['Sales', 'Year', 'Years'], axis=1)
        self.y_train = self.train_data['Sales']
        self.X_test = self.test_data.drop(['Sales', 'Year', 'Years'], axis=1)
        self.y_test = self.test_data['Sales']


    def plot_sales(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.sales_data["Year"], self.sales_data["Sales"], "o-")
        plt.title("Swimwear Sales Trend")
        plt.xlabel("Year")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()


from sklearn.linear_model import LinearRegression

class SalesForecastModel:
    def __init__(self, sales_data):
        self.sales_data = sales_data
        self.regression_model = LinearRegression()
        self.fit_model()

    def fit_model(self):
        self.regression_model.fit(self.sales_data.X_train, self.sales_data.y_train)
        self.predict_sales()

    def predict_sales(self):
        self.predicted_train = self.regression_model.predict(self.sales_data.X_train)
        self.predicted_test = self.regression_model.predict(self.sales_data.X_test)
        self.evaluate_predictions()

    def evaluate_predictions(self):
        train_error = np.mean(np.abs(self.predicted_train - self.sales_data.y_train))
        test_error = np.mean(np.abs(self.predicted_test - self.sales_data.y_test))
        print(f"Mean Absolute Error on Training Set: {train_error:.2f}")
        print(f"Mean Absolute Error on Test Set: {test_error:.2f}")
        self.plot_predictions()

    def plot_predictions(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.sales_data.train_data['Year'], self.sales_data.train_data['Sales'], color='blue', label='Actual Sales', marker='o', linestyle='--')
        plt.plot(self.sales_data.test_data['Year'], self.sales_data.test_data['Sales'], color='orange', label='Future Sales', marker='o', linestyle='--')
        plt.plot(self.sales_data.train_data['Year'], self.predicted_train, color='green', label='Model Predictions on Train Set', linestyle='-')
        plt.plot(self.sales_data.test_data['Year'], self.predicted_test, color='red', label='Model Predictions on Test Set', linestyle='-')
        plt.title("Swimwear Sales Forecasting Model")
        plt.xlabel("Year")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)
        plt.show()


file_path = "vente_maillots_de_bain.csv"
sales_data = SwimwearSalesData(file_path)
forecast_model = SalesForecastModel(sales_data)
