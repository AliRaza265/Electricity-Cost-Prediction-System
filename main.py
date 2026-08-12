import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,root_mean_squared_error, mean_squared_error
import pickle as pkl

# Understand the given data
read_csv = pd.read_csv(r"data_set\electricity_cost_dataset.csv")
print(read_csv.head())
print(read_csv.describe())
print(read_csv.info())
print(read_csv.columns)

# drop useless column 
read_csv.drop("issue reolution time",axis=1,inplace=True)

# Data Filtering
find_missing_values = read_csv.isnull().sum()
find_duplicate_value = read_csv.duplicated().sum()
print(find_duplicate_value)

# Encoding the column in integer
print(read_csv["structure type"].value_counts())
encoder = OrdinalEncoder(
categories= [["Residential", "Commercial", "Mixed-use", "Industrial"]],dtype= int)
read_csv["structure type"] = encoder.fit_transform(read_csv[["structure type"]])
print(read_csv["structure type"].sample(20))

# Split data for training and testing
X = read_csv.drop("electricity cost" , axis = 1)
y =read_csv["electricity cost"]
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
print(x_train.head())

# using MinMaxScaler for scaling the data
scaler = MinMaxScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)
print(x_test_scale)

# Using LinearRegression Model for Prediction
lr_model = LinearRegression()
lr_model.fit(x_train_scale,y_train)
lr_y_pred = lr_model.predict(x_test_scale)

# checking Accuracy of prection of the model
print(f"Linear Regression R2 Score : {r2_score(lr_y_pred,y_test)}")
print(f"Linear Regression Root Mean Squared Error : {root_mean_squared_error(lr_y_pred,y_test)}")
print(f"Linear Regression Mean Squared Error : {mean_squared_error(lr_y_pred,y_test)}")



# Manualy test the model
result = lr_model.predict(x_test_scale[10].reshape(1,-1))
print(result)
print(y_test.iloc[10])

# Import models for further use
pkl.dump(scaler,open(r"Models\Scaler.pkl","wb"))
pkl.dump(lr_model,open(r"Models\Model.pkl","wb"))