import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive")

cars = pd.read_csv("drive/kongkea/Dataset/car.csv")
cars.head(10)
cars.shape
cars.isnull().sum()
cars = cars.dropna(how="any")
cars.shape

res = "190Nm@ 2,000rpm".replace(".", "")
res = res.replace(",", "")
a = [int(s) for s in re.findall("\\d ", res)]
a
torque_list = cars["torque"].to_list()
torque_rpm = []


def extractingRPM(x):
    for item in x:
        res = item.replace(".", "")
        res = res.replace(",", "")
        temp = [int(s) for s in re.findall("\\d ", res)]
        torque_rpm.append(max(temp))


extractingRPM(torque_list)
print(torque_list[:2])
print(torque_rpm[:2])
cars["torque_rpm"] = torque_rpm
cars.head(2)
mil_list = cars["mileage"].to_list()
mil_kmpl = []


def extractingmil(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        mil_kmpl.append(max(temp))


extractingmil(mil_list)
print(mil_list[:2])
print(mil_kmpl[:2])
cars["mil_kmpl"] = mil_kmpl
cars.head(2)
engine_list = cars["engine"].to_list()
engine_cc = []


def extractingEngine(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        engine_cc.append(max(temp))


extractingEngine(engine_list)
print(engine_list[:2])
print(engine_cc[:2])
cars["engine_cc"] = engine_cc
cars.head(2)
power_list = cars["max_power"].to_list()
max_power = []


def extractingPower(x):
    for item in x:
        temp = []
        try:
            for s in item.split(" "):
                temp.append(float(s))
        except:
            pass
        max_power.append(max(temp))


extractingPower(power_list)
print(power_list[:2])
print(max_power[:2])
cars["max_power_new"] = max_power
cars.head(2)
cars_new = cars.drop(["mileage", "engine", "max_power", "torque"], axis=1)
cars_new.describe()
plt.figure(figsize=(8, 8))
sns.heatmap(cars_new.corr(), annot=True, cmap="viridis", linewidths=0.5)
cars_new["fuel"].value_counts()
cars_new["seller_type"].value_counts()
cars_new["transmission"].value_counts()
cars_new["owner"].value_counts()


def ref1(x):
    if x == "Manual":
        return 1
    else:
        return 0


cars_new["transmission"] = cars_new["transmission"].map(ref1)


def ref2(x):
    if x == "Individual":
        return 1
    elif x == "Dealer":
        return 0
    else:
        return -1


cars_new["seller_type"] = cars_new["seller_type"].map(ref2)


def ref3(x):
    if x == "Petrol":
        return 1
    elif x == "Diesel":
        return 0
    else:
        return -1


cars_new["fuel"] = cars_new["fuel"].map(ref3)
owners = pd.get_dummies(cars_new["owner"])
X = pd.concat([cars_new, owners], axis=1)
X.head()
y = X["sale_price"]
X = X.drop(["sale_price", "name", "owner"], axis=1)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X[:3000], y[:3000], test_size=0.2)
random_model = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1)
random_model.fit(Xtrain, ytrain)
y_pred = random_model.predict(Xtest)
random_model_accuracy = round(random_model.score(Xtrain, ytrain) * 100, 2)
print(round(random_model_accuracy, 2), "%")
random_model_accuracy1 = round(random_model.score(Xtest, ytest) * 100, 2)
print(round(random_model_accuracy1, 2), "%")
reg = LinearRegression()
reg.fit(Xtrain, ytrain)
print(round(reg.score(Xtrain, ytrain), 2))
print(round(reg.score(Xtest, ytest), 2))

saved_model = pickle.dump(
    random_model, open("drive/kongkea/Dataset/Models/CarSale.pickle", "wb")
)
