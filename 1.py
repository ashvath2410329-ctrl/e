import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import get_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Dense

tf.keras.utils.set_random_seed(42)

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(get_file("airline-passengers.csv", origin=URL))
df["Month"] = pd.to_datetime(df["Month"])
print(df.shape)
print(df.isnull().sum())

plt.figure(figsize=(10, 4))
plt.plot(df["Month"], df["Passengers"])
plt.title("Airline passengers")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.show()

data = df["Passengers"].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def csq(data, window=12):
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(x), np.array(y)

x, y = csq(data_scaled, 12)
split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]
print(x_train.shape, x_test.shape)

y_train_real = scaler.inverse_transform(y_train)
y_test_real = scaler.inverse_transform(y_test)

layers = {"RNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}
results = {}

for name, Layer in layers.items():
    model = Sequential([Input(shape=(12, 1)), Layer(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x_train, y_train, epochs=100, batch_size=8,
                        validation_data=(x_test, y_test), verbose=0)
    train_pred = scaler.inverse_transform(model.predict(x_train, verbose=0))
    test_pred = scaler.inverse_transform(model.predict(x_test, verbose=0))
    results[name] = {
        "history": history.history,
        "test_pred": test_pred,
        "metrics": {
            "Params": model.count_params(),
            "Train RMSE": np.sqrt(mean_squared_error(y_train_real, train_pred)),
            "Test RMSE": np.sqrt(mean_squared_error(y_test_real, test_pred)),
            "Test MAE": mean_absolute_error(y_test_real, test_pred),
            "Test MSE (scaled)": mean_squared_error(y_test, scaler.transform(test_pred)),
        },
    }

table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.round(3)
print(table)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, r) in zip(axes, results.items()):
    ax.plot(r["history"]["loss"], label="train")
    ax.plot(r["history"]["val_loss"], label="test")
    ax.set_title(f"{name} loss")
    ax.set_xlabel("Epoch")
    ax.legend()
axes[0].set_ylabel("MSE (scaled)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test_real, label="Actual", marker=".")
for name, r in results.items():
    plt.plot(r["test_pred"], label=f"{name} (RMSE {r['metrics']['Test RMSE']:.1f})")
plt.xlabel("Test time step")
plt.ylabel("Passengers")
plt.title("RNN vs LSTM vs GRU")
plt.legend()
plt.show()
