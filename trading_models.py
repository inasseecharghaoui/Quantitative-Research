import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ============================================================
# DATA
# ============================================================

df = pd.read_excel("data/commodities.xlsx", sheet_name="Data")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Commodity", "Date"]).reset_index(drop=True)
print(df.shape)

# ============================================================
# MODEL 1 : RSI MEAN-REVERSION
# ============================================================

def signal_rsi(rsi):
    if rsi < 30:
        return "BUY"
    elif rsi > 70:
        return "SELL"
    else:
        return "HOLD"

df["RSI_Model"] = df["RSI_14"].apply(signal_rsi)

# ============================================================
# MODEL 2 : MULTI-FACTOR SCORE
# ============================================================

df["S_RSI"] = np.where(df["RSI_14"] < 30, 1, np.where(df["RSI_14"] > 70, -1, 0))
df["S_Momentum"] = np.where(df["Momentum_12d"] > 0, 1, np.where(df["Momentum_12d"] < 0, -1, 0))
df["S_MACD"] = np.where(df["MACD"] > df["MACD_Signal"], 1, np.where(df["MACD"] < df["MACD_Signal"], -1, 0))
df["S_Carry"] = np.where(df["Carry"] > 0, 1, np.where(df["Carry"] < 0, -1, 0))

df["Score"] = df["S_RSI"] + df["S_Momentum"] + df["S_MACD"] + df["S_Carry"]

def signal_score(score):
    if score >= 3:
        return "BUY"
    elif score <= -3:
        return "SELL"
    else:
        return "HOLD"

df["MultiFactor_Model"] = df["Score"].apply(signal_score)
print(df["MultiFactor_Model"].value_counts())

# ============================================================
# TARGET : NEXT-DAY RETURN
# ============================================================

df["Future_Return"] = df.groupby("Commodity")["Price"].shift(-1) / df["Price"] - 1

threshold = 0.002  # 0.2 %

df["Target"] = np.where(
    df["Future_Return"] > threshold, 1,
    np.where(df["Future_Return"] < -threshold, -1, 0)
)

features = [
    "Price", "Volume", "Volatility_30d", "Momentum_12d", "RSI_14",
    "MACD", "Carry", "Roll_Yield", "Term_Spread", "ZScore_20"
]

data = df.dropna(subset=features + ["Future_Return"]).copy()

# Chronological split (80% train / 20% test) to avoid look-ahead bias
dates_unique = sorted(data["Date"].unique())
cutoff = dates_unique[int(len(dates_unique) * 0.8)]

train = data[data["Date"] < cutoff].copy()
test = data[data["Date"] >= cutoff].copy()

X_train, y_train = train[features], train["Target"]
X_test, y_test = test[features], test["Target"]

print("Train :", X_train.shape)
print("Test  :", X_test.shape)
print("Split date :", cutoff)

# ============================================================
# MODEL 3a : LINEAR REGRESSION
# ============================================================

def regression_to_signal(x):
    if x > 0.20:
        return 1
    elif x < -0.20:
        return -1
    else:
        return 0

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
test["Linear_Pred"] = [regression_to_signal(x) for x in linear_model.predict(X_test)]

# ============================================================
# MODEL 3b : RIDGE REGRESSION
# ============================================================

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])
ridge_model.fit(X_train, y_train)
test["Ridge_Pred"] = [regression_to_signal(x) for x in ridge_model.predict(X_test)]

# ============================================================
# MODEL 3c : NEURAL NETWORK (KERAS)
# ============================================================

scaler_nn = StandardScaler()
X_train_nn = scaler_nn.fit_transform(X_train)
X_test_nn = scaler_nn.transform(X_test)

# Keras expects classes 0,1,2 instead of -1,0,1
y_train_nn = y_train.map({-1: 0, 0: 1, 1: 2})

nn_model = Sequential([
    Dense(64, activation="relu", input_shape=(len(features),)),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

nn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

nn_model.fit(X_train_nn, y_train_nn, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

pred_nn_class = np.argmax(nn_model.predict(X_test_nn), axis=1)
test["NN_Pred"] = pd.Series(pred_nn_class, index=test.index).map({0: -1, 1: 0, 2: 1})

# ============================================================
# EVALUATION
# ============================================================

to_num = {"SELL": -1, "HOLD": 0, "BUY": 1}
test["RSI_Pred"] = test["RSI_Model"].map(to_num)
test["MultiFactor_Pred"] = test["MultiFactor_Model"].map(to_num)

print("RSI Model          :", round(accuracy_score(y_test, test["RSI_Pred"]) * 100, 2), "%")
print("Multi-Factor Model :", round(accuracy_score(y_test, test["MultiFactor_Pred"]) * 100, 2), "%")
print("Linear Regression  :", round(accuracy_score(y_test, test["Linear_Pred"]) * 100, 2), "%")
print("Ridge Regression   :", round(accuracy_score(y_test, test["Ridge_Pred"]) * 100, 2), "%")
print("Neural Network     :", round(accuracy_score(y_test, test["NN_Pred"]) * 100, 2), "%")

# ============================================================
# EXPORT
# ============================================================

to_label = {-1: "SELL", 0: "HOLD", 1: "BUY"}
test["Linear_Signal"] = test["Linear_Pred"].map(to_label)
test["Ridge_Signal"] = test["Ridge_Pred"].map(to_label)
test["NN_Signal"] = test["NN_Pred"].map(to_label)

results = test[[
    "Date", "Commodity", "Price",
    "RSI_Model", "MultiFactor_Model", "Linear_Signal", "Ridge_Signal", "NN_Signal"
]].sort_values(["Commodity", "Date"]).reset_index(drop=True)

results.to_excel("Results_BUY_HOLD_SELL.xlsx", index=False)
