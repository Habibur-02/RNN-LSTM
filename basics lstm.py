import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
df=pd.read_csv('/content/Dengue_Climate_Bangladesh - DengueAndClimateBangladesh.csv')

df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
df = df.set_index('Date')

features = ['MIN', 'MAX', 'HUMIDITY', 'RAINFALL']
target = 'DENGUE'

df_features = df[features]
df_target = df[[target]]


scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(df_features)
scaled_target = scaler_target.fit_transform(df_target)


def create_sequences(features, target, time_steps=6):
    """
    এই ফাংশনটি টাইম-সিরিজ ডেটাকে LSTM-এর উপযোগী সিকোয়েন্সে রূপান্তর করে।
    time_steps = কত মাসের ডেটা দেখে পরবর্তী মাসের প্রেডিক্ট করবে।
    """
    X, y = [], []
    for i in range(len(features) - time_steps):
        
        X.append(features[i:(i + time_steps), :])
        # টার্গেট (আউটপুট): পরবর্তী মাসের ডেঙ্গু কেস
        y.append(target[i + time_steps, 0])
    return np.array(X), np.array(y)

TIME_STEPS = 6
X, y = create_sequences(scaled_features, scaled_target, TIME_STEPS)


split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Total samples: {len(X)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

n_features = len(features) # ফিচার সংখ্যা = ৪
input_shape = (TIME_STEPS, n_features) # (6, 4)


def build_stacked_lstm(input_shape):
    model = Sequential(name="Stacked_LSTM")
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1)) # আউটপুট লেয়ার
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_bidirectional_lstm(input_shape):
    model = Sequential(name="Bidirectional_LSTM")
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_gru(input_shape):
    model = Sequential(name="GRU")
    model.add(GRU(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

models = {
    "Stacked_LSTM": build_stacked_lstm(input_shape),
    "Bidirectional_LSTM": build_bidirectional_lstm(input_shape),
    "GRU": build_gru(input_shape)
}

models["Stacked_LSTM"].summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_logs = {} # ট্রেনিং লগ সেভ করার জন্য

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1, # ট্রেনিং ডেটা থেকেই কিছু অংশ ভ্যালিডেশনের জন্য
        callbacks=[early_stopping],
        verbose=1 # ট্রেনিং প্রগ্রেস দেখাবে
    )
    history_logs[model_name] = history


results_rmse = {} # RMSE সেভ করার জন্য
plt.figure(figsize=(15, 8))

y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1))

for model_name, model in models.items():
    y_pred_scaled = model.predict(X_test)

    y_pred = scaler_target.inverse_transform(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    results_rmse[model_name] = rmse
    print(f"RMSE for {model_name}: {rmse:.2f}")

    plt.plot(y_pred, label=f'Predicted - {model_name} (RMSE: {rmse:.2f})')

plt.plot(y_test_real, label=f'Actual Dengue Cases (Test Data)', color='black', linestyle='--', linewidth=2)

plt.title('Dengue Case Prediction Comparison (LSTM vs Bi-LSTM vs GRU)')
plt.xlabel('Time (Months) - Test Set')
plt.ylabel('Number of Dengue Cases')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Final Model Comparison (Lower is Better) ---")
for model_name, rmse in results_rmse.items():
    print(f"{model_name} RMSE: {rmse:.2f}")
