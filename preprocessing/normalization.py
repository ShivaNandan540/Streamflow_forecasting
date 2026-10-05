from sklearn.preprocessing import MinMaxScaler

def normalize_train_validation(X_train, X_val, y_train, y_val):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)

    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    return X_train_s, X_val_s, y_train_s, y_val_s, x_scaler, y_scaler
