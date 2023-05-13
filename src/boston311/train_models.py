from .data_clean import clean_and_split_for_linear, clean_and_split_for_logistic
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd

def train_logistic_model(data, scenario) :

    logistic_X, logistic_y = clean_and_split_for_logistic(data, scenario)

    #Train a logistic regression model

    start_time = datetime.now()
    print("Starting Training at {}".format(start_time))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(logistic_X, logistic_y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build model
    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=(X_train.shape[1],), activation='sigmoid')
    ])

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    # Train model with early stopping
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test accuracy:', test_acc)

    end_time = datetime.now()
    total_time = (end_time - start_time)
    print("Ending Training at {}".format(end_time))
    print("Training took {}".format(total_time))

    return model

def train_linear_model(data, scenario) :

    linear_X, linear_y = clean_and_split_for_linear(data, scenario)

    #Train a linear regression model

    start_time = datetime.now()
    print("Starting Training at {}".format(start_time))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(linear_X) # scale the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, linear_y, test_size=0.2, random_state=42)

    # split the data again to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # define the model architecture
    model = keras.Sequential([
        keras.layers.Dense(units=1, input_dim=X_train.shape[1])
    ])

    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train the model
    # we are adding early stopping based on the validation loss
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stop])

    end_time = datetime.now()
    total_time = (end_time - start_time)
    print("Ending Training at {}".format(end_time))
    print("Training took {}".format(total_time))

    return model