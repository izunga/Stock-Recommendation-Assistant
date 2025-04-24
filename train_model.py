import numpy as np
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.models import load_model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler, epochs=10, batch_size=32):
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Rescale predictions and actual values to the original scale
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    print(f"RMSE: {rmse}")
    
    return predictions_rescaled, y_test_rescaled
