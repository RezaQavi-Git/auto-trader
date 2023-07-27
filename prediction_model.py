# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from data_processing import data_processing

def prediction_model():

    print("Prediction model start ...")
    stock_data = data_processing()
    target = 'close'
    X = stock_data.drop(columns=[target, 'ds'])
    y = stock_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and train the Random Forest Regressor model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)


    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)


    svm_model = SVR(kernel='linear')  # You can also try 'rbf' or 'poly' kernels
    svm_model.fit(X_train, y_train)




# Assuming you have trained Random Forest, Gradient Boosting, and SVM models
# and have the test data and true target values (y_test) available.

    # Predict on the test data using the Random Forest model
    y_pred_rf = rf_model.predict(X_test)

    # Predict on the test data using the Gradient Boosting model
    y_pred_gb = gb_model.predict(X_test)

    # Predict on the test data using the SVM model
    y_pred_svm = svm_model.predict(X_test)

    # Calculate evaluation metrics for each model
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    mse_svm = mean_squared_error(y_test, y_pred_svm)
    rmse_svm = mean_squared_error(y_test, y_pred_svm, squared=False)
    mae_svm = mean_absolute_error(y_test, y_pred_svm)
    r2_svm = r2_score(y_test, y_pred_svm)

    # Print the evaluation results for each model
    print("Random Forest:")
    print("Mean Squared Error (MSE):", mse_rf)
    print("Root Mean Squared Error (RMSE):", rmse_rf)
    print("Mean Absolute Error (MAE):", mae_rf)
    print("R-squared (R2) Score:", r2_rf)
    print()

    print("Gradient Boosting:")
    print("Mean Squared Error (MSE):", mse_gb)
    print("Root Mean Squared Error (RMSE):", rmse_gb)
    print("Mean Absolute Error (MAE):", mae_gb)
    print("R-squared (R2) Score:", r2_gb)
    print()

    print("Support Vector Machine (SVM):")
    print("Mean Squared Error (MSE):", mse_svm)
    print("Root Mean Squared Error (RMSE):", rmse_svm)
    print("Mean Absolute Error (MAE):", mae_svm)
    print("R-squared (R2) Score:", r2_svm)






    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print("Mean Squared Error (MSE):", mse)
    # print("Root Mean Squared Error (RMSE):", rmse)
    # print("Mean Absolute Error (MAE):", mae)
    # print("R-squared (R2) Score:", r2)

    # Perform cross-validation on the model
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
    # Calculate the mean and standard deviation of cross-validation scores
    cv_rmse_scores = (-cv_scores) ** 0.5
    mean_cv_rmse = cv_rmse_scores.mean()
    std_cv_rmse = cv_rmse_scores.std()

    # print("Cross-Validation RMSE Mean:", mean_cv_rmse)
    # print("Cross-Validation RMSE Std:", std_cv_rmse)

    print("Prediction model end ...")

    return rf_model, stock_data

# # Import necessary libraries for machine learning
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error


# # Define the features and target variable
# features = ['close', 'volume', 'MA_50', 'MA_200']
# target = 'close'

# # Split data into features (X) and target variable (y)
# X = stock_data[features]
# y = stock_data[target]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the prediction model
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = mse**0.5
# print(f"Root Mean Squared Error (RMSE): {rmse}")
