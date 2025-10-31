# 1. Import necessary libraries
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 2. Define file paths (Update these if your file paths are different in VS Code environment)
employment_filepath = '/content/drive/MyDrive/t/LFEAICTTZAA647S.csv'
ai_publications_filepath = '/content/drive/MyDrive/t/data.csv'
model_filename = 'linear_regression_employment_forecast_model.joblib' # Path where model will be saved/loaded from

# 3. Load the datasets
try:
    df_new = pd.read_csv(employment_filepath)
    print("Employment dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: Employment file not found at {employment_filepath}")
    df_new = None # Handle case where file is not found
except Exception as e:
    print(f"An error occurred while loading the employment dataset: {e}")

try:
    df = pd.read_csv(ai_publications_filepath)
    print("AI Publications dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: AI Publications file not found at {ai_publications_filepath}")
    df = None # Handle case where file is not found
except Exception as e:
    print(f"An error occurred while loading the AI publications dataset: {e}")

# Proceed only if both dataframes were loaded successfully
if df_new is not None and df is not None:

    # 4. Data Cleaning and Preparation
    # Convert observation_date in df_new to datetime
    df_new['observation_date'] = pd.to_datetime(df_new['observation_date'])

    # Extract year from observation_date for merging
    df_new['year'] = df_new['observation_date'].dt.year

    # Aggregate df by year, summing publications (for global AI trend)
    df_aggregated = df.groupby('year')['publications'].sum().reset_index()
    df_aggregated.rename(columns={'publications': 'total_global_publications'}, inplace=True)

    # 5. Data Integration
    # Merge the two dataframes on the 'year' column
    merged_df = pd.merge(df_new, df_aggregated, on='year', how='inner')

    # 6. Feature Engineering
    # Create a lagged feature for South African Employment Levels
    merged_df['LFEAICTTZAA647S_lag1'] = merged_df['LFEAICTTZAA647S'].shift(1)

    # Fill NaN values in the lagged column with 0
    merged_df['LFEAICTTZAA647S_lag1'] = merged_df['LFEAICTTZAA647S_lag1'].fillna(0)

    # 7. Prepare data for Model Training
    features = ['year', 'total_global_publications', 'LFEAICTTZAA647S_lag1']
    target = 'LFEAICTTZAA647S'

    # Ensure features exist before selecting
    available_features = [f for f in features if f in merged_df.columns]
    if not available_features:
        print("Error: None of the specified features are available in the merged DataFrame.")
    else:
        X = merged_df[available_features]
        y = merged_df[target]

        # 8. Model Selection and Training
        model = LinearRegression()
        model.fit(X, y)
        print("\nLinear Regression model trained successfully.")

        # 9. Save the trained model
        try:
            joblib.dump(model, model_filename)
            print(f"Model saved successfully to {model_filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

        # 10. Load the saved model (Example for dashboard use)
        loaded_model = None
        if os.path.exists(model_filename):
            try:
                loaded_model = joblib.load(model_filename)
                print(f"Model loaded successfully from {model_filename}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found at {model_filename}. Cannot load model.")


        # 11. Forecasting Future Employment Levels (Example)
        if loaded_model is not None:
            print("\nForecasting future employment levels (2025-2029):")
            last_employment = merged_df['LFEAICTTZAA647S'].iloc[-1]
            last_publications = merged_df['total_global_publications'].iloc[-1] # Using last known publications

            future_predictions = []
            # Ensure the feature list for prediction matches the training features
            prediction_features = available_features

            for year in range(2025, 2030):
                future_data = {'year': [year],
                               'total_global_publications': [last_publications],
                               'LFEAICTTZAA647S_lag1': [last_employment]}

                # Create DataFrame with only the required features for prediction
                future_features_df = pd.DataFrame(future_data)[prediction_features]

                predicted_employment = loaded_model.predict(future_features_df)[0]
                future_predictions.append({'year': year, 'predicted_employment': predicted_employment})

                # Update last_employment for the next iteration's lagged feature
                last_employment = predicted_employment

            future_forecast_df = pd.DataFrame(future_predictions)
            print("Future Forecast:")
            display(future_forecast_df)

            # 12. Example Visualization of Historical vs. Forecasted Employment
            print("\nGenerating Historical vs. Forecast Plot:")
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='year', y='LFEAICTTZAA647S', data=merged_df, marker='o', label='Historical South African Employment Levels')
            sns.lineplot(x='year', y='predicted_employment', data=future_forecast_df, marker='o', linestyle='--', color='green', label='Forecasted Employment Levels')
            plt.xlabel('Year')
            plt.ylabel('Employment Levels')
            plt.title('South African Employment Levels: Historical vs. Forecast')
            plt.grid(True)
            plt.legend()
            plt.show()

        else:
            print("Model was not loaded, cannot perform forecasting or visualization.")

else:
    print("\nSkipping data processing and modeling steps due to file loading errors.")
