import pickle
import pandas as pd

# Load the trained model from the .pkl file
with open('random_forest_model.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

data = pd.read_csv("heart_attack_prediction_dataset.csv")
data = pd.get_dummies(data, columns=['Sex', 'Diet'])

def predict_input():
    # Prompt user for input
    Age = int(input("Enter Age: "))
    sex = int(input("Enter sex (0 for female, 1 for male): "))
    HeartRate = int(input("Enter Heart Rate: "))
    Diabetes = int(input("Enter Diabetes: "))
    FamilyHistory = int(input("Enter Family History: "))
    Smoking = int(input("Enter Smoking: "))
    Obesity = int(input("Enter Obesity: "))
    AlcoholConsumption = int(input("Enter Alcohol Consumption: "))
    ExerciseHoursPerWeek= float(input("Enter Exercise Hours Per Week: "))
    diet = int(input("Enter diet (0 for healthy, 1 for unhealthy): "))
    PreviousHeartProblems = int(input("Enter Previous Heart Problems: "))
    MedicationUse = int(input("Enter Medication Use: "))
    SedentaryHoursPerDay = float(input("Enter Sedentary Hours Per Day: "))
    Income = int(input("Enter Income: "))
    PhysicalActivityDaysPerWeek = int(input("Enter Physical Activity Days Per Week: "))
    SleepHoursPerDay = int(input("Enter Sleep Hours Per Day: "))
  

    # Preprocess the input to match the format expected by the model
    # One-hot encode categorical variables
    sex_one_hot = [1, 0] if sex == 0 else [0, 1]
    diet_one_hot = [1, 0] if diet == 0 else [0, 1]
    

    # Combine all features into a single feature vector
    input_features = [Age, HeartRate, Diabetes, FamilyHistory, Smoking, Obesity, AlcoholConsumption, ExerciseHoursPerWeek,
                      PreviousHeartProblems, MedicationUse, SedentaryHoursPerDay, Income, PhysicalActivityDaysPerWeek,
                      SleepHoursPerDay] + sex_one_hot + diet_one_hot 

    # Make prediction using the model
    prediction = rf_classifier.predict([input_features])

    # Return the prediction
    return prediction

# Example usage
prediction_result = predict_input()

# Display prediction result
if prediction_result == 1:
    print("High risk of heart disease")
else:
    print("Low risk of heart disease")

