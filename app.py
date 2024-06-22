import tkinter as tk
from tkinter import messagebox
import pickle
import pandas as pd


#graph plot
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
x_data, y_data = [], []


line, = ax.plot([], [], lw=2)
ax.set_ylim(60, 100)
ax.set_xlim(0, 100)  # Initial x-axis limit
ax.set_ylabel('Heartbeat')
ax.set_xlabel('Time')
ax.set_title('Live Heartbeat Graph')


def update(frame):
    x_data.append(frame)
    y_data.append(random.randint(7, 95))
    
   
    if frame > 100:
        ax.set_xlim(frame - 100, frame)
    
    line.set_data(x_data[-100:], y_data[-100:])
    return line,


ani = FuncAnimation(fig, update, frames=range(1000), blit=True, interval=100)

plt.show()


def generate_and_average():
    
    numbers = []
    
    averages = []
    
    for _ in range(100):
        number = random.randint(65, 95)
        numbers.append(number)
        
        if len(numbers) % 20 == 0:
            avg = sum(numbers[-20:]) / 20
            averages.append(avg)
    return avg




# Load the trained model from the .pkl file
with open('random_forest_model.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Load the dataset used for training
data = pd.read_csv("heart_attack_prediction_dataset.csv")
data = pd.get_dummies(data, columns=['Sex', 'Diet'])

# Get column names after one-hot encoding
columns = data.columns.tolist()

def predict_input():
    try:
        # Prompt user for input
        Age = int(age_entry.get())
        Sex = int(sex_entry.get())
        HeartRate = int(generate_and_average())
        Diabetes = int(diabetes_entry.get())
        FamilyHistory = int(family_history_entry.get())
        Smoking = int(smoking_entry.get())
        Obesity = int(obesity_entry.get())
        AlcoholConsumption = int(alcohol_consumption_entry.get())
        ExerciseHoursPerWeek = float(exercise_hours_per_week_entry.get())
        Diet = int(diet_entry.get())
        PreviousHeartProblems = int(previous_heart_problems_entry.get())
        MedicationUse = int(medication_use_entry.get())
        SedentaryHoursPerDay = float(sedentary_hours_per_day_entry.get())
        Income = int(income_entry.get())
        PhysicalActivityDaysPerWeek = int(physical_activity_days_per_week_entry.get())
        SleepHoursPerDay = int(sleep_hours_per_day_entry.get())

      
        sex_one_hot = [1, 0] if Sex == 0 else [0, 1]
        diet_one_hot = [1, 0] if Diet == 0 else [0, 1]

        # Combine all features into a single feature vector
        input_features = [Age, HeartRate, Diabetes, FamilyHistory, Smoking, Obesity, AlcoholConsumption,
                          ExerciseHoursPerWeek, PreviousHeartProblems, MedicationUse, SedentaryHoursPerDay,
                          Income, PhysicalActivityDaysPerWeek, SleepHoursPerDay] + sex_one_hot + diet_one_hot

        # Check if input data columns match the columns used during training
        # if len(input_features) != len(columns):
        #     raise ValueError("Input features do not match the columns used for training")

        # Make prediction using the model
        prediction = rf_classifier.predict([input_features])

        # Display prediction result using a message box
        if prediction == 1:
            messagebox.showinfo("Prediction Result", "High risk of heart disease")
        else:
            messagebox.showinfo("Prediction Result", "Low risk of heart disease")
    except ValueError as e:
        messagebox.showerror("Error", str(e))


# except ValueError:
#             messagebox.showerror("Error", "Please enter valid numerical values")

# Create GUI window
window = tk.Tk()
window.title("Heart Attack Risk Prediction")



# Labels and Entry fields for user inputs
tk.Label(window, text="Age:").grid(row=0, column=0)
age_entry = tk.Entry(window)
age_entry.grid(row=0, column=1)

tk.Label(window, text="Sex (0 for female, 1 for male):").grid(row=1, column=0)
sex_entry = tk.Entry(window)
sex_entry.grid(row=1, column=1)

# tk.Label(window, text="Heart Rate:").grid(row=2, column=0)
# heart_rate_entry = tk.Entry(window)
# heart_rate_entry.grid(row=2, column=1)

tk.Label(window, text="Diabetes:").grid(row=3, column=0)
diabetes_entry = tk.Entry(window)
diabetes_entry.grid(row=3, column=1)

tk.Label(window, text="Family History:").grid(row=4, column=0)
family_history_entry = tk.Entry(window)
family_history_entry.grid(row=4, column=1)

tk.Label(window, text="Smoking:").grid(row=5, column=0)
smoking_entry = tk.Entry(window)
smoking_entry.grid(row=5, column=1)

tk.Label(window, text="Obesity:").grid(row=6, column=0)
obesity_entry = tk.Entry(window)
obesity_entry.grid(row=6, column=1)

tk.Label(window, text="Alcohol Consumption:").grid(row=7, column=0)
alcohol_consumption_entry = tk.Entry(window)
alcohol_consumption_entry.grid(row=7, column=1)

tk.Label(window, text="Exercise Hours Per Week:").grid(row=8, column=0)
exercise_hours_per_week_entry = tk.Entry(window)
exercise_hours_per_week_entry.grid(row=8, column=1)

tk.Label(window, text="Diet (0 for healthy, 1 for unhealthy):").grid(row=9, column=0)
diet_entry = tk.Entry(window)
diet_entry.grid(row=9, column=1)

tk.Label(window, text="Previous Heart Problems:").grid(row=10, column=0)
previous_heart_problems_entry = tk.Entry(window)
previous_heart_problems_entry.grid(row=10, column=1)

tk.Label(window, text="Medication Use:").grid(row=11, column=0)
medication_use_entry = tk.Entry(window)
medication_use_entry.grid(row=11, column=1)

tk.Label(window, text="Sedentary Hours Per Day:").grid(row=12, column=0)
sedentary_hours_per_day_entry = tk.Entry(window)
sedentary_hours_per_day_entry.grid(row=12, column=1)

tk.Label(window, text="Income:").grid(row=13, column=0)
income_entry = tk.Entry(window)
income_entry.grid(row=13, column=1)

tk.Label(window, text="Physical Activity Days Per Week:").grid(row=14, column=0)
physical_activity_days_per_week_entry = tk.Entry(window)
physical_activity_days_per_week_entry.grid(row=14, column=1)

tk.Label(window, text="Sleep Hours Per Day:").grid(row=15, column=0)
sleep_hours_per_day_entry = tk.Entry(window)
sleep_hours_per_day_entry.grid(row=15, column=1)

# Button to trigger prediction
predict_button = tk.Button(window, text="Predict", command=predict_input)
predict_button.grid(row=16, columnspan=2)



# Run the GUI event loop
window.mainloop()