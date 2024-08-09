import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import ttk, messagebox

# Load datasets
nutrition_df = pd.read_csv('nutrition_distribution_large.csv')
food_df = pd.read_csv('food_large.csv')

# Enum for user goals
class Goal:
    LOSE_WEIGHT = 1
    GAIN_WEIGHT = 2
    STAY_HEALTHY = 3

# Define features and labels
USER_FEATURES = ['Age', 'Gender', 'Weight', 'Height', 'Diseases', 'ActivityLevel', 'Goal']

# Preprocess data
def preprocess_data(df):
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df['Diseases'] = LabelEncoder().fit_transform(df['Diseases'])
    df['Goal'] = LabelEncoder().fit_transform(df['Goal'])
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    return df

# Apply preprocessing to nutrition_df
nutrition_df = preprocess_data(nutrition_df)

# Create clustering pipeline
def create_clustering_pipeline():
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', imputer), ('scaler', scaler)]), USER_FEATURES)
        ]
    )
    clustering_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clustering', KMeans(n_clusters=5, random_state=42))
    ])
    return clustering_pipeline

# Fit the clustering pipeline
clustering_pipeline = create_clustering_pipeline()
clustering_pipeline.fit(nutrition_df[USER_FEATURES])

# Create a Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(nutrition_df[USER_FEATURES], nutrition_df['BMI'])

# Recommend meal based on user input
def recommend_meal(user_input):
    user_df = pd.DataFrame([user_input])
    user_df = preprocess_data(user_df)
    user_cluster = clustering_pipeline.predict(user_df[USER_FEATURES])[0]
    bmi_pred = rf_reg.predict(user_df[USER_FEATURES])[0]

    # Filter food items based on user's goal
    if user_input['Goal'] == Goal.LOSE_WEIGHT:
        food_items = food_df[food_df['Calories'] <= 400]
    elif user_input['Goal'] == Goal.GAIN_WEIGHT:
        food_items = food_df[food_df['Calories'] >= 500]
    else:  # Stay Healthy
        food_items = food_df[(food_df['Calories'] > 400) & (food_df['Calories'] < 500)]

    # Recommend a 3-course meal
    breakfast = food_items[food_items['MealType'] == 'Breakfast'].sample(1)
    lunch = food_items[food_items['MealType'] == 'Lunch'].sample(1)
    dinner = food_items[food_items['MealType'] == 'Dinner'].sample(1)

    return {
        'Breakfast': {
            'FoodItem': breakfast['FoodItem'].values[0],
            'Calories': breakfast['Calories'].values[0],
            'Nutrients': breakfast['Nutrients'].values[0]
        },
        'Lunch': {
            'FoodItem': lunch['FoodItem'].values[0],
            'Calories': lunch['Calories'].values[0],
            'Nutrients': lunch['Nutrients'].values[0]
        },
        'Dinner': {
            'FoodItem': dinner['FoodItem'].values[0],
            'Calories': dinner['Calories'].values[0],
            'Nutrients': dinner['Nutrients'].values[0]
        }
    }

# GUI for user input
def submit_form():
    try:
        user_input = {
            'Age': int(age_entry.get()),
            'Gender': gender_var.get(),
            'Weight': float(weight_entry.get()),
            'Height': float(height_entry.get()),
            'Diseases': diseases_entry.get(),
            'ActivityLevel': int(activity_level_entry.get()),
            'Goal': goal_var.get()
        }
        recommended_meal = recommend_meal(user_input)
        result_text.set(f"Recommended Breakfast: {recommended_meal['Breakfast']['FoodItem']}\n"
                        f"Calories: {recommended_meal['Breakfast']['Calories']}\n"
                        f"Nutrients: {recommended_meal['Breakfast']['Nutrients']}\n\n"
                        f"Recommended Lunch: {recommended_meal['Lunch']['FoodItem']}\n"
                        f"Calories: {recommended_meal['Lunch']['Calories']}\n"
                        f"Nutrients: {recommended_meal['Lunch']['Nutrients']}\n\n"
                        f"Recommended Dinner: {recommended_meal['Dinner']['FoodItem']}\n"
                        f"Calories: {recommended_meal['Dinner']['Calories']}\n"
                        f"Nutrients: {recommended_meal['Dinner']['Nutrients']}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("Diet Recommendation System")
root.geometry("500x600")
root.resizable(False, False)

style = ttk.Style(root)
style.configure('TLabel', font=('Arial', 12))
style.configure('TButton', font=('Arial', 12))

frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Age:").grid(row=0, column=0, sticky=tk.W, pady=5)
age_entry = ttk.Entry(frame)
age_entry.grid(row=0, column=1, pady=5)

ttk.Label(frame, text="Gender:").grid(row=1, column=0, sticky=tk.W, pady=5)
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(frame, textvariable=gender_var, values=['Male', 'Female', 'Other'])
gender_combobox.grid(row=1, column=1, pady=5)

ttk.Label(frame, text="Weight (kg):").grid(row=2, column=0, sticky=tk.W, pady=5)
weight_entry = ttk.Entry(frame)
weight_entry.grid(row=2, column=1, pady=5)

ttk.Label(frame, text="Height (cm):").grid(row=3, column=0, sticky=tk.W, pady=5)
height_entry = ttk.Entry(frame)
height_entry.grid(row=3, column=1, pady=5)

ttk.Label(frame, text="Diseases:").grid(row=4, column=0, sticky=tk.W, pady=5)
diseases_entry = ttk.Entry(frame)
diseases_entry.grid(row=4, column=1, pady=5)

ttk.Label(frame, text="Activity Level (1-10):").grid(row=5, column=0, sticky=tk.W, pady=5)
activity_level_entry = ttk.Entry(frame)
activity_level_entry.grid(row=5, column=1, pady=5)

ttk.Label(frame, text="Goal:").grid(row=6, column=0, sticky=tk.W, pady=5)
goal_var = tk.IntVar()
ttk.Radiobutton(frame, text="Lose Weight", variable=goal_var, value=Goal.LOSE_WEIGHT).grid(row=6, column=1, sticky=tk.W)
ttk.Radiobutton(frame, text="Gain Weight", variable=goal_var, value=Goal.GAIN_WEIGHT).grid(row=6, column=2, sticky=tk.W)
ttk.Radiobutton(frame, text="Stay Healthy", variable=goal_var, value=Goal.STAY_HEALTHY).grid(row=6, column=3, sticky=tk.W)

submit_btn = ttk.Button(frame, text="Submit", command=submit_form)
submit_btn.grid(row=7, column=1, pady=20)

result_text = tk.StringVar()
result_label = ttk.Label(frame, textvariable=result_text, justify=tk.LEFT, wraplength=400)
result_label.grid(row=8, columnspan=4, pady=10)

root.mainloop()