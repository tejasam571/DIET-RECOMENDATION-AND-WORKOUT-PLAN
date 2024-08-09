import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st

# CSS to inject a background image
page_bg_img = '''
<style>
body {
background-image: url("https://path/to/your/background.jpg");
background-size: cover;
}
</style>
'''

# Load datasets with error handling
def load_dataset(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        return None

nutrition_df = load_dataset('nutrition_distribution_large.csv')
food_df = load_dataset('food_large.csv')
workout_df = load_dataset('expanded_workout_plan.csv')  # Updated file path

if nutrition_df is None or food_df is None or workout_df is None:
    st.stop()

# Inject CSS with background image
#st.markdown(page_bg_img, unsafe_allow_html=True)

# Inspect workout_df columns
#st.write("Workout DataFrame columns:", workout_df.columns)

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

# Recommend meal and workout based on user input
def recommend_meal_and_workout(user_input):
    user_df = pd.DataFrame([user_input])
    user_df = preprocess_data(user_df)
    user_cluster = clustering_pipeline.predict(user_df[USER_FEATURES])[0]
    bmi_pred = rf_reg.predict(user_df[USER_FEATURES])[0]

    # Filter food items based on user's goal
    if user_input['Goal'] == Goal.LOSE_WEIGHT:
        food_items = food_df[food_df['Calories'] <= 400]
        workout_items = workout_df[workout_df['Type'] == 'Lose Weight']
    elif user_input['Goal'] == Goal.GAIN_WEIGHT:
        food_items = food_df[food_df['Calories'] >= 500]
        workout_items = workout_df[workout_df['Type'] == 'Gain Weight']
    else:  # Stay Healthy
        food_items = food_df[(food_df['Calories'] > 400) & (food_df['Calories'] < 500)]
        workout_items = workout_df[workout_df['Type'] == 'Stay Healthy']

    # Function to safely sample food items
    def safe_sample(df, meal_type):
        filtered = df[df['MealType'] == meal_type]
        if not filtered.empty:
            return filtered.sample(1)
        else:
            return pd.DataFrame({'FoodItem': ['No recommendation'], 'Calories': ['N/A'], 'Nutrients': ['N/A']})

    # Recommend a 3-course meal
    breakfast = safe_sample(food_df[food_df['MealType'] == 'Breakfast'], 'Breakfast')
    lunch = safe_sample(food_items, 'Lunch')
    dinner = safe_sample(food_items, 'Dinner')

    # Recommend a workout plan
    workout_plan = workout_items.sample(1) if not workout_items.empty else pd.DataFrame({'Exercise': ['No recommendation'], 'Timing': ['N/A']})

    return {
        'BMI': bmi_pred,
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
        },
        'Workout': {
            'Exercise': workout_plan['Exercise'].values[0],
            'Timing': workout_plan['Timing'].values[0]
        }
    }

# Streamlit user input
st.title("Diet and Workout Recommendation System")

age = st.number_input("Age", min_value=0, step=1)
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
diseases = st.text_input("Diseases")
activity_level = st.number_input("Activity Level (1-10)", min_value=1, max_value=10, step=1)
goal = st.radio("Goal", [("Lose Weight", Goal.LOSE_WEIGHT), ("Gain Weight", Goal.GAIN_WEIGHT), ("Stay Healthy", Goal.STAY_HEALTHY)], format_func=lambda x: x[0])

if st.button("Submit"):
    user_input = {
        'Age': age,
        'Gender': gender,
        'Weight': weight,
        'Height': height,
        'Diseases': diseases,
        'ActivityLevel': activity_level,
        'Goal': goal[1]
    }
    recommended_plan = recommend_meal_and_workout(user_input)
    
    st.write(f"Calculated BMI: {recommended_plan['BMI']:.2f}")
    st.write("## Recommended Meal Plan")
    st.write(f"**Breakfast**: {recommended_plan['Breakfast']['FoodItem']} - {recommended_plan['Breakfast']['Calories']} Calories\nNutrients: {recommended_plan['Breakfast']['Nutrients']}")
    st.write(f"**Lunch**: {recommended_plan['Lunch']['FoodItem']} - {recommended_plan['Lunch']['Calories']} Calories\nNutrients: {recommended_plan['Lunch']['Nutrients']}")
    st.write(f"**Dinner**: {recommended_plan['Dinner']['FoodItem']} - {recommended_plan['Dinner']['Calories']} Calories\nNutrients: {recommended_plan['Dinner']['Nutrients']}")
    st.write("## Recommended Workout Plan")
    st.write(f"**Exercise**: {recommended_plan['Workout']['Exercise']}\nTiming: {recommended_plan['Workout']['Timing']} minutes")
