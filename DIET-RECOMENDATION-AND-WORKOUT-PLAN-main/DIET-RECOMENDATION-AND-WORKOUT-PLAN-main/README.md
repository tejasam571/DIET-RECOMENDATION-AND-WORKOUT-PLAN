# DIET RECOMMENDATION AND WORKOUT PLAN

## Overview
This project uses machine learning algorithms and Python programming to create personalized diet and workout plans. The goal is to provide users with recommendations that align with their fitness goals, dietary preferences, and health conditions.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure
├── data
│ ├── raw_data.csv # Raw data used for training and evaluation
│ ├── processed_data.csv # Data after preprocessing
├── notebooks
│ ├── data_preprocessing.ipynb # Notebook for data cleaning and preprocessing
│ ├── model_training.ipynb # Notebook for training models
│ ├── evaluation.ipynb # Notebook for model evaluation
├── src
│ ├── data_preprocessing.py # Script for data preprocessing
│ ├── model.py # Script for model architecture and training
│ ├── evaluation.py # Script for evaluating the models
├── output
│ ├── model.pkl # Trained model file
│ ├── results # Folder containing results (graphs, metrics)
├── app3.py # Main Streamlit app
├── requirements.txt
├── Dockerfile
└── README.md

bash
Copy code

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/diet-recommendation-workout-plan.git
   cd diet-recommendation-workout-plan
Set up a virtual environment (optional but recommended):

bash
Copy code
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Run the Streamlit app locally:

bash
Copy code
streamlit run app3.py
Open your browser and go to:

arduino
Copy code
http://localhost:8501
Docker Deployment
You can also run this project inside a Docker container.

Build the Docker Image
bash
Copy code
docker build -t diet-app .
Run the Docker Container
bash
Copy code
docker run -p 8501:8501 diet-app
Now open your browser and visit:

arduino
Copy code
http://localhost:8501
(or use your machine’s IP address if running on a server).

Example: Running on a server
If deployed on a server with IP 192.168.29.41, you can access it at:

cpp
Copy code
http://192.168.29.41:8501
Dataset
The dataset used in this project consists of food, nutrition distribution, and workout plan CSV files.

Model Training
The model is trained using K-Means and Random Forest algorithms.

Training involves classification and decision tree methods.

Evaluation
The model's performance is evaluated using:

Accuracy: Correct BMI predictions

Precision: Relevant food/workout suggestions

Recall: Identified appropriate BMI categories

F1-Score: Balanced recommendation effectiveness

Results and evaluation metrics are stored in the output/results directory.

Results
The model evaluation for the 'Diet Recommendation and Workout Plan' project showed accurate BMI predictions and personalized workout and food plans. The results were summarized in tables, with graphs displaying BMI categories and recommended plans based on user data.

Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository.

Create a new branch (git checkout -b feature-branch).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature-branch).

Open a Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions or suggestions, feel free to contact:

Name: TEJAS A M

Email: tejasam571@gmail.com
