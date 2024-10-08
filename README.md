

---

# DIET RECOMMENDATION AND WORKOUT PLAN

## Overview
This project uses machine learning algorithms and Python programming to create personalized diet and workout plans. The goal is to provide users with recommendations that align with their fitness goals, dietary preferences, and health conditions.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Structure
```
├── data
│   ├── raw_data.csv          # Raw data used for training and evaluation
│   ├── processed_data.csv    # Data after preprocessing
├── notebooks
│   ├── data_preprocessing.ipynb  # Notebook for data cleaning and preprocessing
│   ├── model_training.ipynb      # Notebook for training models
│   ├── evaluation.ipynb          # Notebook for model evaluation
├── src
│   ├── data_preprocessing.py     # Script for data preprocessing
│   ├── model.py                  # Script for model architecture and training
│   ├── evaluation.py             # Script for evaluating the models
├── output
│   ├── model.pkl                 # Trained model file
│   ├── results                   # Folder containing results (graphs, metrics)
├── README.md
└── requirements.txt
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/diet-recommendation-workout-plan.git
   cd diet-recommendation-workout-plan
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**:
   - Run the data preprocessing script to clean and prepare the dataset:
     ```bash
     python src/data_preprocessing.py
     ```

2. **Model Training**:
   - Train the machine learning model using the following command:
     ```bash
     python src/model.py
     ```

3. **Model Evaluation**:
   - Evaluate the model's performance:
     ```bash
     python src/evaluation.py
     ```

4. **Generate Recommendations**:
   - Use the trained model to generate personalized diet and workout plans:
     ```bash
     python src/recommendation.py
     ```

## Dataset

- The dataset used in this project consists of FOOD , Nutrition distribution , workout plan CSV file

## Model Training

- The model is trained using k mean and random forest algorithm 
- Training involves classification and decission tree .

## Evaluation

- The model's performance is evaluated using [- **Accuracy**: Correct BMI predictions  
- **Precision**: Relevant food/workout suggestions  
- **Recall**: Identified appropriate BMI categories  
- **F1-Score**: Balanced recommendation effectiveness  ].
- Results and evaluation metrics are stored in the `output/results` directory.

## Results

- The model evaluation for the 'Diet Recommendation and Workout Plan' project showed accurate BMI predictions and personalized workout and food plans. The results were summarized in tables, with graphs displaying BMI categories and recommended plans based on user data.

## Contributing

Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to contact:
- **Name**: TEJAS A M
- **Email**: tejasam571@gmail.com

---

Feel free to customize the above template according to your specific needs!
