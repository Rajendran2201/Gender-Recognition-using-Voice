## Gender Recognition Using Voice

This project aims to recognize gender based on voice characteristics. It involves splitting the dataset, applying label mapping, scaling features, splitting data into training and testing sets, training a support vector machine (SVM) model, and evaluating its performance.

### Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn (train_test_split, accuracy_score)
- keras

### Loading the Dataset
The dataset is loaded into a pandas DataFrame from a CSV file. It contains 3168 rows and 21 columns.

### Data Preprocessing
- Missing values: The dataset contains no missing values.
- Data types: Features are of type float64, and the label is of type object.
- Summary statistics: Statistical information about the dataset is provided, including mean, standard deviation, minimum, maximum, and quartile values.
- Label encoding: Labels are replaced with numerical values (0 for female, 1 for male).

### Data Visualization
- Correlation: A heatmap is constructed to visualize the correlation between features. This helps understand the relationships between different voice characteristics.

### Model Training and Evaluation
- Features and labels are split.
- Label mapping is applied to convert labels to binary values.
- Features are scaled using Min-Max scaling.
- Data is split into training and testing sets.
- An SVM model with a linear kernel is trained on the training data.
- The model's performance is evaluated using the testing data, and accuracy score is calculated.

### Testing with New Data
The trained model is tested with new data to predict gender based on voice characteristics. Predictions are made using the SVM model, and the predicted gender is mapped back to 'male' or 'female' for interpretation.

### Usage
1. Import the necessary libraries.
2. Load the dataset into a pandas DataFrame.
3. Preprocess the data (check for missing values, data types, etc.).
4. Visualize the data to understand its characteristics.
5. Split the dataset into features and labels.
6. Apply label mapping and scale the features.
7. Split the data into training and testing sets.
8. Train an SVM model on the training data.
9. Evaluate the model's performance using the testing data.
10. Test the model with new data to predict gender based on voice characteristics.

### License
This project is licensed under the MIT [License](https://github.com/Rajendran2201/Gender-Recognition-using-Voice/blob/main/LICENSE)

### Contribution
Contributions are welcome! Please feel free to submit any enhancements or bug fixes.

### Support
For any questions or concerns, please open an issue on this repository.
