import random as random
import csv as csv
import numpy as np
import pandas as pd


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert class column - flag=True: convert string to int, flag=False: convert int to string
def class_column_convert(dataset, column, flag):
    if (flag):
        lookup = {'yes': 1, 'no': 0}
    else:
        lookup = {'1.0': 'yes', '0.0': 'no'}
    for row in dataset:
        row[column] = lookup[row[column]]


# Split the dataset into training set and testing set
def percentage_split(dataset, percentage):
    cut = int(len(dataset) * percentage)
    random.shuffle(dataset)
    train_set = dataset[:cut]
    test_set = dataset[cut:]
    return train_set, test_set


# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights
def train_weights(train, l_rate, n_pass):
    weights = [0.0 for i in range(len(train[0]))]
    for pas in range(n_pass):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights


# Perceptron Algorithm
def perceptron(train, test, l_rate, n_pass):
    predictions = list()
    weights = train_weights(train, l_rate, n_pass)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return (predictions)


# Calculate accuracy percentage and other stats
def accuracy_metric(actual, prediction):
    correct = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            correct += 1
            if (prediction[i] == 0):
                true_negative += 1
            else:
                true_positive += 1
        else:
            if (prediction[i] == 0):
                false_negative += 1
            else:
                false_positive += 1
    return correct / float(len(actual)) * 100.0, true_positive, false_positive, true_negative, false_negative


# Test the Perceptron algorithm
# Import data
filename = 'risk-train-preprocessed.csv'
dataset = load_csv(filename)

# Remove headers
dataset = np.delete(dataset, 0, 0)

# Test Perceptron algorithm using percentage split
train_set, test_set = percentage_split(dataset, percentage=0.7)

# Remove id column
train_set = np.delete(train_set, 0, 1)
order_id_test_set = test_set[:, 0]
test_set = np.delete(test_set, 0, 1)

# Convert class column to numeric
class_column_convert(train_set, len(train_set[0])-1, True)
class_column_convert(test_set, len(test_set[0])-1, True)

# Convert values to float
train_set = train_set.astype(float)
test_set = test_set.astype(float)

# Run the Perceptron algorithm
prediction = perceptron(train_set, test_set, l_rate=0.01, n_pass=1)
actual = [row[-1] for row in test_set]

# Calculate accuracy and other stats
accuracy, true_positive, false_positive, true_negative, false_negative = accuracy_metric(
    actual, prediction)

print('accuracy: %.4f  true_positive: %d  false_positive: %d  true_negative: %d  false_negative: %d' %
      (accuracy, true_positive, false_positive, true_negative, false_negative))

# Merge id and class columns
result = np.column_stack((order_id_test_set, prediction))

# Convert class column to string
class_column_convert(result, 1, False)

# Export result to csv file
headers = ['ORDER_ID', 'CLASS']
df = pd.DataFrame(result)
df.to_csv('output.csv', index=False, header=headers)
