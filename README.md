Naive Bayes Spam Filter

This project implements a Naive Bayes classifier to detect spam emails. The classifier is trained on a dataset of spam and ham (non-spam) emails and then evaluated on a test set. The performance of the classifier is measured using accuracy, precision, recall, and F1 score.
Prerequisites

Ensure you have the following packages installed before running the script:

    os
    re
    matplotlib
    nltk
    collections

You can install the necessary packages using pip:

bash

pip install matplotlib nltk

Setup

    Download and extract the Ling-Spam dataset, or use your own dataset. Ensure the dataset is organized with spam and ham emails in separate folders and a test folder for evaluation.

    Place the dataset in a directory named lingspam_public in the root of the project.

Running the Script

    Ensure you have the required NLTK data files. The script will automatically download them if they are not already installed:

python

nltk.download('stopwords')
nltk.download('punkt')

    Execute the script:

bash

python main.py

Script Overview
Functions

    load_dataset(root_dir):
        Loads the emails from the specified root directory.
        Categorizes emails into spam, ham, and test datasets.

    tokenizeAndFilter(textList):
        Tokenizes the text and filters out stopwords and non-alphabetic words.
        Returns a dictionary with words as keys and their counts as values.

    trainNaiveBayes(spamList, hamList):
        Trains the Naive Bayes classifier.
        Returns word count dictionaries for spam and ham emails and the prior probabilities.

    evaluateModel(testList, fileList, spamWordCount, hamWordCount, priorSpamProb, priorHamProb, totalSpam, totalHam):
        Evaluates the classifier on the test set.
        Returns counts of true positives, false positives, true negatives, and false negatives.

    calculateMetrics(tp, fp, tn, fn):
        Calculates accuracy, precision, recall, and F1 score from the evaluation results.

    runNaiveBayes(root_dir):
        Orchestrates the process from loading the dataset to calculating metrics.
        Prints and returns the performance metrics.

    plotMetrics(metrics):
        Plots the performance metrics using matplotlib.

Main Execution

    The script loads the dataset from lingspam_public.
    Trains the Naive Bayes classifier.
    Evaluates the classifier on the test set.
    Prints and plots the performance metrics.

Example Output

yaml

Accuracy: 0.9500
Precision: 0.9600
Recall: 0.9400
F1 Score: 0.9500

License

This project is licensed under the MIT License. See the LICENSE file for details.