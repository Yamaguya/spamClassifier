import os
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Loads the dataset and returns the spam, ham, test, and file lists
def load_dataset(root_dir):
    spamList = [] # List that contains spam files from parts 1-9
    hamList  = [] # List that contains ham files from parts 1-9
    testList = [] # List that contains files from part 10
    fileList = [] # List that contains filenames from part 10

    for directories, subdirs, files in os.walk(root_dir):
            for file in files:
                with open(os.path.join(directories, file), encoding='latin-1') as f:
                    text = f.read()
                    if 'part10' in directories:
                        testList.append(text)
                        fileList.append(file)
                    elif re.search(r'spmsg', file):
                        spamList.append(text)
                    else:
                        hamList.append(text)
    return spamList, hamList, testList, fileList

# Returns dictionary with word as key and count as value
def tokenizeAndFilter(textList):
    stopWords = set(stopwords.words('english'))
    wordCount = defaultdict(int)
    
    for text in textList:
        words = word_tokenize(text)
        for word in words:
            if word.isalpha() and word not in stopWords:
                wordCount[word.lower()] += 1
                
    return wordCount

# Trains the Naive Bayes classifier, returns the spam and 
# ham word count dictionaries and the prior probabilities
def trainNaiveBayes(spamList, hamList):
    spamWordCount = tokenizeAndFilter(spamList)
    hamWordCount  = tokenizeAndFilter(hamList)
    
    totalSpam = len(spamList)
    totalHam  = len(hamList)
    
    priorSpamProb = totalSpam / (totalSpam + totalHam) 
    priorHamProb  = totalHam  / (totalSpam + totalHam)

    return spamWordCount, hamWordCount, priorSpamProb, priorHamProb

# Returns the probability of a word given the email class
def calculateWordProb(word, wordCount, totalEmails):
    return (wordCount.get(word, 0) + 1) / (totalEmails + len(wordCount))

# Classifies an email as spam or ham
def classifyEmail(email, spamWordCount, hamWordCount, priorSpamProb, 
                  priorHamProb, totalSpam, totalHam):
    words = word_tokenize(email)
    stopWords = set(stopwords.words('english'))
    
    pSpam = priorSpamProb
    pHam = priorHamProb
    
    for word in words:
        if word.isalpha() and word not in stopWords:
            word = word.lower()
            pSpam *= calculateWordProb(word, spamWordCount, totalSpam)
            pHam *= calculateWordProb(word, hamWordCount, totalHam)
            
    return pSpam > pHam

# Evaluates the model on the test dataset
def evaluateModel(testList, fileList, spamWordCount, hamWordCount, priorSpamProb, priorHamProb, totalSpam, totalHam):
   
    truePositive = falsePositive = trueNegative = falseNegative = 0
    
    for i, email in enumerate(testList):
        isSpam = classifyEmail(email, spamWordCount, hamWordCount, priorSpamProb, priorHamProb, totalSpam, totalHam)
        if isSpam and re.search(r'spmsg', fileList[i]):
            truePositive += 1
        elif isSpam:
            falsePositive += 1
        elif re.search(r'spmsg', fileList[i]):
            falseNegative += 1
        else:
            trueNegative += 1
    
    return truePositive, falsePositive, trueNegative, falseNegative

# Calculates the performance metrics
def calculateMetrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1Score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1Score

# Runs the Naive Bayes classifier
def runNaiveBayes(root_dir):
    spamList, hamList, testList, fileList = load_dataset(root_dir)

    spamWordCount, hamWordCount, priorSpamProb, priorHamProb = trainNaiveBayes(spamList, hamList)

    totalSpam = len(spamList)
    totalHam = len(hamList)
    
    tp, fp, tn, fn = evaluateModel(testList, fileList, spamWordCount, hamWordCount, priorSpamProb, priorHamProb, totalSpam, totalHam)
    
    accuracy, precision, recall, f1Score = calculateMetrics(tp, fp, tn, fn)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1Score:.4f}")
    
    return accuracy, precision, recall, f1Score

# Plotting the results
def plotMetrics(metrics):
    """
    Plot performance metrics.

    Parameters:
    metrics (tuple): Tuple containing accuracy, precision, recall, and F1 score.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Spam Filter Performance Metrics')
    plt.show()


if __name__ == '__main__':
    root_dir = "lingspam_public"
    metrics = runNaiveBayes(root_dir)
    plotMetrics(metrics)