import os
import re
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from nltk.tokenize import sent_tokenize, word_tokenize

def getWordSpamProb(word): #returns the probability that a spam file will contain this word
    if word not in spamWordCount:
        return 0
    spamProb = spamWordCount[word]/len(spamList)
    return spamProb

def getWordHamProb(word): #returns the probability that a ham file will contain this word
    if word not in hamWordCount:
        return 0
    hamProb = hamWordCount[word]/len(hamList)
    return hamProb

def getFinalWordSpamProb(word): #returns the probability that a file that contains this word is spam
    spamProb = getWordSpamProb(word)
    hamProb = getWordHamProb(word)
    if (hamProb == 0 and spamProb == 0):
        return 0
    finalSpamProb = (spamProb * priorSpamProb) / ((spamProb * priorSpamProb) + (hamProb * priorHamProb)) #Bayes theorem
    return finalSpamProb



dirList = ["bare", "lemm", "lemm_stop", "stop"]
for d in dirList:
    rootdir = "lingspam_public\\{0}".format(d)
    spamList = [] #list that contains spam files from parts 1-9
    hamList = [] #list that contains ham files from parts 1-9
    testList = [] #list that contains files from part 10
    fileList = [] #list that contains filenames from part 10
    spamFileLength = [] #list that contains the length of spam messages
    hamFileLength = [] #list that contains the length of ham messages
    spamSet = set() #set that contains words from spamList
    hamSet = set() #set that contains words from hamList
    fileSet = set() #set that contains words from current file so no duplicates are counted
    predictedSpam = set() #set that contains files filtered as spam
    predictedHam = set() #set that contains files predicted as ham
    spamWordCount = {}
    hamWordCount = {}
    spamFilter = {}
    failedPredictions = 0
    correctSpamPredictions = 0
    correctHamPredictions = 0
    incorrectSpamPredictions = 0
    totalPop = 0
    accuracy = []
    recall = []
    precision = []
    f1score = []
    p = 1
    pn = 1
    stopWords = set()
    
    for directories, subdirs, files in os.walk(rootdir):
        pass

    for subdirs in os.walk(rootdir):
        if (len(subdirs[1]) == 0):
                subdirName = (format(subdirs[0].split('\\')[-1]))
                if (subdirName == "part10"):
                    for file in files:
                        with open(os.path.join(directories, file)) as f:
                                testText = f.read()
                                testList.append(testText)
                                fileList.append(file)
                else:
                    for file in files:
                        if (re.search("spmsg*", file)):
                            with open(os.path.join(directories, file)) as f:
                                spamText = f.read()
                                spamList.append(spamText)
                        else:
                            with open(os.path.join(directories, file)) as f:
                                hamText = f.read()
                                hamList.append(hamText)

    for spam in spamList:
        spamWordList = word_tokenize(spam)
        for word in spamWordList:
            if (word not in stopWords) and (word not in fileSet): #If the word hasn't already appeared in this file
                spamFileLength.append(len(spamWordList)) 
                spamWordCount[word] = spamWordCount.get(word, 0) + 1 #Increase the number of times this word has been found in spam files
                fileSet.add(word)
        fileSet = set()
            
    for ham in hamList:
        hamWordList = word_tokenize(ham)
        for word in hamWordList:
            if (word not in stopWords) and (word not in fileSet): #If the word hasn't already appeared in this file
                hamFileLength.append(len(hamWordList))
                hamWordCount[word] = hamWordCount.get(word, 0) + 1 #Increase the number of times this word has been found in ham files
                fileSet.add(word)
        fileSet = set()

    priorSpamProb = len(spamList)/(len(spamList)+len(hamList))
    priorHamProb = len(hamList)/(len(spamList)+len(hamList))

    l = 0
    for trainFile in spamList:
        trainWordList = word_tokenize(trainFile)
        for word in trainWordList:
            if (word not in stopWords) and (word not in spamFilter):
                spamFilter[word] = getFinalWordSpamProb(word) #set the value of the key 'word' as the probability that a file containing this word is spam
                l = l + spamFilter[word]
        fileAvg = l/len(spamFilter)
        l = 0
    limit = fileAvg/len(spamList)

    for trainFile in hamList:
        trainWordList = word_tokenize(trainFile)
        for word in trainWordList:
            if (word not in stopWords) and (word not in spamFilter):
                spamFilter[word] = getFinalWordSpamProb(word)

    fileCount = 0
    for test in testList:
        testWordList = word_tokenize(test)
        for word in testWordList:
            if word not in stopWords:
                p *= spamFilter[word]
                pn *= (1-p)
        fileProb = p/(p+pn) #Bayes theorem
        if (fileProb > limit):
            predictedSpam.add(test)
        else:
            predictedHam.add(test)
        if ((fileProb <= limit) and (re.search("spmsg*", fileList[fileCount]))):
            failedPredictions += 1
            totalPop += 1
        if ((fileProb <= limit) and not (re.search("spmsg*", fileList[fileCount]))):
            correctHamPredictions += 1
            totalPop += 1
        if ((fileProb > limit) and (re.search("spmsg*", fileList[fileCount]))):
            correctSpamPredictions += 1
            totalPop += 1
        if ((fileProb > limit) and not (re.search("spmsg*", fileList[fileCount]))):
            incorrectSpamPredictions += 1
            totalPop += 1
        if (correctSpamPredictions > 0 or incorrectSpamPredictions > 0):
            precision.append(correctSpamPredictions/(incorrectSpamPredictions+correctSpamPredictions))
        if (correctSpamPredictions > 0 or incorrectSpamPredictions > 0 or failedPredictions > 0):
            recall.append(correctSpamPredictions/(incorrectSpamPredictions+correctSpamPredictions+failedPredictions))
        accuracy.append((correctSpamPredictions+correctHamPredictions)/totalPop)

        p = 1
        pn = 1
        fileCount += 1
        
    for i in range(len(recall)):
        f1score.append(2*((precision[i]*recall[i])/(precision[i]+recall[i])))

    plt.figure(1)

    plt.plot(recall) #Recall curve
    plt.xlabel('Recall')

    plt.figure(2)

    plt.plot(precision) #Precision curve
    plt.xlabel('Precision')

    plt.figure(3)

    plt.plot(accuracy) #Accuracy curve
    plt.xlabel('Accuracy')

    plt.figure(4)

    plt.plot(f1score) #F1 score curve
    plt.xlabel('F1 Score')


    plt.show()

