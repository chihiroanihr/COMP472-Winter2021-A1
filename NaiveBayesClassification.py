from __future__ import division
import string
from codecs import open
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB


def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = ''.join([word for word in line if word not in string.punctuation])  # remove punctuation
            words = words.lower().strip().split()  # make it lower
            review = list(set(words[3:]))  # remove duplicated text

            docs.append(review)  # append review doc part
            labels.append(words[1])  # append pos/neg part
    return docs, labels


def train_nb(documents, labels):
    classifier = MultinomialNB()
    classifier.fit(documents, labels)
    # return the data you need to classify new instances
    return classifier


# Predict a new document (test document)
def classify_nb(document, classifier):
    predict = classifier.predict(document)
    return predict
    # return the guess of the classifier


def accuracy(true_labels, guessed_labels):
    accuracy = accuracy_score(true_labels, guessed_labels)
    return accuracy





################## MAIN PART ###################


# Read the text file, divide into review documents and labels(pos/neg)
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

# Count the frequency of words in each documents
freq_words = Counter(words for doc in all_docs for words in doc)
    # print(freq_words)
# Count the frequency of positive and negative labels
freq_labels = Counter(labels for labels in all_labels)
    # print(freq_labels)

# Plot the number of each labels in bar chart
num_pos, num_neg = freq_labels['pos'], freq_labels['neg']
data = [num_pos, num_neg]
labels = ('Positive', 'Negative')
plt.xticks(range(len(data)), labels)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.ylim(5800, 6050)
plt.title('Number of labels(positive/negative)')
plt.bar(range(len(data)), data)
# Annotating the bar plot with the values (total label count)
for i in range(len(labels)):
    plt.annotate(data[i], (-0.06 + i, data[i] + 2))

plt.show()
plt.savefig('label_bar_chart.png')

# Split the data into training and an evaluation part
# train_docs, eval_docs, train_labels, eval_labels = train_test_split(documents, labels)
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]
# Transform nested list of text documents into matrix/numerical vectors
mlb = MultiLabelBinarizer()
train_docs = mlb.fit_transform(train_docs)
eval_docs = mlb.transform(eval_docs)


##### 1. Naive Bays Classifier #####
print("==========TRANING==========")
classifier = train_nb(train_docs, train_labels)  # classifier.fit(X_train, y_train)
train_prediction = classify_nb(train_docs, classifier)
print("Prediction label values: ")
print(train_prediction)
print("Actual label values: ")
print(train_labels)
# Result
accuracy_training = accuracy(train_labels, train_prediction)
report_training = classification_report(train_labels, train_prediction)
conf_matrix_training = confusion_matrix(train_labels, train_prediction)
# conf_matrix2_training = train_labels, train_prediction, labels = ['pos', 'neg']
print()
print(report_training)
print()
print("Accuracy Score: " + str(accuracy_training * 100) + "%")
print()
print("Confusion Matrix: ")
print(conf_matrix_training)
print()

print("==========TESTING==========")  # LETS TEST THE MODEL CLASSIFIER ON THE TEST DATA SET
eval_prediction = classify_nb(eval_docs, classifier)
print("Prediction label values: ")
print(eval_prediction)
print("Actual label values: ")
print(eval_labels)
# Result
accuracy_eval = accuracy(eval_labels, eval_prediction)
report_eval = classification_report(eval_labels, eval_prediction)
conf_matrix_eval = confusion_matrix(eval_labels, eval_prediction)
# conf_matrix2_eval = eval_labels, eval_prediction, labels = ['pos', 'neg']
print()
print(report_eval)
print()
print("Accuracy Score: " + str(accuracy_eval * 100) + "%")
print()
print("Confusion Matrix: ")
print(conf_matrix_eval)
print()
# Error Analysis
count_misclassified = (eval_labels != eval_prediction).sum()
print('Misclassified samples: {}'.format(count_misclassified))



### 3 output files for result ###
# [model name]-[database] --> file name
# row num of instance, index of predicted class of that instance
# plot confusion matrix
# precision, recall, f1-measure of each class
# the accuracy
with open('Naive-Bayes-database.txt', 'w') as file:
    file.write("Classification Used: Naive Bayes Classification\n\n\n")

    file.write("==========TRANING==========\n\n")
    file.write("Prediction label values: \n")
    file.write(str(train_prediction))
    file.write("\nActual label values: \n")
    file.write(str(train_labels))
    file.write('\n\n')
    file.write("Confusion Matrix: \n")
    file.write(str(conf_matrix_training))
    file.write('\n\n')
    file.write("Precision/Recall/f-1Measure: \n\n")
    file.write(report_training)
    file.write('\n\n')
    file.write("Accuracy Score: " + str(accuracy_training * 100) + "%")
    file.write('\n\n')
    file.write('Error Analysis:\nMisclassified samples: {}'.format(count_misclassified))
    file.write('\n\n\n')

    file.write("==========TESTING==========\n\n")
    file.write("Prediction label values: \n")
    file.write(str(eval_prediction))
    file.write("\nActual label values: \n")
    file.write(str(eval_labels))
    file.write('\n\n')
    file.write("Confusion Matrix: \n")
    file.write(str(conf_matrix_eval))
    file.write('\n\n')
    file.write("Precision/Recall/f-1Measure: \n\n")
    file.write(report_eval)
    file.write('\n\n')
    file.write("Accuracy Score: " + str(accuracy_eval * 100) + "%")
    file.write('\n\n')
    file.write('Error Analysis:\nMisclassified samples: {}'.format(count_misclassified))