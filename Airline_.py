# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data set and processing
dataset = pd.read_csv('airlines_delay.csv')

airline_labels = ['Airline']
dataset = pd.get_dummies(dataset, columns=airline_labels, drop_first=False)
airportfrom_labels = ['AirportFrom']
dataset = pd.get_dummies(dataset, columns=airportfrom_labels, drop_first=False)
airportto_labels = ['AirportTo']
dataset = pd.get_dummies(dataset, columns=airportto_labels, drop_first=False)


dataset = dataset.drop(['Flight','DayOfWeek'],axis=1)


cols = list(dataset.columns)
a, b = cols.index('Time'), cols.index('Class')
cols[b], cols[a] = cols[a], cols[b]
dataset = dataset[cols]

x = dataset.iloc[:,1:-1]
y = dataset.iloc[:, 0]


# spliting the data to trining and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25
                                                    , random_state = 0)


# Standardscaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state= 0)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)



# confusion_matrix creation for the RFC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
confusion_matrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print('precision',precision)

recall = recall_score(y_test, y_pred)
print('recall',recall)

import seaborn as sns
plt.figure(figsize=(7,7))
sns.heatmap(data=confusion_matrix,linewidth=3, annot=True, square=True, cmap='Blues')
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
all_sample_title = 'Accuracy Score : {0}'.format(classifier.score(x_test,y_test))
plt.title(all_sample_title, size=15)




# viualising the SVC test result
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], alpha=0.9,
c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Airlines Delay')
plt.legend()
plt.show()