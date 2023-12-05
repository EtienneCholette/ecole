# Separate Target Variable and Predictor Variables
TargetVariable = ['Survived']
Predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
              'Embarked_C', 'Embarked_Q', 'Embarked_S']

X = TitanicSurvivalDataNumeric[Predictors].values
y = TitanicSurvivalDataNumeric[TargetVariable].values

### Sandardization of data ###
### We does not standardize the Target variable for classification
from sklearn.preprocessing import StandardScaler

PredictorScaler = StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X)

# Generating the standardized values of X and y
X = PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Quick sanity check with the shapes of Training and Testing datasets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

classifier = Sequential()
# Defining the Input layer and FIRST hidden layer,both are same!
# relu means Rectifier linear unit function
classifier.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))

# Defining the SECOND hidden layer, here we have not defined input because it is
# second layer and it will get input as the output of first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Defining the Output layer
# sigmoid means sigmoid activation function
# for Multiclass classification the activation ='softmax'
# And output_dim will be equal to the number of factor levels
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Optimizer== the algorithm of SGG to keep updating weights
# loss== the loss function to measure the accuracy
# metrics== the way we will compare the accuracy after each step of SGD
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the Neural Network on the training data
survivalANN_Model = classifier.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1)

# fitting the Neural Network on the training data
survivalANN_Model = classifier.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1)


# Defining a function for finding best hyperparameters
def FunctionFindBestParams(X_train, y_train):
    # Defining the list of hyper parameters to try
    TrialNumber = 0
    batch_size_list = [5, 10, 15, 20]
    epoch_list = [5, 10, 50, 100]

    import pandas as pd
    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1

            # Creating the classifier ANN model
            classifier = Sequential()
            classifier.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            survivalANN_Model = classifier.fit(X_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial,
                                               verbose=0)
            # Fetching the accuracy of the training
            Accuracy = survivalANN_Model.history['accuracy'][-1]

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'Accuracy:', Accuracy)

            SearchResultsData = SearchResultsData.append(pd.DataFrame(data=[[TrialNumber,
                                                                             'batch_size' + str(
                                                                                 batch_size_trial) + '-' + 'epoch' + str(
                                                                                 epochs_trial), Accuracy]],
                                                                      columns=['TrialNumber', 'Parameters',
                                                                               'Accuracy']))
    return (SearchResultsData)


###############################################

# Calling the function
ResultsData = FunctionFindBestParams(X_train, y_train)

# Printing the best parameter
print(ResultsData.sort_values(by='Accuracy', ascending=False).head(1))

# Visualizing the results
% matplotlib
inline
ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15, 4), kind='line', rot=20)
classifier.fit(X_train,y_train, batch_size=5 , epochs=100, verbose=1)
