# Data Science - Team 2: 2022 FIFA World Cup Winner Predictor
## Group Members
- Zachary Anderson (zanderson2)
- Ashley Armelin (aarmelin)
- Simran Moolchandaney (smoolcha)
- Jack Schlehr (jschlehr)
- Ryan Wachter (rwachte2)
## Project Description 
Given two countries playing a match in the FIFA World Cup, what will the outcome of the match be (i.e., which country will win/lose, or will there be a draw)? How does the model we create compare to the real-life outcomes of those matches?
In order to answer this question we developed 4 ML Classifiers: KNN, Random Forest, MLP, and SVM
## Files
- **internationalMatches.csv**: original dataset downloaded from Kaggle (prior to data pre-processing)
- **matchesWithFeatureVector.csv**: a file that contains all matches and their feature vector
- **preprocessedFifaMatches.csv**: pre-processed dataset
- **Resnet.py: main program running the classifiers
- **requirements.txt**: environment requirments (install by running "pip3 install -r requirements.txt")
## Usage
`Python3 Models.py`
## Implementation 
- **create_feature_vector()**: takes the initial dataset and creates our feature vector for each game
- **preprocess_data()**: performs data pre-processing
- **load_data()**: loads data from pre-processed file and encodes the classification labels
- **get_most_corr_feature(X,Y)**: this function find the top 10  correlated features using chi squared
- **get_knn_value_folds(X, Y)**: returns best k value for KNN after performing k-folds cross-validation (which also computes average and standard deviations for F1 Macro and Micro)
- **get_random_forest_value_folds(X, Y)**: returns best number of estimators value for Random Forest after performing k-folds cross-validation (which also computes average and standard deviations for F1 Macro and Micro)
- **MLP_K_folds(X, Y)**: performs k-folds cross-validation (which also computes average and standard deviations for F1 Macro and Micro) for MLP
- **SVM_K_folds(X, Y)**: performs k-folds cross-validation (which also computes average and standard deviations for F1 Macro and Micro) for SVM
- **knn(X, Y, k)**: runs K-Nearest-Neighbors classifier
- **random_forest(X, Y, estimators)**: runs Random Forest classifier
- **MLP(X, Y, hidden_layer)**: runs MLP classifier
- **SVM(X, Y)**: runs SVM classifier
- **plot_ROC_curve(model, xtrain, ytrain, xtest, ytest)** returns ROC curve for given classification model
- **plot_PR_curve(model, X_train, y_train, X_test, y_test)**: returns Precision Recall curve for given classification model
- **get_knn_value_folds(X,Y)**: obtain the confusion matrix and other statistics from the KNN model
- **get_random_forest_value_folds(X,Y)**: obtain the confusion matrix and other statistics from the random forest
- **baseline(Y)**: this is a simulation that we created to demonstrate baseline accuracy for guessing the outcome of a game