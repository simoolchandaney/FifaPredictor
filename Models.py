from itertools import cycle
import os
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
import time
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.model_selection import KFold
import statistics
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from operator import itemgetter

#This is to suppress the warning from using .append() when creating the feature vectors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# The feature vector
features = ['home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_fifa_rank','neutral_location','home_team_4year_mean','away_team_4year_mean','home_team_4year_q1','away_team_4year_q1', 'home_team_4year_q2', 'away_team_4year_q2', 'home_team_4year_q3', 'away_team_4year_q3', 'home_team_4year_stdev','away_team_4year_stdev', 'home_team_4year_var', 'away_team_4year_var','home_team_4year_min_goals', 'away_team_4year_min_goals','home_team_4year_max_goals', 'away_team_4year_max_goals','home_team_4year_iqr_goals', 'away_team_4year_iqr_goals','home_team_wc_games', 'away_team_wc_games','home_team_4year_percent_wins', 'away_team_4year_percent_wins','home_team_4year_percent_loss', 'away_team_4year_percent_loss','home_team_4year_percent_draw', 'away_team_4year_percent_draw', 'home_team_rank_advantage']

def main():
    # create_feature_vector()
    # preprocess_data()
    df = load_data()
    X = df.loc[:, features ].to_numpy(dtype='float64', na_value=0)
    Y = df.loc[:, df.columns == "home_team_result"].to_numpy( dtype='int')
    most_corr_features = get_most_corr_feature(X, Y)
    X_best_features  = df.loc[:, most_corr_features ].to_numpy(dtype='float64', na_value=0)
    hidden_layer = (100,)
    #MLP_K_folds(X, Y)
    #MLP_K_folds(X_best_features, Y)

    MLP(X, Y, hidden_layer)
    #SVM_K_folds(X,Y)
    #SVM_K_folds(X_best_features,Y)
    # SVM(X, Y)
   
    #neighbors = get_knn_value_folds(X, Y)
    #knn(X, Y, neighbors)

    #estimators = get_random_forest_value_folds(X, Y)
    #random_forest(X, Y, estimators)
    # get_knn_value_folds(X,Y)
    # get_random_forest_value_folds(X,Y)

def create_feature_vector():
    print("Creating feature vector......................")
    df = pd.read_csv('internationalMatches.csv')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    fifa_rank = df[['date', 'home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank']]
    home = fifa_rank[['date', 'home_team','home_team_fifa_rank']].rename(columns = {'home_team': 'team', 'home_team_fifa_rank' : 'rank'})
    away = fifa_rank[['date', 'away_team','away_team_fifa_rank']].rename(columns = {'away_team': 'team', 'away_team_fifa_rank' : 'rank'})
    fifa_rank = home.append(away)

    # Select for each country the latest match
    fifa_rank  = fifa_rank.sort_values(['team', 'date'], ascending=[True, False])
    fifa_rank['row_number'] = fifa_rank.groupby('team').cumcount()+1
    fifa_rank_top = fifa_rank[fifa_rank['row_number']==1].drop('row_number',axis=1).nsmallest(211, 'rank')


    df['home_team_4year_mean'] = ''
    df['away_team_4year_mean'] = ''
    df['home_team_4year_q1'] = ''
    df['away_team_4year_q1'] = ''
    df['home_team_4year_q2'] = ''
    df['away_team_4year_q2'] = ''
    df['home_team_4year_q3'] = ''
    df['away_team_4year_q3'] = ''
    df['home_team_4year_stdev'] = ''
    df['away_team_4year_stdev'] = ''
    df['home_team_4year_var'] = ''
    df['away_team_4year_var'] = ''
    df['home_team_4year_min_goals'] = ''
    df['away_team_4year_min_goals'] = ''
    df['home_team_4year_max_goals'] = ''
    df['away_team_4year_max_goals'] = ''
    df['home_team_4year_iqr_goals'] = ''
    df['away_team_4year_iqr_goals'] = ''
    df['home_team_wc_games'] = ''
    df['away_team_wc_games'] = ''
    df['home_team_4year_percent_wins'] = ''
    df['away_team_4year_percent_wins'] = ''
    df['home_team_4year_percent_loss'] = ''
    df['away_team_4year_percent_loss'] = ''
    df['home_team_4year_percent_draw'] = ''
    df['away_team_4year_percent_draw'] = ''

    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        home_team = row['home_team']
        away_team = row['away_team']
        date = row['date']
        home_team_4_years_scores = df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['home_team'] == home_team ) ]['home_team_score'].append( df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['away_team'] == home_team ) ]['away_team_score'] )
        away_team_4_years_scores = df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['away_team'] == away_team ) ]['away_team_score'].append( df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['home_team'] == away_team ) ]['home_team_score'] )  
        
        home_team_4_years_wins = df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & (df['home_team'] == home_team ) & ( df['home_team_result'] == 'Win') ]['home_team_result'].append( df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['away_team'] == home_team ) & ( df['home_team_result'] == 'Lose')]['home_team_result'])
        home_team_4_years_draws = df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['home_team'] == home_team ) & ( df['home_team_result'] == 'Draw' ) ]['home_team_result'].append( df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['away_team'] == home_team ) & ( df['home_team_result'] == 'Draw') ]['home_team_result'] )
        home_team_4year_games_played = len(home_team_4_years_scores)


        away_team_4_years_games_played = len(away_team_4_years_scores)
        away_team_4_years_wins = df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & (df['home_team'] == away_team ) & ( df['home_team_result'] == 'Win') ]['home_team_result'].append( df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['away_team'] == away_team ) & ( df['home_team_result'] == 'Lose')]['home_team_result'])
        away_team_4_years_draws = df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['home_team'] == away_team ) & ( df['home_team_result'] == 'Draw' ) ]['home_team_result'].append( df.loc[ ( df['date'] > date - pd.DateOffset(years=4) ) & ( df['date'] < date ) & ( df['away_team'] == away_team ) & ( df['home_team_result'] == 'Draw') ]['home_team_result'] )
        
        try: df.at[index, 'home_team_4year_percent_draw'] = len(home_team_4_years_draws) / len(home_team_4_years_scores) * 100
        except: df.at[index, 'home_team_4year_percent_draw'] = 0
        try: df.at[index, 'home_team_4year_percent_wins'] = len(home_team_4_years_wins) / len(home_team_4_years_scores) * 100
        except: df.at[index, 'home_team_4year_percent_wins'] = 0
        try: df.at[index, 'home_team_4year_percent_loss'] = 100 - ((len(home_team_4_years_wins) + len(home_team_4_years_draws))  / len(home_team_4_years_scores) * 100) 
        except: df.at[index, 'home_team_4year_percent_loss'] = 0
        
        try: df.at[index, 'away_team_4year_percent_draw'] = len(away_team_4_years_draws) / len(away_team_4_years_scores) * 100
        except: df.at[index, 'away_team_4year_percent_draw'] = 0
        try: df.at[index, 'away_team_4year_percent_wins'] = len(away_team_4_years_wins) / len(away_team_4_years_scores) * 100
        except: df.at[index, 'away_team_4year_percent_wins'] = 0
        try: df.at[index, 'away_team_4year_percent_loss'] = 100 - (len(away_team_4_years_wins) + len(away_team_4_years_draws)) / len(away_team_4_years_scores) * 100
        except: df.at[index, 'away_team_4year_percent_loss'] = 0
        
        df.at[index,'home_team_4year_mean'] = home_team_4_years_scores.mean()
        df.at[index,'away_team_4year_mean'] = away_team_4_years_scores.mean()
        df.at[index,'home_team_4year_q1'] = home_team_4_years_scores.quantile(q=0.25)
        df.at[index,'away_team_4year_q1'] = away_team_4_years_scores.quantile(q=0.25)
        df.at[index,'home_team_4year_q2'] = home_team_4_years_scores.quantile(q=0.5)
        df.at[index,'away_team_4year_q2'] = away_team_4_years_scores.quantile(q=0.5)
        df.at[index,'home_team_4year_q3'] = home_team_4_years_scores.quantile(q=0.75)
        df.at[index,'away_team_4year_q3']    = away_team_4_years_scores.quantile(q=0.75)
        df.at[index,'home_team_4year_stdev'] = home_team_4_years_scores.std()
        df.at[index,'away_team_4year_stdev'] = away_team_4_years_scores.std()
        df.at[index,'home_team_4year_var'] = home_team_4_years_scores.var()
        df.at[index,'away_team_4year_var'] = away_team_4_years_scores.var()
        df.at[index,'home_team_4year_min_goals'] = home_team_4_years_scores.min()
        df.at[index,'away_team_4year_min_goals'] = away_team_4_years_scores.min()
        df.at[index,'home_team_4year_max_goals'] = home_team_4_years_scores.max()
        df.at[index,'away_team_4year_max_goals'] = away_team_4_years_scores.max()
        df.at[index,'home_team_4year_iqr_goals'] = home_team_4_years_scores.quantile(q=0.75) - home_team_4_years_scores.quantile(q=0.25)
        df.at[index,'away_team_4year_iqr_goals'] = away_team_4_years_scores.quantile(q=0.75) - away_team_4_years_scores.quantile(q=0.25)
        # Total WC games played
        home_team_wc = df.loc[ ( df['date'] < date ) & ( df['home_team'] == home_team ) & ( df['tournament'] == 'FIFA World Cup' ) ].append(df.loc[ ( df['date'] < date ) & ( df['away_team'] == home_team ) & ( df['tournament'] == 'FIFA World Cup' ) ])
        away_team_wc = df.loc[ ( df['date'] < date ) & ( df['away_team'] == home_team ) & ( df['tournament'] == 'FIFA World Cup' ) ].append(df.loc[ ( df['date'] < date ) & ( df['home_team'] == home_team ) & ( df['tournament'] == 'FIFA World Cup' ) ])
        df.at[index,'home_team_wc_games'] = len(home_team_wc)
        df.at[index,'away_team_wc_games'] = len(home_team_wc)


    df.to_csv('matchesWithFeatureVector.csv')
    print("Successfully created feature vector and saved to csv.............")
    
def preprocess_data():
    print("Preprocessing data....................")
    df = pd.read_csv('matchesWithFeatureVector.csv', index_col=0)
    df['date'] = pd.to_datetime(df['date'])

    # get rid of first 4 years because their feature vector will have all 0 values
    min_date = pd.to_datetime( df['date'].min() )
    min_date += pd.DateOffset(years=4)
    df.loc[ df['date'] >= min_date ]

    # encode the team names
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.home_team)
    df.home_team = label_encoder.fit_transform(df.home_team)
    df.away_team = label_encoder.fit_transform(df.away_team)

    # encode the label
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df.home_team_result)
    df.home_team_result = label_encoder.fit_transform(df.home_team_result)
    
    # append away_team_result
    away_team_result = []
    for result in df.home_team_result:
        if result == 0:
            away_team_result.append(1)
        elif result == 1:
           away_team_result.append(0)
        else:
            away_team_result.append(2)
    df['away_team_result'] = away_team_result

    # add fifa team rank difference
    df['home_team_rank_advantage'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
    min_val = min(df['home_team_rank_advantage'])
    df['home_team_rank_advantage'] -= min_val

    # Fill the null values with averages for that team
    for feature in features:
        if not df[feature].isnull().values.any():
            continue 

        if 'home' in feature:
            away_feature = feature.replace('home', 'away')
            sum = df.groupby('home_team')[feature].transform('sum') + df.groupby('away_team')[away_feature].transform('sum')
            count=df.groupby('home_team')[feature].transform('count')+ df.groupby('away_team')[away_feature].transform('count')
            df[feature].fillna(  sum/count, inplace = True)
        elif 'away' in feature:
            home_feature = feature.replace('away', 'home' )
            sum = df.groupby('home_team')[home_feature].transform('sum') + df.groupby('away_team')[feature].transform('sum')
            count=df.groupby('home_team')[home_feature].transform('count')+ df.groupby('away_team')[feature].transform('count')
            df[feature].fillna(  sum/count, inplace = True)

    cols_to_keep = features
    cols_to_keep.extend(['home_team_result','date'])
    
    df.drop(df.columns.difference(cols_to_keep ), 1, inplace=True)
    

     
    df.to_csv('preprocessedFifaMatches.csv')   
    print("Successfully preprocessing data and saved to csv....................")


def load_data():
    print("Loading data from csv.................")
    df = pd.read_csv('preprocessedFifaMatches.csv', index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    print("CSV of preprocessed matches loaded.............")
    return df

def get_most_corr_feature(X, Y):
    features = ['home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_fifa_rank','neutral_location','home_team_4year_mean','away_team_4year_mean','home_team_4year_q1','away_team_4year_q1', 'home_team_4year_q2', 'away_team_4year_q2', 'home_team_4year_q3', 'away_team_4year_q3', 'home_team_4year_stdev','away_team_4year_stdev', 'home_team_4year_var', 'away_team_4year_var','home_team_4year_min_goals', 'away_team_4year_min_goals','home_team_4year_max_goals', 'away_team_4year_max_goals','home_team_4year_iqr_goals', 'away_team_4year_iqr_goals','home_team_wc_games', 'away_team_wc_games','home_team_4year_percent_wins', 'away_team_4year_percent_wins','home_team_4year_percent_loss', 'away_team_4year_percent_loss','home_team_4year_percent_draw', 'away_team_4year_percent_draw', 'home_team_rank_advantage']
    chi2_selector = SelectKBest(score_func=chi2, k="all")
    fit = chi2_selector.fit(X, Y)
    values = fit.scores_
    vals = NormalizeData(values)
    dict_chi = {}
    i = 0
    for feature in features:
        dict_chi[feature] = vals[i]
        i+=1
    feature_names = list(dict_chi.keys())
    coeffs = list(dict_chi.values())
    plt.barh(feature_names, coeffs)
    plt.title('Chi square - feature/label correlation')
    plt.ylabel('feature')
    plt.xlabel('coeff')
    plt.show() 

    res = dict(sorted(dict_chi.items(), key = itemgetter(1), reverse = True)[:10])
    corr_features = res.keys()
    return list(corr_features)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_knn_value_folds(X, Y):
    kf = KFold(n_splits=10)
    accs = []
    best_acc = 0
    std = 0
    best_k = 0
    for i in tqdm(range(2,200)):
        accs = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            neigh = KNeighborsClassifier(n_neighbors = i)
            neigh.fit(X_train, y_train.ravel())
            y_knn = neigh.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_knn)
            accs.append(acc)
        if statistics.mean(accs) > best_acc:
            best_acc = statistics.mean(accs)
            std = statistics.stdev(accs)
            best_k = i 
    print(best_acc)
    print(std)
    print(best_k)
    return best_k

def get_random_forest_value_folds(X, Y):
    kf = KFold(n_splits=10)
    accs = []
    best_acc = 0
    std = 0
    best_n = 0
    for i in tqdm(range(2,200)):
        accs = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            clf = RandomForestClassifier(n_estimators = i)
            clf.fit(X_train, y_train.ravel())
            y_knn = clf.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_knn)
            accs.append(acc)
        if statistics.mean(accs) > best_acc:
            best_acc = statistics.mean(accs)
            std = statistics.stdev(accs)
            best_n = i
    print(best_acc)
    print(std)
    print(best_n)
    return best_n

def MLP_K_folds(X, Y):
    kf = KFold(n_splits=10)
    accs = []
    hidden_layer_sizes = (100,)
    for train_index, test_index in tqdm(kf.split(X), total=kf.get_n_splits(), desc="k-fold"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        mlp = MLPClassifier(solver='adam', hidden_layer_sizes=hidden_layer_sizes, max_iter=400 )
        mlp.fit(X_train, np.ravel(y_train))
        y_pred = mlp.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        accs.append(acc)
    print(f'{round(sum(accs)/len(accs)*100,3)}% average accuracy from MLP with {np.shape(X)[1]} features')
    print(f'STDEV of {round(statistics.stdev(accs),3)} from MLP with {np.shape(X)[1]} features')
    return 100

def SVM_K_folds(X, Y):
    kf = KFold(n_splits=10)
    accs = []
    for train_index, test_index in tqdm(kf.split(X), total=kf.get_n_splits(), desc="k-fold"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        svm = SVC( )
        svm.fit(X_train, np.ravel(y_train))
        y_pred = svm.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        accs.append(acc)
    print(f'{round(sum(accs)/len(accs)*100,3)}% average accuracy from SVM with {np.shape(X)[1]} features')
    print(f'STDEV of {round(statistics.stdev(accs),3)} from SVM with {np.shape(X)[1]} features')

def knn(X, Y, k):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors = k, weights = 'uniform',algorithm = 'brute',metric = 'minkowski')
    knn.fit(X_train, np.ravel(y_train))
    y_knn = knn.predict(X_test)
    print('KNN')
    print(confusion_matrix(y_test, y_knn))
    print(classification_report(y_test, y_knn))
    # plot ROC curve
    plot_ROC_curve(knn, X_train, y_train, X_test, y_test)
    # plot presicion recall curve
    plot_PR_curve(knn, X_train, y_train, X_test, y_test)

def random_forest(X, Y, estimators):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators = estimators) 
    clf.fit(X_train, np.ravel(y_train))
    y_pred = clf.predict(X_test)
    print('RANDOM FOREST')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # plot ROC curve
    plot_ROC_curve(clf, X_train, y_train, X_test, y_test)
    # plot presicion recall curve
    plot_PR_curve(clf, X_train, y_train, X_test, y_test)

def MLP(X, Y, hidden_layer):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=hidden_layer, max_iter=400 )
    mlp.fit(X_train, np.ravel(y_train))
    pred = mlp.predict(X_test)
    print('-------------------------------------MLP-------------------------------------')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    f1_micro = f1_score(y_test, pred, average='micro')
    f1_macro = f1_score(y_test, pred, average='macro')
    print(f'F1 Micro score for MLP {f1_micro}')
    print(f'F1 Macro score for MLP {f1_macro}')
    plot_ROC_curve(mlp, X_train, y_train, X_test, y_test)
    plot_PR_curve(mlp, X_train, y_train, X_test, y_test)
    print('-----------------------------------------------------------------------------')

def SVM(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    svm = SVC()
    svm.fit(X_train, np.ravel(y_train))
    pred = svm.predict(X_test)
    print('-------------------------------------SVM-------------------------------------')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    f1_micro = f1_score(y_test, pred, average='micro')
    f1_macro = f1_score(y_test, pred, average='macro')
    print(f'F1 Micro score for SVM {f1_micro}')
    print(f'F1 Macro score for SVM {f1_macro}')
    plot_ROC_curve(svm, X_train, y_train, X_test, y_test)
    plot_PR_curve(svm, X_train, y_train, X_test, y_test)
    print('-----------------------------------------------------------------------------')

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):
    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'win', 1: 'loss', 2: 'draw'})                
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()

def plot_PR_curve(model, X_train, y_train, X_test, y_test):
    # plot presicion recall curve
    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(model, per_class=True, cmap="Set1", encoder={0: 'win', 1: 'loss', 2: 'draw'})
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()
    
def get_knn_value_folds(X, Y):

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    kf = KFold(n_splits=10)
    total_accs = []
    total_macros = []

    total_accs_std = []
    total_macros_std = []

    i_arr = []
    best_macro = 0
    std = 0
    best_k = 0
    printedReport = False

    for i in tqdm(range(1,50,2)):
        i_arr.append(i)
        accs = []
        f1_micros = []
        f1_macros = []

        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            neigh = KNeighborsClassifier(n_neighbors = i)
            neigh.fit(X_train, y_train.ravel())
            y_knn = neigh.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_knn)
            f1_micro = f1_score(y_test, y_knn, average='micro')
            f1_macro = f1_score(y_test, y_knn, average='macro')

            accs.append(acc)
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)
            if not printedReport :
                print(confusion_matrix(y_test, y_knn))
                print(classification_report(y_test, y_knn))
                printedReport = True

        total_accs.append(statistics.mean(accs))
        total_macros.append(statistics.mean(f1_macros))

        total_accs_std.append(statistics.stdev(accs))
        total_macros_std.append(statistics.stdev(f1_macros))

        if statistics.mean(f1_macros) > best_macro:
            best_acc = statistics.mean(accs)
            best_f1_micros = statistics.mean(f1_micros)
            best_macro = statistics.mean(f1_macros)
            
            std = statistics.stdev(accs)
            micros_std = statistics.stdev(f1_micros)
            macros_std = statistics.stdev(f1_macros)
        
            best_k = i
    
    print("Accuracies:")
    print(best_acc)
    print(std)

    print("F1_Micros:")
    print(best_f1_micros)
    print(micros_std)

    print("F1_Macros:")
    print(best_macro)
    print(macros_std)

    print(best_k)

    plt.plot(i_arr, total_accs, 'bo')
    plt.show()

    plt.plot(i_arr, total_macros, 'ro')
    plt.show()


    plt.plot(i_arr, total_accs_std, 'go')
    plt.show()

    plt.plot(i_arr, total_macros_std, 'mo')
    plt.show()



def get_random_forest_value_folds(X, Y):
    
    kf = KFold(n_splits=10)

    total_accs = []
    total_macros = []

    total_accs_std = []
    total_macros_std = []

    i_arr = []
    best_macro = 0
    std = 0
    best_k = 0
    printedReport = False

    for i in tqdm(range(13,14)):
        i_arr.append(i)
        accs = []
        f1_micros = []
        f1_macros = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            clf = RandomForestClassifier(n_estimators = i)
            clf.fit(X_train, y_train.ravel())
            y_knn = clf.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_knn)
            f1_micro = f1_score(y_test, y_knn, average='micro')
            f1_macro = f1_score(y_test, y_knn, average='macro')
            
            accs.append(acc)
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)
            if not printedReport:
                print(confusion_matrix(y_test, y_knn))
                print(classification_report(y_test, y_knn))

                plot_PR_curve(clf, X_train, y_train, X_test, y_test)

                plot_ROC_curve(clf, X_train, y_train, X_test, y_test)
                printedReport = True
 

        total_accs.append(statistics.mean(accs))
        total_macros.append(statistics.mean(f1_macros))

        total_accs_std.append(statistics.stdev(accs))
        total_macros_std.append(statistics.stdev(f1_macros))

        if statistics.mean(f1_macros) > best_macro:
            best_acc = statistics.mean(accs)
            best_f1_micros = statistics.mean(f1_micros)
            best_macro = statistics.mean(f1_macros)
            
            std = statistics.stdev(accs)
            micros_std = statistics.stdev(f1_micros)
            macros_std = statistics.stdev(f1_macros)
        
            best_k = i
    
    print("Accuracies:")
    print(best_acc)
    print(std)

    print("F1_Micros:")
    print(best_f1_micros)
    print(micros_std)

    print("F1_Macros:")
    print(best_macro)
    print(macros_std)

    print(best_k)

    plt.plot(i_arr, total_accs, 'bo')
    plt.show()

    plt.plot(i_arr, total_macros, 'ro')
    plt.show()


    plt.plot(i_arr, total_accs_std, 'go')
    plt.show()

    plt.plot(i_arr, total_macros_std, 'mo')
    plt.show()

def baseline(Y):
    n_wins = 10449
    n_draws = 4849
    n_losses = 6090

    predictions = []
    for i in range(23921):

        a = random.randint(1, 23921)

        if a <=  n_losses:
            predictions.append(0)
        elif a <= n_losses+n_draws:
            predictions.append(1)
        else:
            predictions.append(2)
    
    print(metrics.accuracy_score(Y, predictions))
    print(f1_score(Y, predictions, average='macro'))
 
if __name__ == "__main__":
    main()