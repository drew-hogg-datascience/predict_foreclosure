import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics as metrics
from sklearn import model_selection

from setup_data_reduction import *

def filter_data(data):


    le = LabelEncoder()
    data['occupancy_status'] = le.fit_transform(data['occupancy_status'])

    initial_rows = data.shape[0]
    data = data.drop(columns = ['Unnamed: 0', 'co_borrower_credit_score', 'foreclosure_date', 'recovery_costs', 'time_bins', 'credit_score_bins', 'interest_rate', 'property_state'])
    data = data.dropna(axis=0, how='any')
    final_rows = data.shape[0]
    print 'Dropped ' + str(1.0-final_rows/initial_rows) + r'% of data in filtering'

    defaults = data[data['foreclosure']==1.0]
    good = data[data['foreclosure']==0.0]
    frac = (1.0*defaults.shape[0])/(1.0*good.shape[0])
    data = pd.concat([defaults, good.sample(frac=frac*2.)]).sample(frac=1.0)

    return data

def make_logistic_predictions(data):

    data = filter_data(data)

    classifier = LogisticRegression(random_state=1, class_weight="balanced")

    predictors = data.columns.tolist()
    #predictors = [p for p in predictors if p not in ['id', 'foreclosure', 'occupancy_status', 'acquisition_date', 'origination_date', 'zip', 'first_time_homebuyer', 'property_state']]
    predictors = [p for p in predictors if p not in ['id', 'foreclosure', 'acquisition_date', 'origination_date']]

    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data['foreclosure'], test_size=0.3, stratify=data['foreclosure'])

    kfold = model_selection.KFold(n_splits=5, random_state=0)

    results = model_selection.cross_val_score(classifier, X_train, y_train, cv=kfold)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    prediction_prob = classifier.predict_proba(X_test)


    output_data = X_test.join(data['origination_date'])
    output_data['foreclosure'] = y_test
    output_data['prediction'] = predictions
    output_data['probability'] = prediction_prob[:,1]

    diagnostics(y_test, predictions, prediction_prob)

    return classifier, output_data

def make_rand_forest_grid():
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.logspace(start = 10, stop = 1000, num = 10)]
    n_estimators = [10, 20, 40, 80, 160, 320, 640, 1280]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth = [2, 4, 8, 16, 32, 64, 128]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    return grid

def make_rand_forest_predictions(data):

    data = filter_data(data)

    classifier = RandomForestClassifier(n_estimators = 20, n_jobs=-1, class_weight='balanced')

    predictors = data.columns.tolist()
    predictors = [p for p in predictors if p not in ['id', 'foreclosure', 'acquisition_date', 'origination_date']]

    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data['foreclosure'], test_size=0.3, stratify=data['foreclosure'])

    grid = make_rand_forest_grid()
    kfold = model_selection.KFold(n_splits=5, random_state=0)
    CV_rfc = RandomizedSearchCV(estimator = classifier,
                                param_distributions = grid,
                                n_iter = 10,
                                cv = kfold,
                                random_state=1,
                                n_jobs = -1)
    CV_rfc.fit(X_train, y_train)
    params = CV_rfc.best_params_
    print 'Best parameters for random forest classification are ', params

    best_fit = CV_rfc.best_estimator_
    predictions = best_fit.predict(X_test)
    prediction_prob = best_fit.predict_proba(X_test)
    print prediction_prob.shape

    output_data = X_test.join(data['origination_date'])
    output_data['foreclosure'] = y_test
    output_data['prediction'] = predictions
    output_data['probability'] = prediction_prob[:,1]

    # Calcuate various accuracy rates
    diagnostics(y_test, predictions, prediction_prob)

    return best_fit, output_data

def get_importances(features, classifier):

    feature_importances = pd.DataFrame(classifier.feature_importances_,
                                        index = features,
                                        columns=['importance']).sort_values('importance',  ascending=False)

    return feature_importances

def diagnostics(true, predictions, prediction_prob):
    print 'Accuracy Score = ', metrics.accuracy_score(true, predictions)

    tn, fp, fn, tp = metrics.confusion_matrix(true, predictions).ravel()
    print 'True Neg = ', tn, '   False Pos = ', fp,'   False Neg = ', fn, '   True Pos = ', tp

    print metrics.classification_report(true, predictions)

    print 'Log-Loss Score = ', metrics.log_loss(true, prediction_prob)

def correlations(data):

    del data['id']
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    '''
    pg = sns.PairGrid(data[::100], vars=['interest_rate',
                'balance',
                'ltv',
                'dti',
                'borrower_credit_score',
                'first_time_homebuyer',
                'property_state',
                'foreclosure'], hue='foreclosure', palette='RdBu_r')
    pg.map(plt.scatter, alpha=0.8)
    pg.add_legend()
    plt.show()
    '''
    #list_of_cmaps=['Blues','Greens','Reds','Purples']
    g = sns.PairGrid(data.dropna(axis=0, how='any'), vars=['interest_rate',
                'balance',
                'ltv',
                'dti',
                'borrower_credit_score',
                'first_time_homebuyer',
                'property_state',
                'foreclosure'], hue='foreclosure', palette='RdBu_r')
    g.map_upper(plt.scatter)
    g.map_lower(sns.kdeplot,shade=True, cmap=cmap, shade_lowest=False)
    #g.map_diag(sns.distplot)
    # g.map_diag(plt.hist)
    g.add_legend()
    plt.show()

def percentage(value, total):
    return value/total
