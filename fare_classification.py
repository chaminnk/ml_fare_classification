import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from pyproj import Geod


wgs84_geod = Geod(ellps='WGS84') #Distance will be measured on this ellipsoid - more accurate than a spherical method

#Get distance between pairs of lat-lon points
def Distance(lat1,lon1,lat2,lon2):
  az12,az21,dist = wgs84_geod.inv(lon1,lat1,lon2,lat2) 
  return dist


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


input_file = "train.csv"
df = pd.read_csv(input_file)
df.fillna(df.mean(), inplace=True)
#df.fillna(0, inplace=True)
X = df
##X = df.drop(columns=["tripid", "pickup_time","drop_time","label"])
X['distance'] = Distance(X['pick_lat'].tolist(),X['pick_lon'].tolist(),X['drop_lat'].tolist(),X['drop_lon'].tolist())
X = df.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon","label"])
#X = df.drop(columns=["tripid","label"])
with pd.option_context('display.max_columns', None):  
    print(X)
X = X.iloc[:,:].values
print(X)

df['output_label'] = (df['label'] == 'correct').astype('int')
y = df["output_label"].values



input_file2 = "test.csv"
df2 = pd.read_csv(input_file2)
df2.fillna(df2.mean(), inplace=True)
#df2.fillna(0, inplace=True)
tripid_test = np.asarray(df2.iloc[:, 0].values)
X2 = df2
##X2 = df2.drop(columns=["tripid","pickup_time","drop_time"])
X2['distance'] = Distance(X2['pick_lat'].tolist(),X2['pick_lon'].tolist(),X2['drop_lat'].tolist(),X2['drop_lon'].tolist())
X2 = df2.drop(columns=["tripid","pickup_time","drop_time","pick_lat","pick_lon","drop_lat","drop_lon"])
#X2 = df2.drop(columns=["tripid"])
with pd.option_context('display.max_columns', None):  
    print(X2)
X2 = X2.iloc[:,:].values
print(X2)
#print (df.iloc[4080:4085])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#print(X_train)
lr = 0.404
def knnClassifier(X_train,X_test,y_train,y_test):
    print("knn")
    model1 = KNeighborsClassifier()
    model1.fit(X_train, y_train)
    y_pred=model1.predict(X_test)
    print(f1_score(y_test,y_pred))

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    print(model1.get_params())
def adaboostClassifier(X_train,X_test,y_train,y_test):
    print("adaboost")
    model2 = AdaBoostClassifier(random_state=1, learning_rate=0.404)
    model2.fit(X_train, y_train)
    y_pred=model2.predict(X_test)
    
    print(f1_score(y_test,y_pred))

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    print(model2.get_params())
    
##model3= GradientBoostingClassifier(learning_rate=lr,random_state=1)
##print('gbc')
##model3.fit(X_train, y_train)
##pred3=model3.predict(X_test)
##print(model3.score(y_test,pred3))

##model4=xgb.XGBClassifier(random_state=1,learning_rate=lr)
##model4=xgb.XGBClassifier(learning_rate =0.401,
## n_estimators=1000,
## max_depth=5,
## min_child_weight=1,
## gamma=0,
## subsample=0.8,
## colsample_bytree=0.8,
## objective= 'binary:logistic',
## nthread=4,
## scale_pos_weight=1,
## seed=27)
def xgboostModel(X_train,X_test,y_train,y_test,tripid_test):
    print("xgboost")
    
##    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
##    learning_rate = []
##    n = 0.402
##    for i in range (0,51):
##        n = n + 0.0001
##        n = round(n,4)
##        learning_rate.append(n)
##    params = {
##        'n_estimators':n_estimators,
##        'learning_rate':learning_rate,
##        'min_child_weight': [1, 5, 10],
##        'gamma': [8, 9, 10, 12, 15],
##        'subsample': [0.7,0.8, 0.9, 1.0,1.1],
##        'colsample_bytree': [0.7,0.8, 0.9, 1.0,1.1],
##        'max_depth': [4, 5, 7,9,11]
##        }
##    param_comb = 50
    
##    model4 = XGBClassifier(learning_rate=0.404, n_estimators=600, objective='binary:logistic',
##                    silent=True, nthread=1)
    model4 = XGBClassifier(silent=False,
                               scale_pos_weight=1,
                               learning_rate=0.404,
                               colsample_bytree = 0.9,
                               subsample = 0.9,
                               objective='binary:logistic',
                               n_estimators=1110,
                               reg_alpha = 0.3,
                               max_depth=7,
                               gamma=10
                               )
    
##    model4 = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
##              colsample_bynode=1, colsample_bytree=1.0, gamma=10, gpu_id=-1,
##              importance_type='gain', interaction_constraints=None,
##              learning_rate=0.404, max_delta_step=0, max_depth=7,
##              min_child_weight=10, monotone_constraints=None,
##              n_estimators=1110, n_jobs=0, num_parallel_tree=1,
##              objective='binary:logistic', random_state=0, reg_alpha=0.3,
##              reg_lambda=1, scale_pos_weight=1, silent=False, subsample=1.0,
##              tree_method=None, validate_parameters=False, verbosity=None)

##    model4 = XGBClassifier()
##    random_search = RandomizedSearchCV(model4, param_distributions=params, n_iter=100, cv = 3, scoring='roc_auc', n_jobs=-1, verbose=2, random_state=42 )
##    start_time = timer(None) # timing starts from this point for "start_time" variable
##    random_search.fit(X_train, y_train)
##    timer(start_time) # timing ends here for "start_time" variable
##    print('\n All results:')
##    print(random_search.cv_results_)
##    print('\n Best estimator:')
##    print(random_search.best_estimator_)
##    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
##    print(random_search.best_score_ * 2 - 1)
##    print('\n Best hyperparameters:')
##    print(random_search.best_params_)
##    results = pd.DataFrame(random_search.cv_results_)
##    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
##    file_path = "./xgb_-random-grid-search-results-01.csv"
##    with open(file_path, mode='w', newline='\n') as f:
##        results.to_csv(f, float_format='%.2f', index=False)

    model4.fit(X_train, y_train)
    y_pred=model4.predict(X_test)
##    print(f1_score(y_test,y_pred))
    
    tripid_test.resize(8576, 1)
    data = np.column_stack([tripid_test, y_pred])
    print(tripid_test)
    label = ["tripid", "prediction"]
    frame = pd.DataFrame(data, columns=label)
    file_path = "./xgb_output.csv"
    with open(file_path, mode='w', newline='\n') as f:
        frame.to_csv(f, float_format='%.2f', index=False, header=True)
        
    # Look at parameters used by our current classifier
    print('Parameters currently in use:\n')
    print(model4.get_params())

def catBoost(X_train,X_test,y_train,y_test,tripid_test):
    print("Catboost")
    ##categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
    model5 = CatBoostClassifier(iterations=55, depth=3, learning_rate=0.407)
    model5.fit(X_train, y_train)
    y_pred=model5.predict(X_test)
    print(f1_score(y_test,y_pred))
    
##    data = np.column_stack([tripid_test, y_pred])
##    label = ["tripid", "prediction"]
##    frame = pd.DataFrame(data, columns=label)
##    file_path = "./catboost_output.csv"
##    with open(file_path, mode='w', newline='\n') as f:
##        frame.to_csv(f, float_format='%.2f', index=False, header=True)
    
    # Look at parameters used by our current forest
    
    print('Parameters currently in use:\n')
    print(model5.get_params())
    
def randomForestModel(X_train,X_test,y_train,y_test,tripid_test):
    print("Random forest")
##    # Number of trees in random forest
##    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
##    # Number of features to consider at every split
##    max_features = ['auto', 'sqrt']
##    # Maximum number of levels in tree
##    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
##    max_depth.append(None)
##    # Minimum number of samples required to split a node
##    min_samples_split = [2, 5, 10]
##    # Minimum number of samples required at each leaf node
##    min_samples_leaf = [1, 2, 4]
##    # Method of selecting samples for training each tree
##    bootstrap = [True, False]
##    # Create the random grid
##    random_grid = {'n_estimators': n_estimators,
##                   'max_features': max_features,
##                   'max_depth': max_depth,
##                   'min_samples_split': min_samples_split,
##                   'min_samples_leaf': min_samples_leaf,
##                   'bootstrap': bootstrap}
    
    model6 = RandomForestClassifier(n_estimators = 1600, min_samples_split = 5, min_samples_leaf = 1, max_features = 'sqrt', max_depth = 70, bootstrap = False)
##    rf_random = RandomizedSearchCV(estimator = model6, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
##    rf_random.fit(X_train, y_train)
##    print(rf_random.best_params_)
    model6.fit(X_train, y_train)
    y_pred=model6.predict(X_test)
    print(f1_score(y_test,y_pred))
    
##    tripid_test.resize(8576, 1)
##    data = np.column_stack([tripid_test, y_pred])
##    #print(f1_score(y_test,y_pred,average='macro'))
##    label = ["tripid", "prediction"]
##    frame = pd.DataFrame(data, columns=label)
##    file_path = "./rf_output.csv"
##    with open(file_path, mode='w', newline='\n') as f:
##        frame.to_csv(f, float_format='%.2f', index=False, header=True)
    
    # Look at parameters used by our current forest

##    print('Parameters currently in use:\n')
##    print(model6.get_params())




#knnClassifier(X_train,X_test,y_train,y_test)
#adaboostClassifier(X_train,X_test,y_train,y_test)
xgboostModel(X,X2,y,y_test,tripid_test)
##xgboostModel(X_train,X_test,y_train,y_test,tripid_test)
##randomForestModel(X_train,X_test,y_train,y_test,tripid_test)
##randomForestModel(X,X2,y,y_test,tripid_test)
#catBoost(X_train,X_test,y_train,y_test,tripid_test)
#catBoost(X,X2,y,y_test,tripid_test)
    
##model = DecisionTreeClassifier()
##model.fit(X_train, y_train)
##y_pred=model.predict(X_test)
##
##print(f1_score(y_test,y_pred,average='macro'))
##print('Parameters currently in use:\n')
##print(model.get_params())


##param_grid = {'n_neighbors':np.arange(1,50)}
##knn = KNeighborsClassifier()
##knn_cv= GridSearchCV(knn,param_grid,cv=5)
##knn_cv.fit(X,y)
##print (knn_cv.best_score_)
##print(knn_cv.best_params_)

##knn = KNeighborsClassifier(n_neighbors = 17)
##knn.fit(X_train,Y_train)
##result = knn.predict(X_test)
##print(Y.shape)
##print (result.shape)

##..................................................stacking
##def get_stacking():
##	# define the base models
##	level0 = list()
##	level0.append(('lr', LogisticRegression()))
####	level0.append(('knn', KNeighborsClassifier(learning_rate=0.404)))
####	level0.append(('cart', DecisionTreeClassifier()))
####	level0.append(('svm', SVC()))
####	level0.append(('bayes', GaussianNB()))
##	level0.append(('rf', RandomForestClassifier()))
##	level0.append(('xgboost',XGBClassifier(silent=False,
##                               scale_pos_weight=1,
##                               learning_rate=0.404,
##                               colsample_bytree = 0.9,
##                               subsample = 0.9,
##                               objective='binary:logistic',
##                               n_estimators=1110,
##                               reg_alpha = 0.3,
##                               max_depth=7,
##                               gamma=10,
##                               reg_lambda=1
##                               )))
####	level0.append(('catboost', CatBoostClassifier(iterations=55, depth=3, learning_rate=0.407)))
##	
##	# define meta learner model
##	level1 = LogisticRegression()
##	# define the stacking ensemble
##	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
##	return model
## 
### get a list of models to evaluate
##def get_models():
##	models = dict()
##	models['lr'] = LogisticRegression()
####	models['knn'] = KNeighborsClassifier()
####	models['cart'] = DecisionTreeClassifier()
####	models['svm'] = SVC()
####	models['bayes'] = GaussianNB()
##	models['rf'] = RandomForestClassifier()
##	models['xgboost'] = XGBClassifier(silent=False,
##                               scale_pos_weight=1,
##                               learning_rate=0.404,
##                               colsample_bytree = 0.9,
##                               subsample = 0.9,
##                               objective='binary:logistic',
##                               n_estimators=1110,
##                               reg_alpha = 0.3,
##                               max_depth=7,
##                               gamma=10
##                               )
####	models['catboost'] = CatBoostClassifier(iterations=55, depth=3, learning_rate=0.407)
##	models['stacking'] = get_stacking()
##	return models
## 
### evaluate a give model using cross-validation
##def evaluate_model(model):
##	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
##	scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
##	return scores
## 
##
### get the models to evaluate
##models = get_models()
### evaluate the models and store results
##results, names = list(), list()
##for name, model in models.items():
##	scores = evaluate_model(model)
##	results.append(scores)
##	names.append(name)
##	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
### plot model performance for comparison
##pyplot.boxplot(results, labels=names, showmeans=True)
##pyplot.show()
