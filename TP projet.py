
## ---(Sun Jan 20 19:31:11 2019)---
import pandas as pd
import time
data=pd.read_csv("C:/Users/kid14/PycharmProjects/untitled/venv/anonymized-sq-dataset.tsv", sep='\t', header=0)
data_gp = data.groupby("LABEL")
data_gp.groups

def typicalSampling(group, p):
    return group.sample(frac=p,axis=0)


minidata = data.groupby( 'LABEL', group_keys=False).apply(typicalSampling, p=0.1)

minidata
minidata["LABEL"]
print(minidata.shape)

grp=minidata.groupby("LABEL")
grp.count()
data_gp.count()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(minidata.iloc[:,3:], minidata["LABEL"], random_state=0)

def timer(start_time=None):
    if not start_time:
        start_time = time.clock()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((time.clock() - start_time), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
param_grid = { 'min_child_weight': [2.0, 2.25, 2.5],
               'gamma': [0.001, 0.01, 0.1],
               'eta' : [0.05, 0.075, 0.1],
               'max_depth' : [6,8,10]}
grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: ", grid.best_params_)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
param_grid = { 'knn__n_neighbors': [2,3,4,5,6,7,8,9,10]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train, y_train)
timer(start_time)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.ensemble import RandomForestClassifier
param_grid = { 'n_estimators': [10,20,30,40,50,60,70,80],
                'random_state':[2],
                'max_depth':[5,6,7,8]}
grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.tree import DecisionTreeClassifier
param_grid = { 'max_depth':[1,2,3,4,5],
              'max_features':[7,8,9,10],               
              'random_state':[0] }
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = { 'logisticregression__C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),threshold="median")
start_time = timer(None)
select.fit(X_train, y_train)
timer(start_time)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

%matplotlib inline
import matplotlib.pyplot as plt
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

rfc = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', max_depth=6)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
start_time = timer(None)
boruta_selector.fit(X_train.values, y_train.values)
timer(start_time)

print ("Number of selected features: {} \n " .format(boruta_selector.n_features_))
print (" Feature ranking {} \n :".format(boruta_selector.ranking_))
print ("Selected features {}: \n".format(boruta_selector.support_))

selected=minidata.iloc[:,3:].columns[boruta_selector.support_]
X_train_new=X_train[selected].copy()
X_test_new=X_test[selected].copy()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
param_grid = { 'knn__n_neighbors': [2,3,4,5,6,7,8,9,10]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train_new, y_train)
timer(start_time)
print("Best cross-validation accuracy {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test_new, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.ensemble import RandomForestClassifier
param_grid = { 'n_estimators': [10,20,30,40,50,60,70,80],
                'random_state':[2],
                'max_depth':[5,6,7,8]}
grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train_new, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test_new, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.tree import DecisionTreeClassifier
param_grid = { 'max_depth':[1,2,3,4,5],
              'max_features':[7,8,9,10],               
              'random_state':[0] }
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train_new, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test_new, y_test)))
print("Best parameters: ", grid.best_params_)

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
param_grid = { 'min_child_weight': [2.0, 2.25, 2.5],
               'gamma': [0.001, 0.01, 0.1],
               'eta' : [0.05, 0.075, 0.1],
               'max_depth' : [6,8,10]}

grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train_new, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test_new, y_test)))
print("Best parameters: ", grid.best_params_)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = { 'logisticregression__C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=10)
start_time = timer(None)
grid.fit(X_train_new, y_train)
timer(start_time)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test_new, y_test)))
print("Best parameters: ", grid.best_params_)