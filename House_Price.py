# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np;
import pandas as pd;

# Data Loading
train = pd.read_csv('train.csv');
test = pd.read_csv('test.csv');

train_ID = train['Id'];
test_ID = test['Id'];
train.drop(['Id'], axis=1, inplace=True);
test.drop(['Id'], axis=1, inplace=True);


print(train.shape);
print(test.shape);

#General Summary About Data

#print(train.head(10));
#print(train.describe().T);
#print(train.info());
#print(train.dtypes.value_counts());




#EDA


import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
import matplotlib.style as style;
import seaborn as sns;
import scipy.stats as stats;



# ######## Correlation & Scatterplot

print((train.corr()**2)['SalePrice'].sort_values(ascending = False));
def customized_scatterplot(y, x):
    style.use('fivethirtyeight');
    plt.subplots(figsize = (12, 8));
    sns.scatterplot(y = y, x = x);

print(customized_scatterplot(train.SalePrice, train.OverallQual));
customized_scatterplot(train.SalePrice, train.GarageArea);
customized_scatterplot(train.SalePrice, train.TotalBsmtSF);
customized_scatterplot(train.SalePrice, train['1stFlrSF']);
customized_scatterplot(train.SalePrice, train.MasVnrArea);


#descriptive statistics summary
train['SalePrice'].describe();

#histogram
print(sns.displot(train['SalePrice']));


#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew());
print("Kurtosis: %f" % train['SalePrice'].kurt());

#scatterplot
sns.set();
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'];
sns.pairplot(train[cols], height = 2.5);
plt.show();

#Get also the QQ-plot
fig = plt.figure();
res = stats.probplot(train['SalePrice'], plot=plt);
plt.show();


# Heatmap

corr = train.corr();
plt.subplots(figsize=(15,12));
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True);

#Boxplot
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1);
f, ax = plt.subplots(figsize=(8, 6));
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data);
fig.axis(ymin=0, ymax=800000);


# Outliers Treatment
print(customized_scatterplot(train.SalePrice, train.GrLivArea));
train = train[train.GrLivArea < 4500];
train.reset_index(drop = True, inplace = True);

# Split features and labels
train_labels = train['SalePrice'].reset_index(drop=True);
train_features = train.drop(['SalePrice'], axis=1);
test_features = test;

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_data = pd.concat([train_features, test_features]).reset_index(drop=True);
all_data.shape;


#Missing Value Analysis

#pip install missingno;
#pip install xgboost;



import missingno as msno;
print(all_data.isna().sum());
print(msno.matrix(all_data));

total = all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0];
percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False)[(all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False) != 0];
missing = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent']);
print(missing);



        # Missing Value Treatment
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()));

missing_val_col = ["Alley", 
                   "PoolQC", 
                   "MiscFeature",
                   "Fence",
                   "FireplaceQu",
                   "GarageType",
                   "GarageFinish",
                   "GarageQual",
                   "GarageCond",
                   'BsmtQual',
                   'BsmtCond',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'BsmtFinType2',
                   'MasVnrType'];

for i in missing_val_col:
    all_data[i] = all_data[i].fillna('None');
    
    
missing_val_col2 = ['BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath', 
                    'BsmtHalfBath', 
                    'GarageYrBlt',
                    'GarageArea',
                    'GarageCars',
                    'MasVnrArea'];

for i in missing_val_col2:
    all_data[i] = all_data[i].fillna(0);
    
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str);
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]));
#The mode function returns 'pandas.Series'. Therefore, we use [0] to extract the element.
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0]);
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0]);
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0]);
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0]);
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0]);
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0]);
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]);

if len(all_data.isnull().sum().sort_values(ascending = False)[all_data.isnull().sum().sort_values(ascending = False) != 0]) == 0:
    print('there is no null');

# Feature Engineering

all_data['TotalSF'] = (all_data['TotalBsmtSF'] 
                       + all_data['1stFlrSF'] 
                       + all_data['2ndFlrSF']);

all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd'];

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] 
                                 + all_data['BsmtFinSF2'] 
                                 + all_data['1stFlrSF'] 
                                 + all_data['2ndFlrSF']
                                );
                                 

all_data['Total_Bathrooms'] = (all_data['FullBath'] 
                               + (0.5 * all_data['HalfBath']) 
                               + all_data['BsmtFullBath'] 
                               + (0.5 * all_data['BsmtHalfBath'])
                              );
                               

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] 
                              + all_data['3SsnPorch'] 
                              + all_data['EnclosedPorch'] 
                              + all_data['ScreenPorch'] 
                              + all_data['WoodDeckSF']
                             );

all_data['hasapool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0);
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0);
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0);
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0);
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0);

drop_features = []
for i in all_data.columns:
    counts = all_data[i].value_counts(ascending = False);
    zeros = counts.iloc[0];
    if zeros / len(all_data) > 0.995:
        print(i);
        drop_features.append(i);
        
all_data = all_data.drop(drop_features, axis = 1);











###########################################
#Encode Catagorical features
all_features = pd.get_dummies(all_data).reset_index(drop=True);

all_data=all_features;

# Split labels
X = all_data.iloc[:len(train_labels), :];
X_test = all_data.iloc[len(train_labels):, :];
X.shape, train_labels.shape, X_test.shape;

"""
# Finding numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'];
numeric = [];
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass;
        else:
            numeric.append(i) ;
            
X = X.select_dtypes(exclude=['object']);
"""
# Modeling

###
#creating matrices for sklearn:


##################################################
from sklearn.linear_model import  Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV;
from sklearn.model_selection import KFold, cross_val_score,RandomizedSearchCV;
from sklearn.metrics import mean_squared_error;
from xgboost import XGBRegressor;
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor;
# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True);

from sklearn import metrics;
# Define error metrics/Validation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred));

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf));
    return (rmse);


#model = XGBRegressor();
#print("Training the XGBRegressor model on the train dataset")
#model.fit(X,X_test);

#rf.fit(X,X_test);

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42);

rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42);


# Train Models
scores = {};
score = cv_rmse(xgboost);
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()));
scores['xgb'] = (score.mean(), score.std());


score = cv_rmse(rf);
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()));
scores['rf'] = (score.mean(), score.std());

#Fit the models/

print('xgboost');
xgb_model_full_data = xgboost.fit(X, train_labels);


print('RandomForest');
rf_model_full_data = rf.fit(X, train_labels);


# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.5 * xgb_model_full_data.predict(X)) + \
            (0.5 * rf_model_full_data.predict(X))) ;


#Predict the Model

# Get final precitions from the blended model
blended_score = rmsle(train_labels, blended_predictions(X));
scores['blended'] = (blended_score, 0);
print('RMSLE score on train data:');
print(blended_score);


# Hyper parameter tuning/Boosting

params= {
        "learning_rate": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 
        "max_depth": [1, 3, 4, 5, 6, 10, 12,  15, 18], 
        "min_child_weight": [1, 3, 5, 7], 
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4], 
        "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        };

random_search = RandomizedSearchCV(xgboost, param_distributions=params, n_iter=5, \
                                      scoring='r2', n_jobs=-1, cv=5, verbose=3)
random_search.fit(X,train_labels);
print('Best parameters for the model are:-');
print(random_search.best_estimator_);
y_hyper_pred = random_search.predict(X);
print(y_hyper_pred);
hyper_mae = metrics.mean_absolute_error(train_labels, y_hyper_pred);
print('MAE after hyper parameter tuning is: ', hyper_mae); 



