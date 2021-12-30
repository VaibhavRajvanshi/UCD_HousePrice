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


#Missing Value Analysis

#pip install missingno;
import missingno as msno;
print(train.isna().sum());
print(msno.matrix(train));

total = train.isnull().sum().sort_values(ascending = False)[train.isnull().sum().sort_values(ascending = False) != 0];
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending = False)[(train.isnull().sum() / train.isnull().count()).sort_values(ascending = False) != 0];
missing = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent']);
print(missing);

all_data = train;


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








# Outliers Treatment
print(customized_scatterplot(all_data.SalePrice, all_data.GrLivArea));
all_data_ = all_data[all_data.GrLivArea < 4500];
all_data_.reset_index(drop = True, inplace = True);
previous_train = all_data_.copy();
customized_scatterplot(all_data_.SalePrice, all_data_.GrLivArea);





# Modeling


#creating matrices for sklearn:
X_train = all_data_[:train.shape[0]];
X_test = all_data_[train.shape[0]:];
y = train.SalePrice;


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV;
from sklearn.model_selection import cross_val_score;


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5));
    return(rmse);

model_ridge = Ridge();


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75];
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas];

