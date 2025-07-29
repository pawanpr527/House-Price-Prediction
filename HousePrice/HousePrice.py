import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('HousePrice/train.csv')
test_data = pd.read_csv('HousePrice/test.csv')
id = test_data['Id']
#target value
y = train_data['SalePrice']

train_data = train_data.drop('SalePrice',axis=1)
combined = pd.concat([train_data,test_data],ignore_index=True)

#UnderStand dataset

# print(train_data.shape)
# print(combined.shape)
# print(combined.isnull().sum().sum())

# Data Cleaning And Preproccessing
temp = combined.isnull().sum()
# print(temp[temp>0]) #it shows which feature have null values


#shows outliers
# sns.boxplot(data=combined,x='LotFrontage')
# plt.show()

#distribution  shows right skewd 
# sns.histplot(combined['LotFrontage'].dropna(),bins=50,kde=True)
# plt.show()

# print(combined[combined['Neighborhood']=='NAmes']['LotFrontage'].mean()) #most LotFrontage values are lies b/w 60-85 which Neighborhood is NAmes

#There are 34 columns which have null values so create a function
#first understand which columns have categorical data or numerical data
# print(combined['Exterior1st'].value_counts())


# for i in list(temp.index):
#     if combined[i]=='NA':

def FillNull(raw_data):
    #fill with NA replace 'None' for categorical
    col1 = ['Alley','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 
                 'MasVnrType']
    for col in col1:
        raw_data[col] = raw_data[col].fillna('None')

    #fill with NA replace 0 for numerical
    col2 = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
                 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 
                 'MasVnrArea']
    for col in col2:
        raw_data[col] = raw_data[col].fillna(0)

    #fill with mode
    col3 = ['MSZoning', 'KitchenQual', 'Functional', 'Electrical', 
                 'SaleType', 'Exterior1st', 'Exterior2nd', 'Utilities']
    for col in col3:
        raw_data[col] = raw_data[col].fillna(raw_data[col].mode()[0])
    #fill LotFrontage using Neighborhood median
    raw_data['LotFrontage'] = raw_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    return raw_data

combined = FillNull(combined)
# temp2 = combined.isnull().sum().sum()

# print(combined.shape)
# print(combined.isnull().sum().sum())

#data is cleaned

#Feature selection
train_data = combined.iloc[:1460].copy()
test_data = combined[1460:].copy()
train_data.loc[:,'SalePrice'] = y

#target value 
target = train_data['SalePrice']


#distribution of SalePrice is Right skewed

# sns.histplot(data=train_data,x='SalePrice',kde=True)
# plt.show()

# Log-transform SalePrice to improve linearity
# If SalePrice is skewed, apply:

# train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

# sns.histplot(data=train_data,x='SalePrice',kde=True)
# plt.show()  #now data is centerd

#  linear Regression


#check linearity with target

corr_matrix = train_data.corr(numeric_only=True)
target_corr = corr_matrix['SalePrice'].sort_values(ascending=False).drop('SalePrice')
strong_correlation = target_corr[target_corr.abs()>0.5]
strong_linear_feature = list(strong_correlation.index)
# print(len(strong_correlation))  #10 feature is highly correlated with sale prince
# print(strong_linear_feature)

# categorical feature

catagorical_feature = train_data.select_dtypes(include=['object','category']).columns.tolist()
encoded = pd.get_dummies(train_data[catagorical_feature],drop_first=True)
encoded['SalePrice'] = target

cat_corr = encoded.corr()
target_encoded = cat_corr['SalePrice'].sort_values(ascending=False).drop('SalePrice')
strong_cat = list(target_encoded[target_encoded.abs()>0.3].index)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error

# Features
# strong_linear_feature = ['GrLivArea', 'OverallQual', 'GarageCars', 'GarageArea']
# strong_cat = ['Neighborhood', 'ExterQual']  # use original names, not encoded

# X = train_data[strong_linear_feature + strong_cat]
# y = train_data['SalePrice']

# Preprocessor
# preprocessor = ColumnTransformer(transformers=[
#     ('num', StandardScaler(), strong_linear_feature),
#     ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), strong_cat)
# ])

# # Pipeline
# pipeline = Pipeline(steps=[
#     ('preprocess', preprocessor),
#     ('model', LinearRegression())
# ])

# Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train
# pipeline.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = pipeline.predict(X_test)
from math import sqrt
# rmse = sqrt(mean_squared_error(y_test, y_pred))

# print(f"Linear Regression RMSE: {rmse:.2f}")

# from sklearn.linear_model import Ridge

# pipeline.steps[-1] = ('model', Ridge(alpha=10))
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# print(f"Ridge Regression RMSE: {rmse:.2f}")

# from sklearn.linear_model import Lasso

# pipeline.steps[-1] = ('model', Lasso(alpha=0.1))
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# print(f"Lasso Regression RMSE: {rmse:.2f}")

# from sklearn.linear_model import LassoCV

# pipeline.steps[-1] = ('model', LassoCV(cv=5))
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# rmse = sqrt(mean_squared_error(y_test, y_pred))
# print(f"LassoCV RMSE: {rmse:.2f}")
# print(f"Best alpha: {pipeline.named_steps['model'].alpha_}")


import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Select features
features = ['GrLivArea', 'OverallQual', 'GarageCars', 'GarageArea', 'Neighborhood', 'ExterQual']
X = train_data[features]
test_x = test_data[features]
y = train_data['SalePrice']

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)
test_x = pd.get_dummies(test_x,drop_first=True)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(test_x)

submission = pd.DataFrame({
    'Id' :id,
    'SalePrice' : y_pred
})

submission.to_csv('submission.csv',index=False)
