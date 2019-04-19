
# coding: utf-8

# # Intorduction

# ## What is in this notebook?
# 1. Initial exploration
# 2. Test data set
# 3. Data exploration
# 4. Data cleaning
# 5. Transformation pipeline
# 6. Trying different models
# 7. Fine-tune the model
# 8. Analyze best model
# 9. Evaluate the model

# ## Inputs

# The following are the inputs which the model needs to run, please select one of the below for each input:

# In[1]:

# inputs go here


# ## Magics & Versions

# The below table shows the version of libraries and packages used for running the model.

# In[2]:

# Inline matplotlib
get_ipython().magic('matplotlib inline')

# Interactive matplotlib plot()
#%matplotlib notebook

# Autoreload packages before runs
# https://ipython.org/ipython-doc/dev/config/extensions/autoreload.html
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# %install_ext http://raw.github.com/jrjohansson/version_information/master/version_information.py
# ~/anaconda/bin/pip install version_information
get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')


# ## Standard imports

# In[3]:

# Standard library
import os
import sys
sys.path.append("../src/")

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Date and time
import datetime
import time

# Ipython imports
from IPython.display import FileLink


# ## Other imports

# In[4]:

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

# pandas plots
from pandas.plotting import scatter_matrix


# ## Customization

# In[5]:

# Customizations
sns.set() # matplotlib defaults

# Any tweaks that normally go in .matplotlibrc, etc., should explicitly go here
plt.rcParams['figure.figsize'] = (12, 12)


# In[6]:

# Find the notebook the saved figures came from
fig_prefix = "../figures/2019-04-13-mh-model"


# # Data downloading
# Data was downloaded using the notebook below:

# In[7]:

FileLink('2019-04-10-mh-data-downloading.ipynb')


# # Initial exploration

# In[8]:

# Loading the data
housing = pd.read_csv('../data/training/housing.csv')
housing.head()


# In[9]:

housing.info()


# ## Notes
# 1. The dataset size is 20640, which is relatively small.
# 2. The total_bedrooms has only 20433, which means that there are some missing data which should be taken care of.
# 3. All attributes are numerical except for the ocean_proximity which means it is probably a categorical feature.

# ## Exploring the categorical feature

# In[10]:

housing.ocean_proximity.value_counts()


# ## Checking other fields

# In[11]:

housing.describe()


# In[12]:

# Plotting histograms for numerical features
housing.hist(bins=50)


# ## Notes
# 1. thedata has been scaled and capped at 15 (actually 15.0001) for higher medianincomes, and at 0.5 (actually 0.4999) for lower median incomes.
# 2. The housing median age and the median house value were also capped. The lattermay be a serious problem since it is your target attribute (your labels). The Machine Learning algorithms may learn that prices never go beyond that limit. Check with your client team (the team that will use the system’s output) to see if this is a problem or not. If they tell that they need precise predictions even beyond \$500,000, then you have mainly two options:
#     1. Collect proper labels for the districts whose labels were capped.
#     2. Remove those districts from the training set (and also from the test set, since your system should not be evaluated poorly if it predicts values beyond
# \$500,000).
# 3. The attributes have very different scales. Scalling will be required.
# 4. Finally, many histograms are tail heavy: they extend much farther to the right of
# the median than to the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. We will try transforming these attributes
# later on to have more bell-shaped distributions.

# # Create test set

# ## Data snooping bias
# Creating a test set that early is good to void _*data snooping bias*_ where your brain might recognize patterns in the test set and start overfitting those patterns.

# In[13]:

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) 
print(len(train_set), "train +", len(test_set), "test")


# ## Stratified sampling

# The median income is a very important feature for predicting the median housing price.
# Thus, we will need statified sampling for the train test split.

# Looking at the median_income histogram, most values are clustered around 2-5 but some go far beyond 6.
# 
# The following code creates an income category attribute by dividing the median income by 1.5 (to limit the number of income categories), and rounding up using ceil (to have discrete categories), and then merging all the categories greater than 5 into category 5:

# In[14]:

# creating income_cat column
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)


# In[15]:

# creating the split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[16]:

# removing the income_cat folumn
for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis=1, inplace=True)


# # Data exploration

# In[17]:

# making a copy of the stratified training data
housing = strat_train_set.copy()


# ## Data visualization

# ### Geographical data

# In[18]:

# plotting latitude and longitude
housing.plot(kind='scatter', x='longitude', y='latitude')


# In[19]:

# changing alpha to look at areas of high density
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)


# In[20]:

# adding more visualization
# size -> poplulation
# color -> price
housing.plot(kind='scatter',
             x='longitude', 
             y='latitude',
             s=housing.population/100,
             label='population',
             c='median_house_value',
             cmap=plt.get_cmap('jet'),
             colorbar=True)
plt.legend()

# Saving the plot
plt.savefig(fig_prefix + "latitude-longitude-median-value.png", dpi=350)


# ## Looking for correlations

# In[21]:

# creating a correlation matrix 
corr_matrix = housing.corr()
corr_matrix.median_house_value.sort_values(ascending=False)


# In[22]:

# plotting correlations
# using pandas scatter matrix
attributes = ['median_house_value', 'median_income',
              'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes],  figsize=(12, 8))


# In[23]:

# further investigation for the median income vs the median house value
housing.plot(kind='scatter', x='median_income', y='median_house_value',
             alpha=0.1)


# ### Notes
# 1. There is a strong correlation between the median house price and the median income.
# 2. There is a clear horizontal line at \$500,000 because of the cap.
# 3. There is also few other horizonral lines at \$450,000 \$350,000 \$280,000 and few more. We may need to remove these districts to prevent the algorithm from learning to reproduce these data quirks.

# ## Attribute combinations

# In[24]:

# extracting new attributes 
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] =  housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


# In[25]:

# checking correlations
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# The new bedrooms_per_room attribute is much more correlated with
# the median house value than the total number of rooms or bedrooms. Apparently houses with a lower bedroom/room ratio tend to be more expensive. The number of rooms per household is also more informative than the total number of rooms in a district—obviously the larger the houses, the more expensive they are.

# # Data cleaning

# In[26]:

# splitting the data to predictors and targets
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# ## Getting rid of NaN

# ### Manually

# There are 3 options:
# 1. Get rid of the corresponding districts. (row)
# 2. Get rid of the whole attribute. (column)
# 3. Set the values to some value (zero, mean, median, most common, etc).

# option 1
# ```python
# housing.dropna(subset=['total_bedrooms'])
# ```
# 
# option 2
# ```python
# housing.drop('total_bedrooms', axis=1)
# ```
# option 3
# ```python
# median = housing['total_bedrooms'].median()
# housing['total_bedrooms'].fillna(median)
# ```

# The median should be stored because the same value will be used for the test set and all new data as well.

# ### Using sklearn imputer

# In[27]:

# initiating an imputer
imputer = Imputer(strategy='median')

# removing categorical fetures
housing_num = housing.drop('ocean_proximity', axis=1)

# fitting the imputer
imputer.fit(housing_num)

# Checing the imputer
imputer.statistics_


# In[28]:

# transforming the numerical features
X = imputer.transform(housing_num)

# putting the data in a dataframe
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# ## Handling text and categorical attributes

# In[29]:

# encoding categorical features
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[30]:

# using the one hot encoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot


# In[31]:

# using the Label Binarizer instead
encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot


# ## Custom transformers

# Although Scikit-Learn provides many useful transformers, you will need to write
# your own for tasks such as custom cleanup operations or combining specific
# attributes. You will want your transformer to work seamlessly with Scikit-Learn functionalities
# (such as pipelines), and since Scikit-Learn relies on duck typing (not inheritance),
# all you need is to create a class and implement three methods: fit()
# (returning self), transform(), and fit_transform(). You can get the last one for
# free by simply adding TransformerMixin as a base class. Also, if you add BaseEstima
# tor as a base class (and avoid *args and **kargs in your constructor) you will get
# two extra methods (get_params() and set_params()) that will be useful for automatic
# hyperparameter tuning. For example, here is a small transformer class that adds
# the combined attributes we discussed earlier:

# In[32]:

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, 
                         population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[33]:

# trying the previous class
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attributes = attr_adder.transform(housing.values)
housing_extra_attributes


# # Transformation piplines 

# In[34]:

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[35]:

# Getting lists of numerical and categorical features
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

# Numerical attributes pipeline
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# Categorical attributes pipeline
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

# Full pipeline
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


# In[36]:

# Running the whole pipeline
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# # Trying different models

# For a list of strings that can be passed to the scoring parameter:
# 
# https://scikit-learn.org/stable/modules/model_evaluation.html

# In[37]:

# A function to display the scores
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


# ## Linear regression

# In[38]:

lin_reg = LinearRegression()
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# ## Decision tree

# In[39]:

tree_reg = DecisionTreeRegressor()
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


# ## Random Forest

# In[40]:

forest_reg = RandomForestRegressor()
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# ## Pickling the models

# ```python
# from sklearn.externals import joblib
# joblib.dump(my_model, 'my_model.pkl')
# # and later...
# my_model = joblib.load('my_model.pkl')
# ```

# # Fine-tune the model

# The random forest model seems to be giving the best results. So, I will try to fine-tune that model.

# ## Grid Search

# In[41]:

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)


# In[42]:

# Best params
grid_search.best_params_


# __TODO:__ Try higher n_estimators since 30 is the best and 30 was the last in that list. 

# In[43]:

# Best estimator
grid_search.best_estimator_


# In[44]:

# Evaluation scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ## Randomized Search
# to be tested.

# # Analyze best model

# In[45]:

# Feature importances
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[46]:

# Display attributes and their importances
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# Based on this information, we may need to drop som eless useful features such as 'NEAR OCEAN' and 'NEAR BA'.
# 
# We should also look at the specific errors that your system makes, then try to understand
# why it makes them and what could fix the problem (adding extra features or, on
# the contrary, getting rid of uninformative ones, cleaning up outliers, etc.).

# # Evaluating the model

# In[47]:

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse

