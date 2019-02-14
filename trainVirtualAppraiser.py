"""
BAT 2018-2019

Top-level script for training the Virtual Art Appraiser on Sotheby's/ArtNet datasets.

By the time you run this, you should have scraped and cleaned all of the relevant information,
including image data, and stored as pickled pandas DataFrames.

There is a lot of flexibility in what models are tried, and cross-validation is used for final
model/feature selection.

Dates represent a special case, since they are not consistently known. To deal with this,
two classes of models are constructed: a "no-date" model that fits the full dataset with dates
ignored, and a "dated" model that selectively fits dated paintings including the Date (and any
derived features thereof). The latter must exhibit an improvement in CV score to be kept.

Simpler models are also used as references, to establish performance gains.
"""



###################
from   time import time
import itertools
import pandas as pd
import numpy as np
from   numpy import nan, array
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib.pyplot import plot, scatter, hist, show
import seaborn as sns
from   math import log10, sqrt
from   sklearn import base, preprocessing, dummy, linear_model
from   sklearn.metrics import mean_squared_error
from   sklearn.model_selection import cross_val_score, GridSearchCV
from   sklearn.pipeline import Pipeline
from   sklearn.utils import shuffle
from   sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from   scipy.stats import norm
import random
import datetime
import dill
from   predictiveIntervalRegressor import PredictiveIntervalRegressor
###################
import warnings
from   sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
###################


model_pickle_filename = 'ModelPickles/models.pkd'

input_df_list = []
input_df_list.append(pd.read_pickle('DataFrames/sb_data_cleaned.pkl'))
input_df_list.append(pd.read_pickle('DataFrames/an_data_cleaned.pkl'))

n_bootstrap = 50
n_cv = 10
n_core = 8

## Cutoffs for sizes of each category
## (minor artists may eventually be bundled, and their individual thresholds set lower)
nMin_major_artist = 25
nMin_minor_artist = 10
nMin_nation = 20

## Flags for model classes to try
do_poly_model = True
poly_max_degree = 3
do_tree_model = True
do_boost_model = False

## Write out the best models to file?
output_models = True


############################################
##
##    Data Preparation
##
#############################################


####################
"""
Preliminaries
"""

random.seed(42)

## Matplotlib setup
plt.ion()
matplotlib.rc('font', size=16)
np.set_printoptions(precision=3)

## If there are multiple DataFrames, sew them together
## Chop up by artist name, and pick input DataFrame where that artist has the most entries
## (Assuming here a common schema)
artist_dfIndex_count = {}
for dfIndex, input_df in enumerate(input_df_list):
    for artist, count in input_df.groupby('Artist', sort=False).size().iteritems():
        if artist not in artist_dfIndex_count or count > artist_dfIndex_count[artist][1]:
            artist_dfIndex_count[artist] = (dfIndex, count)
artist_keep_lists = [ set() for _ in range(len(input_df_list)) ]
for artist, (dfIndex, count) in artist_dfIndex_count.items():
    artist_keep_lists[dfIndex].add(artist)
for dfIndex, (input_df, artist_keep) in enumerate(zip(input_df_list, artist_keep_lists)):
    input_df_list[dfIndex] = input_df[input_df['Artist'].isin(artist_keep)]
df = pd.concat(input_df_list)

## Shuffle the data to scramble any patterns from data-taking and avoid issues in cross-validation
df = shuffle(df, random_state=42).reset_index(drop=True)

## Condense/modify column names as needed
df.rename(columns = {'Est Low':'Low'}, inplace=True)
df.rename(columns = {'Est High':'High'}, inplace=True)
df.rename(columns = {'Dating Method':'DatingMethod'}, inplace=True)
df.rename(columns = {'Auction Location':'AuctionLocation'}, inplace=True)
df.rename(columns = {'Catalogue Note':'CatalogueNote'}, inplace=True)

## New/simplified variables
df['LogHeight'] = df['Height'].map(log10)
df['LogWidth'] = df['Width'].map(log10)
df['LogArea'] = df[['Height','Width']].apply(lambda x: log10(x['Height']*x['Width']), axis=1)
df['LogAspect'] = df[['Height','Width']].apply(lambda x: log10(x['Height']/x['Width']), axis=1)
df['PortraitLandscape'] = df['LogAspect'].map(lambda x: 1 if x >=0 else -1 if x < 0 else np.nan)
df['LogAvg'] = df[['Low','High']].apply(lambda x: log10(x['High']*x['Low']) / 2, axis=1)
df['LogErr'] = df[['Low','High']].apply(lambda x: log10(x['High']/x['Low']) / 2, axis=1)
df['AuctionYear'] = df['Auction Date'].map(lambda x: x.year + (x.month-0.5)/12)
df['DateIsUnknown'] = df['DatingMethod'].map(lambda x: int('impute' in x))
df['DateIsSignature'] = df['DatingMethod'].map(lambda x: int(x == 'signature'))
df['DateIsEstimate'] = df['DatingMethod'].map(lambda x: int(x == 'catalogue note'))
df['Sale Price'] = df['Sale Price'].map(lambda x: x if x != 0 else np.nan)  # sometimes "0"(!)
df['Random'] = pd.Series([random.random() for _ in range(len(df))], index=df.index)
df['Signed'] = df['Signed'].map(int)


## Additional basic cleaning
df = df[df['LogAvg'].notnull()]
df = df[df['LogArea'].notnull()]

## *Optional special preselections
#df = df.query("Nationality in ['French']")
#df = df.query("AuctionLocation  == 'London'")
#df = df.query("Artist in ['JEAN-BAPTISTE-CAMILLE COROT', 'JEAN-LÉON GÉRÔME', 'JOAQUÍN SOROLLA', 'JULES BRETON', 'WILLIAM BOUGUEREAU']")
#df = df[ df.Artist.isin( list(df.groupby('Artist').size()
#                              .sort_values(ascending=False).head(10).keys()) ) ]

## Criterion for splitting into test and training sets based on auction year
#test_set_conditions = 'AuctionYear >= 2099'  #  ***All data used for training***
test_set_conditions = 'AuctionYear >= 2016'
#test_set_conditions = "AuctionLocation.str.contains('Sotheby')"
#test_set_conditions = 'Random < 0.25'

## A preliminary version of the training data set, just used to identify training categories
## (Using query engine as 'python' to optionally allow for some pythonic conditionals)
df_train_pre = df.query(f'not ({test_set_conditions})', engine='python')

## Helper function for plotting error bands
def band(x,dx=0):
    return list(x-dx) + list(reversed(x+dx))



###############
"""
Define our categories, upon which we will build dictionaries of independent models.

This includes "major" artists, as well as batches of minor artists grouped by nationality 
(or maybe later some clustering in image space).

Note that minor artists will be indicated within their parent category by a dedicated dummy 
variable, which can optionally be folded into regression fit.

[TECHNICALITY: These category distinctions are formally part of the training, and would
               need to be folded into cross-validation. Try to give yourself big enough
               margins so that this does not turn into a tuning-via-snooping issue.]
"""

## Identify major artists from training data (only personal names, not "FRENCH SCHOOL", etc)
major_artists_series = df_train_pre[~df_train_pre.Artist.str.contains('SCHOOL')].groupby('Artist') \
                           .size().sort_values(ascending=False)
major_artists = major_artists_series[major_artists_series >= nMin_major_artist].index.tolist()

## Identify minor artists, who will be bundled together into parent categories below
minor_artists_series = df_train_pre[~df_train_pre.Artist.isin(major_artists)].groupby('Artist') \
                           .size().sort_values(ascending=False)
minor_artists = minor_artists_series[minor_artists_series >= nMin_minor_artist].index.tolist()

## Identify parent categories for the minor artists
nationalities_series = df_train_pre[df_train_pre.Artist.isin(minor_artists)] \
                           .groupby('Nationality').size().sort_values(ascending=False)
nationalities = nationalities_series[(nationalities_series >= nMin_nation) & \
                                     (nationalities_series.index != 'NA')].index.tolist()
nationalities = [] ####### FOR NOW, IGNORING NATIONALITY CLASSES!

## Create a "Category" column in the original dataframe
out_of_category_label = 'NA'
df['Category'] = \
    df[['Artist','Nationality']] \
    .apply(lambda x: x['Artist'] if x['Artist'] in major_artists else
           x['Nationality'] if x['Artist'] in minor_artists and
           x['Nationality'] in nationalities else
           out_of_category_label, axis=1)
categories = [*df[df.Category != out_of_category_label].groupby('Category').groups]

## Perform the training/testing split, make "global" (union over valid categories) dataframes
df_global_train = df[df.Category != out_of_category_label] \
                    .query('not (' + test_set_conditions + ')', engine='python')
df_global_test  = df[df.Category != out_of_category_label] \
                    .query(test_set_conditions, engine='python')
print(df_global_train.groupby('Category').size().sort_values(ascending=False))
print()

## Recast into category-by-category dataframes
df_train = { category:df_global_train[df_global_train.Category == category]
             for category in categories }
df_test  = { category:df_global_test [df_global_test .Category == category]
             for category in categories }



##########################################
##
##    ML Models  
##
##########################################


################
"""
Generic column selector class to extract named features from either a DataFrame (or Series 
  selected from a single column) or a list of dicts

Includes a 'dummy' option for the columns variable, which returns a single zero-valued feature
"""

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.columns == 'dummy':
            return [[0]]*len(X)
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if isinstance(X, pd.DataFrame):
            X = X[self.columns].to_dict('records')
        return [ [ x[name] for name in self.columns ] for x in X ]


########################
"""
Slightly nontrivial way to choose the "best index" in a CV scan over models.

Nominal cross-validation selects the highest-scoring model. This is dangerous when we
explore a big model space, as CV itself becomes subject to subtle forms of overfitting
via "lucky" fluctuations. To combat this, optionally use the prescription of Tibshirani,
et al, where we "back off" in model complexity by O(1-sigma) in CV score from the minimum.

To work properly, the CV models should be arrayed in order of increasing complexity.
This might require some creative naming of pipeline steps/parameters. (I believe they
are traversed in alphabetical order. Can check GridSearchCV.cv_results_['params'] for the exact
sequence.)

In practice, I found that backing off one full sigma hurt performance, but backing off by a
small fraction of a sigma seems to maintain a reasonable performance while yielding simpler
models. I leave this active for a bit of insurance 8^)

When comparing two models, I choose the smaller of the two CV error bars, so that a simple model
with a huge error bar will be disfavored relative to a more complicated model with nominally
better performance and a small error bar.
"""

def get_cv_best_index(scores, stderrs, n_sigma=1/10):
    best_i = None
    for i, (score, stderr) in enumerate(zip(scores, stderrs)):
        if best_i == None or \
           ( score - scores[best_i] > n_sigma * min(stderr, stderrs[best_i]) ):
            best_i = i
    return best_i


#############
"""
The simplest model: Median of *all* previous appraisals

This is also an elementary test of the pipeline setup and predictive interval wrapper.
"""

# (The selected feature here doesn't matter, use special 'dummy')
pl = Pipeline([
         ('ColumnSelector', ColumnSelectTransformer('dummy')),
         ('Regressor', dummy.DummyRegressor(strategy='median'))
])
r = PredictiveIntervalRegressor(pl, n_resamplings=n_bootstrap, save_models=False,
                                max_residuals=None)
r.fit([[0]]*len(df_global_train), df_global_train['LogAvg'].values)
global_median_model = { category:r for category in categories }


############
"""
Next-to-simplest model: Medians by category
"""

category_median_model = {}
for category in categories:
    pl = Pipeline([
         ('ColumnSelector', ColumnSelectTransformer('dummy')),
         ('Regressor', dummy.DummyRegressor(strategy='median'))
    ])
    r = PredictiveIntervalRegressor(pl, n_resamplings=n_bootstrap, save_models=False,
                                    max_residuals=None)
    r.fit([[0]]*len(df_train[category]), df_train[category]['LogAvg'].values)
    category_median_model[category] = r


############
"""
Simple linear regression over (log) painting area, as another baseline model
"""

area_model = {}
for category in categories:
    pl = Pipeline([
         ('ColumnSelector', ColumnSelectTransformer(['LogArea'])),
         ('Regressor', linear_model.LinearRegression())
    ])
    r = PredictiveIntervalRegressor(pl, n_resamplings=n_bootstrap, save_models=False,
                                    max_residuals=None)
    r.fit(df_train[category][['LogArea']], df_train[category]['LogAvg'].values)
    area_model[category] = r

    
##############
"""
More advanced models

Date painted is optionally added as a feature, forming the dated models. If there are very few dated
paintings or if the CV performance is not improved much, set to None and default back to
nodate model.
"""

## Method to run the dated/nodate cross validations
def cross_validate(do_model=False, model_name='', regr_variables=[], pipeline_factory=None,
                   param_grid={}):
    nodate_model = {category:None for category in categories}
    nodate_cv_score = {category:-np.inf for category in categories}
    nodate_cv_stderr = {category:np.inf for category in categories}
    dated_model = {category:None for category in categories}
    dated_cv_score = {category:-np.inf for category in categories}
    dated_cv_stderr = {category:np.inf for category in categories}
    regr_start_time = time()
    for category in categories * do_model:
        for include_date in [False, True]:
            t_start_cv = time()
            regr_variables_used = regr_variables + include_date*['Date']
            dft = df_train[category] if not include_date else \
                  df_train[category].query('DateIsUnknown == 0')
            if not include_date:
                print(f"No-date {model_name} fit on {category} (N = {len(dft)})")
                # point to nodate dicts
                model = nodate_model
                cv_score = nodate_cv_score
                cv_stderr = nodate_cv_stderr
            else:
                print(f"Dated {model_name} fit on {category} "
                      f"(N = {len(dft)}/{len(df_train[category])})")
                if len(dft) < nMin_major_artist:
                    # too few dated training points for dated model, leave as None
                    print()
                    continue
                # point to dated dicts
                model = dated_model
                cv_score = dated_cv_score
                cv_stderr = dated_cv_stderr
            X = dft[regr_variables_used]
            y = dft['LogAvg'].values
            pl = pipeline_factory(regr_variables_used)
            clf = GridSearchCV(pl,
                               param_grid=param_grid,
                               cv=min(n_cv, len(dft)),
                               scoring='neg_median_absolute_error',
                               n_jobs=n_core,
            )
            clf.fit(X,y)
            scan_scores = clf.cv_results_['mean_test_score']
            scan_stderrs = clf.cv_results_['std_test_score']/sqrt(n_cv)
            print(scan_scores)
            print(' +/- ')
            print(scan_stderrs)
            best_index = get_cv_best_index(scan_scores, scan_stderrs)
            print(f"Best hyperparams: {clf.cv_results_['params'][best_index]}")
            cv_score[category] = scan_scores[best_index]
            cv_stderr[category] = scan_stderrs[best_index]
            if not include_date:
                print(f"Best score: {nodate_cv_score[category]}")
            else:
                print(f"Best score (nodated -> dated): "
                      f"{nodate_cv_score[category]} -> {dated_cv_score[category]}")
            t_end_cv = time()
            print(f"{n_cv}-fold cross-Validation time = {t_end_cv-t_start_cv}")
            if include_date:
                if get_cv_best_index([nodate_cv_score[category], dated_cv_score[category]],
                                     [nodate_cv_stderr[category], dated_cv_stderr[category]]) == 0:
                    # dated model wasn't better than nodate, leave as None
                    print()
                    continue
                else:
                    print("** FOUND IMPROVED MODEL USING DATE")
            pl.set_params(**clf.cv_results_['params'][best_index])
            r = PredictiveIntervalRegressor(pl, n_resamplings=n_bootstrap, save_models=False,
                                            max_residuals=None)
            r.fit(X,y)
            model[category] = r
            print(f"{n_bootstrap}-resampling bootstrap time = {time()-t_end_cv}")
            print()
    print(model_name.upper(), "REGRESSIONS TIME =", time()-regr_start_time)
    print()
    return (nodate_model, nodate_cv_score, nodate_cv_stderr), \
           (dated_model, dated_cv_score, dated_cv_stderr)


regr_variables = ['LogArea', 'LogAspect', 'AuctionYear', 'Signed']
#regr_variables = ['LogArea', 'LogAspect', 'AuctionYear']
#regr_variables = ['LogArea', 'LogAspect']
#regr_variables = ['LogArea']

# a test feature-vector (Not always sensible for all artists...careful!)
tfv = {'LogArea':3.3, 'LogAspect':-0.3, 'Date':1880, 'DateIsUnknown':0, 'AuctionYear':2018,
       'Signed':0}

model_names = []

## Polynomial regression models with lasso regularization
model_names.append('polynomial')
alpha_scan = list(reversed([1e-12, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1, 1e12]))
poly_scan = list(range(1, poly_max_degree + 1))
pipeline_factory = lambda regr_variables_used: \
    Pipeline([
        ('ColumnSelector', ColumnSelectTransformer(regr_variables_used)),
        ('Poly',  preprocessing.PolynomialFeatures(include_bias=False)),
        ('Scaler', preprocessing.RobustScaler()),
        ('Regressor', linear_model.Lasso(alpha=1e-6, max_iter=10000)),
    ])
(nodate_poly_model, nodate_poly_cv_score, nodate_poly_cv_stderr), \
(dated_poly_model, dated_poly_cv_score, dated_poly_cv_stderr) = \
    cross_validate(do_poly_model, model_names[-1], regr_variables, pipeline_factory,
                   {'Regressor__alpha':alpha_scan, 'Poly__degree':poly_scan})

## Random forest models
model_names.append('random forest')
pipeline_factory = lambda regr_variables_used: \
    Pipeline([
        ('ColumnSelector', ColumnSelectTransformer(regr_variables_used)),
        ('Scaler', preprocessing.RobustScaler()),
        ('Tree', RandomForestRegressor(n_estimators=100, criterion='mae', random_state=43)),
    ])
(nodate_tree_model, nodate_tree_cv_score, nodate_tree_cv_stderr), \
(dated_tree_model, dated_tree_cv_score, dated_tree_cv_stderr) = \
    cross_validate(do_tree_model, model_names[-1], regr_variables, pipeline_factory,
                   {'Tree__min_samples_leaf':[30,10,3,1], 'Tree__max_depth':[1,3,10,None]})


## Gradient boosted models (pretty slow, not much performance gain...)
model_names.append('gradient boost')
pipeline_factory = lambda regr_variables_used: \
    Pipeline([
        ('ColumnSelector', ColumnSelectTransformer(regr_variables_used)),
        ('Scaler', preprocessing.RobustScaler()),
        ('Boost', GradientBoostingRegressor(n_estimators=100, criterion='mae', loss='lad',
                                            learning_rate=0.1,
                                            random_state=43)),
    ])
(nodate_boost_model, nodate_boost_cv_score, nodate_boost_cv_stderr), \
(dated_boost_model, dated_boost_cv_score, dated_boost_cv_stderr) = \
    cross_validate(do_boost_model, model_names[-1], regr_variables, pipeline_factory,
                   {'Boost__min_samples_leaf':[30,10,3,1], 'Boost__max_depth':[1,3,10,None]})

## Final model selections
nodate_best_model = {category:None for category in categories}
dated_best_model = {category:None for category in categories}
for category in categories:
    print("Final model selection on", category)
    print("   nodate:")
    # ...sorry, this explicit listing of model types is a bit awkward!
    nodate_models = [nodate_poly_model[category], nodate_tree_model[category],
                     nodate_boost_model[category]]
    nodate_scores = [nodate_poly_cv_score[category], nodate_tree_cv_score[category],
                     nodate_boost_cv_score[category]]
    nodate_stderrs = [nodate_poly_cv_stderr[category], nodate_tree_cv_stderr[category],
                      nodate_boost_cv_stderr[category]]
    print("    ", nodate_scores, "\n     +/-\n    ", nodate_stderrs)
    nodate_best_index = get_cv_best_index(nodate_scores, nodate_stderrs)
    nodate_best_model[category] = nodate_models[nodate_best_index]
    print("     choose ", nodate_best_index, " (", model_names[nodate_best_index], ")", sep='')
    print("   dated:")
    dated_models = [dated_poly_model[category], dated_tree_model[category],
                    dated_boost_model[category]]
    dated_scores = [dated_poly_cv_score[category], dated_tree_cv_score[category],
                    dated_boost_cv_score[category]]
    dated_stderrs = [dated_poly_cv_stderr[category], dated_tree_cv_stderr[category],
                     dated_boost_cv_stderr[category]]
    print("    ", dated_scores, "\n     +/-\n    ", dated_stderrs)
    dated_best_index = -1 + \
                       get_cv_best_index([nodate_scores[nodate_best_index]] + dated_scores,
                                         [nodate_stderrs[nodate_best_index]] + dated_stderrs)
    if dated_best_index >= 0:
        dated_best_model[category] = dated_models[dated_best_index]
        print("     choose ", dated_best_index, " (", model_names[dated_best_index], ")", sep='')
    else:
        print("     no significant improvement with dates")
    print()



###########################
"""
Output the dictionary of models and supporting info as a pickle, for portability to Heroku 
or elsewhere
"""

if output_models:
    dill.settings['recurse'] = True
    model_pack = {}
    for category in categories:
        dfc = df_train[category]
        minDate = dfc[dfc['Category'] == category]['Date'].min()
        maxDate = dfc[dfc['Category'] == category]['Date'].max()
        if np.isnan(minDate):
            minDate = 1800
        if np.isnan(maxDate):
            maxDate = 1900
        model_pack[category] = {'category_median_model':category_median_model[category],
                                'nodate_regr_model':nodate_best_model[category],
                                'dated_regr_model':dated_best_model[category],
                                'minDate':minDate,
                                'maxDate':maxDate,
                                }
    dill.dump(model_pack, open(model_pickle_filename, 'wb'))



##########################################################
##
##   Plots / Analysis
##
##########################################################


########################
"""
Wrap the nodate/dated regression models into a single flexi-regressor that chooses the
correct model on-the-fly based on whether date is present as a feature.

(This is purely to streamline the performance tests below.)
"""

class FlexiRegressor:

    def __init__(self, nodate_regr_model=None, dated_regr_model=None):
        self.nodate_regr_model = nodate_regr_model
        self.dated_regr_model = dated_regr_model

    def predict(self, X):
        return self.run_method(PredictiveIntervalRegressor.predict, X)
        
    def predictive_median(self, X):
        return self.run_method(PredictiveIntervalRegressor.predictive_median, X)

    def predictive_percentiles(self, X, **kwargs):
        return self.run_method(PredictiveIntervalRegressor.predictive_percentiles, X, **kwargs)

    def predictive_half_width(self, X, **kwargs):
        return self.run_method(PredictiveIntervalRegressor.predictive_half_width, X, **kwargs)

    def run_method(self, method, X, **kwargs):
        if self.dated_regr_model == None:
            return method(self.nodate_regr_model, X, **kwargs)
        else:
            iX_nodate = [ (i, feature_vector) for i, feature_vector in
                          enumerate(PredictiveIntervalRegressor.make_iterable(X))
                          if feature_vector['DateIsUnknown'] == 1 ]
            iX_dated  = [ (i, feature_vector) for i, feature_vector in
                          enumerate(PredictiveIntervalRegressor.make_iterable(X))
                          if feature_vector['DateIsUnknown'] != 1 ]
            X_nodate = [ feature_vector for i, feature_vector in iX_nodate ]
            X_dated  = [ feature_vector for i, feature_vector in iX_dated  ]
            if len(X_nodate) == 0:
                return method(self.dated_regr_model, X_dated, **kwargs)
            if len(X_dated) == 0:
                return method(self.nodate_regr_model, X_nodate, **kwargs)
            result_nodate = method(self.nodate_regr_model, X_nodate, **kwargs)
            result_dated   = method(self.dated_regr_model, X_dated, **kwargs)
            result = []
            it_nodate, it_dated = 0, 0
            while it_nodate < len(X_nodate) and it_dated < len(X_dated):
                if iX_nodate[it_nodate][0] < iX_dated[it_dated][0]:
                    result.append(result_nodate[it_nodate])
                    it_nodate += 1
                else:
                    result.append(result_dated[it_dated])
                    it_dated += 1
            while it_nodate < len(X_nodate):
                result.append(result_nodate[it_nodate])
                it_nodate += 1
            while it_dated < len(X_dated):
                result.append(result_dated[it_dated])
                it_dated += 1
            return np.array(result)


test_model = {}
for category in categories:
    test_model[category] = FlexiRegressor(nodate_best_model[category], dated_best_model[category])
    




#########################
"""
Performance against test data, if available
"""

N_artists_test_max = 50 ##len(major_artists)
test_categories = major_artists[:N_artists_test_max]
N_artists_test = len(test_categories)

def get_test_residuals(model, test_cats=test_categories):
    residuals = []
    for category in test_cats:
        if len(df_test[category]) > 0:
            res = df_test[category]['LogAvg'].values - model[category].predict(df_test[category])
            residuals.extend(res)
    return np.array(residuals)

def get_test_abs_err(model, test_cats=test_categories, CL=50):
    abs_residuals = np.abs(get_test_residuals(model, test_cats))
    if len(abs_residuals) == 0:
        return -1
    return np.percentile(abs_residuals, CL)

def get_test_sq_err(model, test_cats=test_categories):
    residuals = get_test_residuals(model, test_cats)**2
    if len(residuals) == 0:
        return -1
    return np.sqrt(residuals.mean())

def get_test_N_in_band(model, test_cats=test_categories, CL=50):
    N_in_band = 0
    for category in test_cats:
        if len(df_test[category]) > 0:
            error_bands = model[category] \
                          .predictive_percentiles(df_test[category], percentiles=(50-CL/2, 50+CL/2))
            logAvgs = df_test[category]['LogAvg'].values
            N_in_band += ((error_bands[:,0] <= logAvgs) * (logAvgs <= error_bands[:,1])).sum()
    return N_in_band

def get_train_avg_half_widths(model, CL=50):
    return np.array(
        [ model[category].predictive_half_width(df_train[category], CL=CL).mean()
          for category in categories ] )

def get_train_appraisal_stats():
    logErr_means = []
    logErr_stds = []
    for category in categories:
        dft = df_train[category]
        appraisalErrs = dft[dft['LogErr'] > 0]['LogErr'].values
        if len(appraisalErrs) > 0:
            logErr_means.append(appraisalErrs.mean())
            logErr_stds.append(appraisalErrs.std())
        else:
            logErr_means.append(0)
            logErr_stds.append(0)
    return np.array(logErr_means), np.array(logErr_stds)


draw_test_plots = True
model_colors = ['C0', 'C2', 'C4', 'C3', 'black', 'C1']
def make_test_plots(hist=False):
    
    N_test = len(get_test_residuals(global_median_model))
    if N_test == 0:
        return
    
    err_global_median_model = 10**get_test_abs_err(global_median_model)
    err_category_median_model = 10**get_test_abs_err(category_median_model)
    err_area_model = 10**get_test_abs_err(area_model)
    err_test_model = 10**get_test_abs_err(test_model)
    err_appraiser_low  = 1.2
    err_appraiser_high = 1.3
    
    ###sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(9,6))
    model_names = ['Historical Avg Model','Avgs by Artist Model', 'Area Model',
                   'Virtual Art Appraiser', '. . .', "Elite Human"]
    model_errs = [err_global_median_model, err_category_median_model, err_area_model,
                  err_test_model, 0, err_appraiser_high]
    plt.bar(x=model_names, height=model_errs, color=model_colors, alpha=0.5)
    plt.bar(x=model_names, height=model_errs[:-1]+[err_appraiser_low], color=model_colors,
            alpha=0.35)
    plt.ylabel('Error Factor', labelpad=25, fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.xticks(rotation=-30, ha='left')
    sns.despine()
    plt.tick_params(axis='x', bottom=False, pad=0)
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0, x1, 1.0, y1))
    plt.subplots_adjust(left=0.15, right=0.9, top=0.88, bottom=0.25)
    loffset = 0.05
    plt.text(0, err_global_median_model+loffset, '{:.3}'.format(err_global_median_model),
             horizontalalignment='center', fontsize=16, fontweight='bold', color=model_colors[0])
    plt.text(1, err_category_median_model+loffset, '{:.3}'.format(err_category_median_model),
             horizontalalignment='center', fontsize=16, fontweight='bold', color=model_colors[1])
    plt.text(2, err_area_model+loffset, '{:.3}'.format(err_area_model),
             horizontalalignment='center', fontsize=16, fontweight='bold', color=model_colors[2])
    plt.text(3, err_test_model+loffset, '{:.3}'.format(err_test_model),
             horizontalalignment='center', fontsize=16, fontweight='bold', color=model_colors[3])
    plt.text(5, err_appraiser_high+loffset,
             '{:.3}~{:.3}'.format(err_appraiser_low, err_appraiser_high),
             horizontalalignment='center', fontsize=16, fontweight='bold', color=model_colors[5])
    title = f"Performance test on recent auctions among top-{N_artists_test} well-known artists" + \
            f"\n({N_test} paintings auctioned in 2016-2018)"
    ax.text(-0.19, 1.18, title, transform=ax.transAxes, fontsize=14, verticalalignment='top',
            color='black', fontweight='bold')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,4))
    plt.tick_params(axis='both', top=True, right=True)
    #sns.distplot(get_test_residuals(global_median_model), hist=hist, color=model_colors[0])
    sns.distplot(get_test_residuals(category_median_model), hist=hist, color=model_colors[1])
    sns.distplot(get_test_residuals(test_model), hist=hist, color=model_colors[3])
    plt.plot([0,0],[0,1], color='black', alpha=0.2)
    plt.show()
    
if draw_test_plots:
    make_test_plots(hist=True)

print()
print("Average test appraisal half-width in LogAvg:")
def print_half_widths(model, test_cats=test_categories):
    for category in test_cats:
        if len(df_test[category]) > 0:
            print(category, model[category].predictive_half_width(df_test[category]).mean())
        else:
            print(category, "[no test paintings]")
print_half_widths(test_model)


## Plots how accurate our CL predictive bands are at actually containing test paintings
## Bands indicate approximate statistical errors on expected containment for finite test samples
##   (binomial, but using gaussian approx)
## Does not indicate any uncertainties associated with the CL-band construction itself, though
##   bear in mind that these also exist!
def containment_plot(model, test_cats=test_categories, CL_list=[50, 60, 70, 80, 90]):
    N_test = len(get_test_residuals(global_median_model, test_cats))
    rate_1sigma_list = [ sqrt(CL * (100-CL) / N_test)  for CL in CL_list ]
    rate_2sigma_list = [ 2*rate_1sigma for rate_1sigma in rate_1sigma_list ]
    rate_in_band_list = [ 100*get_test_N_in_band(model, test_cats, CL)/N_test
                          for CL in CL_list ]
    dRate_in_band_list = [ rate_in_band - CL
                           for rate_in_band, CL in zip(rate_in_band_list, CL_list) ]
    fig, ax = plt.subplots(figsize=(6,4))
    plt.tick_params(axis='both', top=True, right=True)
    plt.fill(band(np.array(CL_list)), band(np.zeros(len(CL_list)), np.array(rate_2sigma_list)),
             c='C0', alpha=0.25)
    plt.fill(band(np.array(CL_list)), band(np.zeros(len(CL_list)), np.array(rate_1sigma_list)),
             c='C0', alpha=0.50)
    plt.plot(CL_list, [0]*len(CL_list), c='C0', linewidth=1)
    plt.plot(CL_list, dRate_in_band_list, c='C3')
    plt.xlabel('Confidence Level (%)', labelpad=10)
    plt.ylabel('Difference from \nNominal Containment (%)', labelpad=10)
    ax.text(0., 1.18, f"{N_test} test paintings", transform=ax.transAxes,
            fontsize=14, verticalalignment='top', color='black', fontweight='bold')
    plt.subplots_adjust(left=0.25, right=0.9, top=0.88, bottom=0.25)
    plt.show()


## Show how predictive interval widths evolve as we refine the models
draw_PI_plot = True
if draw_PI_plot:
    fig, ax = plt.subplots(figsize=(6,4))
    plt.tick_params(axis='both', top=True, right=True)
    rng = list(range(len(categories)))
    #indices = np.argsort([ len(df_train[category]) for category in categories ])
    #indices = np.argsort(get_train_avg_half_widths(test_model))
    indices = np.argsort(get_train_avg_half_widths(category_median_model))
    plt.fill(band(np.array([0, rng[-1]])), band(np.array([1.25, 1.25]), 0.05),
             c='C1', linewidth=0, alpha=0.5)
    plt.plot(rng, 10**get_train_avg_half_widths(test_model)[indices], c='C3', linewidth=2)
    plt.plot(rng, 10**get_train_avg_half_widths(area_model)[indices], c='C4', linewidth=2)
    plt.plot(rng, 10**get_train_avg_half_widths(category_median_model)[indices], c='C2',
             linewidth=2)
    #logErr_means, logErr_stds = get_train_appraisal_stats()
    #plt.plot(rng, 10**(logErr_means[indices]+logErr_stds[indices]), c='black')
    #plt.plot(rng, 10**(logErr_means[indices]-logErr_stds[indices]), c='black')    
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0, x1, 1.0, y1))
    plt.xlabel('Artist (Ranked by median-model predictive width)', labelpad=10)
    plt.ylabel('50% CL Predictive Factor', labelpad=10)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.88, bottom=0.25)
    plt.show()
    

###############
"""
A quick analysis of how "accurate" the Sotheby's appraisers actually are.
Bear in mind the possible effects on final sale price of the "buyer's premium".
"""

buyers_premium = 1.0  # ....actually, this is bigger for lower-ticket items, and changes with time

sales_data = df_global_train[['LogAvg', 'Sale Price']].dropna().values
appraiser_residuals = np.apply_along_axis(lambda x: log10(x[1]/buyers_premium) - x[0],
                                          1, sales_data)
appraiser_accuracy = log10(1.2)
appraiser_naive_CL = \
   len(appraiser_residuals[(appraiser_residuals > -appraiser_accuracy) &
                           (appraiser_residuals <  appraiser_accuracy)]) / \
   len(appraiser_residuals)

## Can also see if there's any particular difference between major auction locations
plot_appraisals = False
appraisal_locations = ['New York', 'London', 'Amsterdam']
appraiser_residuals_by_location = {}
for appraisal_location in appraisal_locations:
    local_sales_data = df_global_train[(df_global_train['AuctionLocation'] == appraisal_location) &
                                       (df_global_train['AuctionYear'] >= 2013)] \
                                       [['LogAvg', 'Sale Price']].dropna().values
    if len(local_sales_data) > 0:
        local_appraiser_residuals = np.apply_along_axis(lambda x: log10(x[1]/buyers_premium) - x[0],
                                                        1, local_sales_data)
        appraiser_residuals_by_location[appraisal_location] = local_appraiser_residuals
        if plot_appraisals:
            sns.distplot(local_appraiser_residuals, hist=False)


## See if the Sotheby's/Christie's buyer's premium effect kicks in at higher/lower prices
## ...It's tricky to deconvolve from up-bidding effects, but we do see that lower appraisals tend to
##    sell for about 40% more than central appraisal on average, whereas higher apprasisals tend to
##    sell for about 20% more. It's not inconsistent with everybody bidding near the central
##    appraisal, and having a graduated premium applied. However, even accounting for this, the
##    quantiles in sale/appraisal ratio should not be drastically different.
plot_premium_check = False
r = linear_model.LinearRegression()
r.fit(np.apply_along_axis(lambda x: [x[0]], 1, sales_data), appraiser_residuals)
if plot_premium_check:
    fig, ax = plt.subplots(figsize=(6,4))
    plt.scatter(sales_data[:,0], appraiser_residuals)
    plt.plot([3,7], r.predict([[3],[7]]), c='red')


