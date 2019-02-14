"""
BAT, October 2018

A wrapper that endows a generic regression model with approximate predictive intervals/quantiles.

Written for establishing appraisal ranges for artworks, with robustness to possible heavy tails.

There is no universal method for establishing predictive intervals, and the problem becomes
especially tricky for new data points that are outside of the training range, or "just at the
edge" as in using historical records to predict present-day behavior. Here we will be satisfied
with an empirically informed hack. This should be sufficient, especially because we are not going 
for very high confidence levels.

The wrapper was original written using bootstrapped models to get the variability of the underlying
fit at each x-point, convolved with the global fit residuals from each data resampling. This worked 
okay for linear models, but gave over-confident predictive intervals for nonlinear models (like 
polynomials or random forests), as measured using holdout test data. It was also really slow, and
the saved ensembles of models were disk/memory intensive.

Currently, the approach is analogous to cross-validation. Bootstrap resamples are drawn and refit
as usual, and residuals are assessed by doing an independent resampling on the out-of-bag points.
The ensembles of out-of-bag residuals are concatenated and used as a proxy for a global
error distribution. This distribution gets centered on each nominal model prediction to
define the predictive distributions. No attempt is made to deduce predictive distribution shapes
point-by-point, especially since x-space may not be densely populated enough to make an informed 
x-dependent distribution. In practice, the resulting bands will be correct "on average". Indeed,
the CL-band containment on holdout test data sets (either randomly-sampled or just recent auctions)
tends to look reasonable.

(The approach can also be adapted to do literal cross-validation instead, concatenating the
residuals from all of the K-folds. In practice, though, that tends to give over-confident
predictive intervals when compared to holdout data consisting of recent paintings, even when using
distinct K-fold partitions from what were used in the model selection steps. I'm not totally 
sure why this is, but I can speculate. For example, suppose that for a very flexible model like a 
random forest, extrapolating up to future behavior might get dominated by the most recent paintings
in the training set. If we do literal K-folding, then a particular recent point will get to 
dominate the fitted future behavior in 1 - 1/K fraction of the samples, and itself only acts as a 
residual in 1/K samples. With the bootstrap approach, by contrast, that point will contribute to 
the fit in ~2/3 of the resamplings, and act as a residual in the others. So fluctuating predictions 
near the boundaries might get more robustly picked up.)

...

As with usual scikit-learn estimators, many of these methods anticipate being fed a list of 
feature-vectors "X". However, if you wrap around a pipeline with selectors/transformers, X may be
a Pandas DataFrame or a list of feature dictionaries, and is treated accordingly below.

For some of the methods, to avoid a multiplication of predictive values in memory, you can only
feed in one feature-vector/dictionary/Series at a time.
"""

##############################
import numpy as np
import pandas as pd
from   random import Random
from   sklearn import base
from   sklearn.utils import resample
from   sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from   sklearn.base import clone
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib.pyplot import plot, scatter, hist, bar
import seaborn as sns
###############################


class PredictiveIntervalRegressor(base.BaseEstimator, base.RegressorMixin):
 
    def __init__(self, model, n_resamplings=100, auto=True, max_residuals=None, random_state=42,
                 save_models=True):
        self.model = model
        self.n_resamplings = n_resamplings
        self.auto = auto
        self.max_residuals = max_residuals
        self.save_models = save_models
        #
        self.did_fit_ = False
        self.did_bootstrap_ = False
        self.rnd_ = Random(random_state)
        #
        self.model.copy_X = False
        
    def fit(self, X, y):
        """
        Fit the internal model, with the option to automatically bootstrap
        (Note that X here could be a dataframe or list of dicts or whatever, however y should
          be a simple list or 1D array)
        If fed a list of dicts or DataFrame, convert to a numpy array of dicts for simple slice-
          based resampling
        """
        if isinstance(X, pd.Series):
            self.X_ = X.to_frame().T
        if isinstance(X, pd.DataFrame):
            self.X_ = np.array(pd.DataFrame(X).to_dict(orient='records'))
        else:
            self.X_ = np.array(X)
        self.y_ = np.array(y)
        self.model.fit(X, y)
        self.did_fit_ = True
        if self.auto:
            self.bootstrap()
        return self

    def predict(self, X):
        if not self.did_fit_:
            print(" -- WARNING: ATTEMPTED TO PREDICT BEFORE FITTING THE MODEL")
            return
        if self.did_bootstrap_:
            return self.predictive_median(X)
        else:
            return self.model.predict(X)

    def nominal_predict(self, X):
        if not self.did_fit_:
            print(" -- WARNING: ATTEMPTED TO PREDICT BEFORE FITTING THE MODEL")
            return
        return self.model.predict(X)
        
    def bootstrap(self):
        """
        Builds the set of bootstrap-resamples and refits models
        These can then be used to build up statistics with other routines below
        Be sure to use a large enough value of n_resamplings for your needs!
        Residuals are constructed from resampled out-of-bag points (~1/3 of data, upsampled to
          fixed 2/3), to decouple from any overfitting
        Note: After bootstrapping, the original (X_,y_) data is thrown away
        """
        if not self.did_fit_:
            print(" -- WARNING: ATTEMPTED TO BOOTSTRAP BEFORE FITTING THE MODEL")
            return
        self.resampled_models_ = []
        self.resampled_residuals_ = []
        nX = len(self.X_)
        iResampling = 0
        while iResampling < self.n_resamplings:
            resampled_indices = [ self.rnd_.randint(0,nX-1) for ix in range(nX) ]
            resampled_indices_set = set(resampled_indices)
            oob_indices = [ ix for ix in range(nX) if ix not in resampled_indices_set ]
            if len(oob_indices) < 2:
                # make sure we have *some* oob points from which to form residuals
                continue
            else:
                iResampling += 1
            resampled_oob_indices = [ self.rnd_.choice(oob_indices)
                                      for ix in range(int(2/3 * nX)) ]
            bs_model = clone(self.model)
            bs_model.fit(self.X_[resampled_indices], self.y_[resampled_indices])
            if self.save_models:
                self.resampled_models_.append(bs_model)
            residuals = self.y_[resampled_oob_indices] - \
                        bs_model.predict(self.X_[resampled_oob_indices])
            self.resampled_residuals_.append(residuals)
        self.concatenated_residuals_ = np.concatenate(self.resampled_residuals_)
        if isinstance(self.max_residuals, int):
            self.concatenated_residuals_ = \
                np.array(self.rnd_.sample(list(self.concatenated_residuals_), self.max_residuals))
        self.concatenated_residuals_ -= np.median(self.concatenated_residuals_)
        self.did_bootstrap = True
        self.X_ = None
        self.y_ = None
        self.resampled_residuals_ = []
        
    def MLE_distribution(self, X_test):
        """
        MLE here refers to just the "best guess" prediction from each resampled/refit model,
          *without* accounting for the additional noise on the prediction from the residuals
          distribution
        """
        return np.swapaxes(
                 np.array([ bs_model.predict(X_test) for bs_model in self.resampled_models_ ]),
                 0, 1)
    
    def MLE_percentiles(self, X_test, percentiles=(25,50,75,)):
        if np.isscalar(percentiles):
            return np.percentile(self.MLE_distribution(X_test), percentiles, axis=1)
        else:
            return np.swapaxes(
                     np.percentile(self.MLE_distribution(X_test), percentiles, axis=1),
                     0, 1)

    def MLE_median(self, X_test):
        return self.MLE_percentiles(X_test, 50)

    def predictive_ensemble(self, feature_vector):
        """
        Returns the full ensemble of y-predictions for a single feature-vector, 
        (Does *not* work on full lists of feature-vectors, to avoid possible explosion of working
          array sizes. It is always to be iterated over for full data sets.)
        """
        return self.nominal_predict([feature_vector])[0] + self.concatenated_residuals_
    
    def predictive_percentiles(self, X_test, percentiles=(25,50,75,)):
        """
        Percentiles for predictive distributions
        """
        predictions = self.nominal_predict(X_test)
        qs = np.percentile(self.concatenated_residuals_, percentiles)
        return np.swapaxes(np.array( [predictions + q for q in qs] ), 0, 1)
     
    def predictive_median(self, X_test):
        return self.predictive_percentiles(X_test, 50) 

    def predictive_half_width(self, X_test, CL=50):
        percentiles = self.predictive_percentiles(X_test, (50-CL/2, 50+CL/2))
        return np.apply_along_axis(lambda x: (x[1]-x[0])/2, -1, percentiles)
    
    def predictive_histogram(self, X_test, nbins=100):
        """
        Returns a list of  numpy histograms, each of which is a 2-tuple made of a len(X_test) 
          array of bin counts and a len(X_test)+1 array of bin edges.
        To plot this in matplotlib, see predictive_plot below
        """
        return [ np.histogram(self.predictive_ensemble(feature_vector), bins=nbins)
                 for feature_vector in self.make_iterable(X_test) ]

    def predictive_plot(self, feature_vector, nbins=100, size=(6,4)):
        frq, edges = self.predictive_histogram([feature_vector], nbins)[0]
        fig, ax = plt.subplots(figsize=size)
        bar(edges[:-1], frq, align="edge", width=np.diff(edges))
        plt.show()
        return fig, ax

    def predictive_kdeplot(self, feature_vector, size=(6,4), **kwargs):
        """
        Prettified Seaborn KDE plot of the predictive distribution
        """
        fig, ax = plt.subplots(figsize=size)
        sns.distplot(self.predictive_ensemble(feature_vector), **kwargs)
        return fig, ax
        
    @staticmethod
    def make_iterable(X_test):
        """
        Several of methods above may assume that X is broken up into a list or array so that 
          individual feature vectors can be iterated over (as an intentional bottleneck on 
          point-by-point generation of residuals ensembles to avoid possible memory overflows)
        If we feed in a DataFrame, we can't iterate over its rows like a simple list without this
          extra layer
        """
        if isinstance(X_test,  pd.DataFrame):
            for _, row in X_test.iterrows():
                yield row
        else:
            for x in X_test:
                yield x
        
    
if __name__ == "__main__":
    # execute only if run as a script
    main()
