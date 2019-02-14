# Virtual Art Appraiser Code/Analysis Samples

This repository contains some of the backend code used to build the [Virtual Art Appraiser](https://virtual-appraiser.herokuapp.com/) demo app, as well as some illustrative analysis results. For a bit more background, see [here](https://virtual-appraiser.herokuapp.com/about).

The app works off of a custom database of historical auctions of 19th-Century European paintings scraped from the websites of the Sotheby's auction houses. For a small sample of well-known artists, we also include aggregated global auction sales compiled from ArtNet.

Each painting put up for auction carries an appraised value, determined by an elite human appraiser. The app attempts to learn how basic features of a painting, such who painted it and its overall dimensions, correlate with its appraised value. (Upcoming versions will exploit features contained in the composition of the image itself, with the aid of Google Vision.) Based on this virtual "apprenticeship," it can then formulate its own appraisals for newly-presented paintings.

The machine learning stage employs models such as polynomial-feature linear regression and random forests. An appropriate model and hyperparameters for each artist are determined using a cross-validation scan. In cases where a given feature may not be reliably available, such as the date that a painting was created, sequences of different models may be created that allow the Virtual Art Appraiser to adapt to the available information.

An important feature of the modeling approach is that, much like an honest human appraiser, the Virtual Art Appraiser reports its own uncertainty. It does so using a kind of hybrid of bootstrapping and cross-validation (see [predictiveIntervalRegressor.py](predictiveIntervalRegressor.py)). If it has an easy job finding a pattern in the historical record, it will report a narrow appraisal range. If a clear pattern turns out to be more elusive, the reported range will be broadened accordingly.

