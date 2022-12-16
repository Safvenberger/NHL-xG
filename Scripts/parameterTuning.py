#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from typing import Dict
from pandera.typing import DataFrame


# Define the search space for optimization
space = {"max_depth":        hp.quniform("max_depth", 2, 10, 1),
         "subsample":        hp.quniform("subsample", 0.5, 0.9, 0.05),
         "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
         "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
         "max_delta_step":   hp.quniform("max_delta_step", 0, 10, 1),
         "learning_rate":    hp.quniform("learning_rate", 0.001, 0.5, 0.05),
         "gamma":            hp.qloguniform("gamma",  -10, 5, 1),
         "alpha":            hp.qloguniform("alpha",  -10, 5, 1),
         "lambda":           hp.qloguniform("lambda", -10, 5, 1),
         # Non-tuned components
         "n_estimators": 100, 
         "random_state": 0, 
         "seed": 0}


def objective(space: Dict, X_train: DataFrame, y_train: DataFrame, 
              X_val: DataFrame, y_val: DataFrame) -> Dict:
    """
    Define the objective (minimization) function used for optimization.

    Parameters
    ----------
    space : Dict
        The parameter search space.
    X_train : DataFrame
        Design matrix to use for training.
    y_train : DataFrame
        Labels to use for training.
    X_val : DataFrame
        Design matrix to use for validation.
    y_val : DataFrame
        Labels to use for validation.

    Returns
    -------
    Dict
        Dictionary containing the current loss and the status of optimization.

    """
    
    # Define the classifier and its parameters
    xg_clf = xgb.XGBClassifier(max_depth=int(space["max_depth"]), 
                               gamma=space["gamma"],
                               subsample=space["subsample"],
                               min_child_weight=int(space["min_child_weight"]),
                               max_delta_step=int(space["max_delta_step"]),
                               learning_rate=space["learning_rate"],
                               reg_alpha=space["alpha"],
                               reg_lambda=space["lambda"],
                               n_estimators=space["n_estimators"],
                               random_state=space["random_state"],
                               seed=space["seed"],
                               use_label_encoder=False, 
                               objective="binary:logistic",
                               eval_metric="logloss")
    
    # Fit the classifier on the training data
    xg_clf.fit(X_train, y_train, verbose=False)
    
    # Predict the probability of scoring on the validation set
    y_hat = xg_clf.predict_proba(X_val)
    
    # Compute the log loss of the predictions
    loss = log_loss(y_val, y_hat[:, 1])
    
    return {"loss": loss, "status": STATUS_OK }


def optimize_model(X_train: DataFrame, y_train: DataFrame, 
                   X_val: DataFrame, y_val: DataFrame, 
                   space: Dict, max_evals: int=100) -> Dict:
    """
    Optimize a model with a given objective function and parameter space.

    Parameters
    ----------
    X_train : DataFrame
        Design matrix to use for training.
    y_train : DataFrame
        Labels to use for training.
    X_val : DataFrame
        Design matrix to use for validation.
    y_val : DataFrame
        Labels to use for validation.
    space : Dict
        The parameter search space.
    max_evals : int, optional. Default is 100.
        The maximum number of evaluations to optimize for.

    Returns
    -------
    best_hyperparams : Dict
        Dictionary containing the hyperparameters that minimize the objective function.

    """
    
    # Storage for optimization results
    trials = Trials()

    # Perform function minimization on a given search space and objective function
    best_hyperparams = fmin(fn=lambda x: objective(x, X_train, y_train, X_val, y_val),
                            space=space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials,
                            rstate=np.random.default_rng(0))
   
    return best_hyperparams
