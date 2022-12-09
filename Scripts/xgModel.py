#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb
import pickle
from typing import Dict, Tuple, List
from pandera.typing import DataFrame
from preprocessShots import get_shots, create_dummy_vars
from xgboostParams import ev_params, pp_params, sh_params, en_params
from parameterTuning import optimize_model, space
from evaluateModel import compute_xg


def split_data(shot_data: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, 
                                              DataFrame, DataFrame, DataFrame, List]:
    """
    Split the data into training, validation and test. Also fit and evaluate
    a baseline classifier.

    Parameters
    ----------
    shot_data : DataFrame
        All shots for a given manpower situation.

    Returns
    -------
    X_train : DataFrame
        Design matrix to use for training.
    y_train : DataFrame
        Labels to use for training.
    X_val : DataFrame
        Design matrix to use for validation.
    y_val : DataFrame
        Labels to use for validation.
    X_test : DataFrame
        Design matrix to use for testing.
    y_test : DataFrame
        Labels to use for testing.
    meta_cols : List
        List of all columns that were exlcluded from modeling but still are to be kept.           

    """
    
    # Create all dummy variables for the modeling
    dummy_vars_list = create_dummy_vars(shot_data)
    
    # Define columns to keep but not used for modeling
    meta_cols = ["GameId", "ShooterId", "ScoreDifferential", "ManpowerSituation", "EventNumber"]
    
    # Specify the design matrix
    X = pd.concat([shot_data[["Distance", "Angle", "AngleChangeSpeed",
                              "OffWing", "IsForward", "BehindNet",
                              "X", "Y", "LastX", "LastY", "IsHome", 
                              "SpeedFromLastEvent", "DistFromLastEvent",
                              "TotalElapsedTime"] + meta_cols], 
                   pd.concat(dummy_vars_list[0:], axis=1)],
                   axis=1)
    
    # Create a vector where no goal = 0 and goal = 1
    y = np.array(shot_data.EventType.eq("GOAL").values, dtype=int)
    
    # Training data: 2010-2011 until 2019-2020 seasons
    X_train = X.loc[X.GameId.lt(2020010001)].drop(meta_cols, axis=1)
    y_train = y[X.GameId.lt(2020010001)]
    
    # Validation data: 2020-2021 season
    X_val = X.loc[X.GameId.between(2020010001, 2021010001)].drop(meta_cols, axis=1)
    y_val = y[X.GameId.between(2020010001, 2021010001)]
    
    # Test data: 2021-2022 season
    X_test = X.loc[X.GameId.ge(2021010001)]
    y_test = y[X.GameId.ge(2021010001)]
    
    # Create a dummy classifier to serve as a baseline
    dummy_pred = DummyClassifier(strategy="most_frequent").fit(X_train, y_train).predict_proba(X_test)

    # Evaluate the baseline classifier
    print(f"The baseline log loss is: {log_loss(y_test, dummy_pred[:, 1])}")
    print(f"The baseline AUC is: {roc_auc_score(y_test, dummy_pred[:, 1])}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, meta_cols
    

def fit_model(shot_data: DataFrame, space: Dict, best_params: Dict=None,
              optimize: bool=False, max_evals: int=100) -> Tuple[DataFrame, 
                                                                 xgb.sklearn.XGBClassifier]:
    """
    Train the classifier for predicting the probability of scoring.

    Parameters
    ----------
    shot_data : DataFrame
        All shots for a given manpower situation.
    space : Dict
        The parameter search space.
    best_params : Dict, optional. Default is None.
        If a specific set of parameters should be used instead of optimizing.
    optimize : bool, optional. Default is False.
        If an optimization of the hyperparameters should be performed.
    max_evals : int, optional. Default is 100.
        The maximum number of evaluations to optimize for.

    Raises
    ------
    Exception
        If best_params is None and optimize is False.

    Returns
    -------
    xg_eval : DataFrame
        Meta data and xG values for all shots in the test set.
    xg_clf : xgb.sklearn.XGBClassifier
        An XGboost classifier.

    """
   
    # Get data for the model to train
    X_train, y_train, X_val, y_val, X_test, y_test, meta_cols = split_data(shot_data)
   
    if optimize and best_params is None:
        # Evaluate the parameter space to find the optimal hyperparameters
        best_hyperparams = optimize_model(X_train, y_train, X_val, y_val, space, max_evals)
    elif best_params is not None:
        # Use provided hyperparameters
        best_hyperparams = best_params
    else:
        raise Exception("Either provide parameters or have the model optimize.")
     
    # Specify the classifier and it's parameters
    xg_clf = xgb.XGBClassifier(max_depth=int(best_hyperparams["max_depth"]), 
                               gamma=best_hyperparams["gamma"],
                               subsample=best_hyperparams["subsample"],
                               min_child_weight=int(best_hyperparams["min_child_weight"]),
                               max_delta_step=int(best_hyperparams["max_delta_step"]),
                               learning_rate=best_hyperparams["learning_rate"],
                               reg_alpha=best_hyperparams["alpha"],
                               reg_lambda=best_hyperparams["lambda"],
                               n_estimators=space["n_estimators"],
                               random_state=space["random_state"],
                               seed=space["seed"],
                               use_label_encoder=False, 
                               objective="binary:logistic",
                               eval_metric="logloss")
    
    # Fit the model on the training data
    xg_clf.fit(X_train, y_train)
    
    # Predict the probability of no goal/goal
    y_hat = xg_clf.predict_proba(X_test.drop(meta_cols, axis=1))
    
    # Print the evaluation metrics
    print(f"The log loss is: {log_loss(y_test, y_hat[:, 1])}")
    print(f"The AUC is: {roc_auc_score(y_test, y_hat[:, 1])}")
    
    # Construct a data frame of meta columns and xG predictions
    xg_eval = pd.concat([X_test[meta_cols + ["TotalElapsedTime", "X"]].reset_index(drop=True),
                         pd.Series(y_hat[:, 1], name="xG"), 
                         pd.Series(y_test, name="Goal")], axis=1)
    
    return xg_eval, xg_clf, best_hyperparams


if __name__ == "__main__":
    # Get play by play events and all (fenwick) shots
    pbp, shots = get_shots(end_season=2022)
    
    # All shots during even strength
    ev_shots = shots.loc[shots.ManpowerSituation.isin(["5v5", "4v4", "3v3"]) & 
                         ~shots.EmptyNet.astype(bool)].copy()
    
    # All shots during powerplay
    pp_shots = shots.loc[shots.ManpowerSituation.isin(["5v4", "5v3", "4v3",
                                                       "6v5", "6v4", "6v3"]) & 
                         ~shots.EmptyNet.astype(bool)].copy()
    
    # All shots while short-handed
    sh_shots = shots.loc[shots.ManpowerSituation.isin(["4v5", "3v4", "3v5"]) & 
                         ~shots.EmptyNet.astype(bool)].copy()
    
    # All shots toward an empty net
    en_shots = shots.loc[shots.EmptyNet.astype(bool)].copy()
    
    # Train the models
    ev_xg, ev_xg_mod, ev_params = fit_model(ev_shots, space=space, 
                                            best_params=None, optimize=True, max_evals=250)
    pp_xg, pp_xg_mod, pp_params = fit_model(pp_shots, space=space, 
                                            best_params=None, optimize=True, max_evals=500)
    sh_xg, sh_xg_mod, sh_params = fit_model(sh_shots, space=space, 
                                            best_params=None, optimize=True, max_evals=1000)
    en_xg, en_xg_mod, en_params = fit_model(en_shots, space=space, 
                                            best_params=None, optimize=True, max_evals=1000)
    
    # Save the models
    pickle.dump(ev_xg_mod, open("../Models/ev_xg_mod", "wb"))
    pickle.dump(pp_xg_mod, open("../Models/pp_xg_mod", "wb"))
    pickle.dump(sh_xg_mod, open("../Models/sh_xg_mod", "wb"))
    pickle.dump(en_xg_mod, open("../Models/en_xg_mod", "wb"))
    
    # Compute xG for all manpowers
    xg_name = compute_xg(pbp, ev_xg, pp_xg, sh_xg, en_xg)
