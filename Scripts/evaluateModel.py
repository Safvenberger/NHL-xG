#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
from pandera.typing import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_xg(pbp: DataFrame, ev_xg: DataFrame, pp_xg: DataFrame, 
               sh_xg: DataFrame, en_xg: DataFrame) -> DataFrame:
    """
    Compute the expected goals of all shots in different manpower situations.

    Parameters
    ----------
    pbp : DataFrame
        Data frame of all play by play events from all seasons.
    ev_xg : DataFrame
        Expected goals of all shots in even strength.
    pp_xg : DataFrame
        Expected goals of all shots in powerplay.
    sh_xg : DataFrame
        Expected goals of all shots while short-handed.
    en_xg : DataFrame
        Expected goals of all shots toward an empty net.

    Returns
    -------
    xg_name : DataFrame
        Total expected goals with player name added.

    """
    
    # Read player data
    players = pd.read_csv("../Data/players.csv", low_memory=False)[["id", "fullName"]]
    
    # Compute total goals across all situations for the season
    goals = pd.concat([ev_xg, pp_xg, sh_xg, en_xg]).groupby("ShooterId", as_index=False)["Goal"].sum(
        ).rename(columns={"Goal": "Goals"}).sort_values(by="Goals", ascending=False)
    
    # Compute total xG across all situations for the season
    xg = pd.concat([ev_xg, pp_xg, sh_xg, en_xg]).groupby("ShooterId", as_index=False)["xG"].sum(
        ).sort_values(by="xG", ascending=False)
    
    # Create a data frame of xG across each situation
    situation = pd.concat([ev_xg.assign(Situation="EV"), pp_xg.assign(Situation="PP"), 
                           sh_xg.assign(Situation="SH"), en_xg.assign(Situation="EN")]
                          )
    
    # Compute the number of goals per situation
    goals_per_situation = situation.groupby(["Situation", "ShooterId"], as_index=False)["Goal"].sum(
        ).rename(columns={"Goal": "Goals"}).sort_values(by="Goals", ascending=False)
    
    # Compute xG per situation
    xg_per_situation = situation.groupby(["Situation", "ShooterId"], as_index=False)["xG"].sum(
        ).sort_values(by="xG", ascending=False)
    
    # Combine total and situation goals into one
    goals_situational = goals.merge(goals_per_situation, on="ShooterId",
                                                  suffixes=("", "_Situational")).pivot(
        index=["ShooterId", "Goals"], columns="Situation", values="Goals_Situational").fillna(0).reset_index()
    
    # Combine total and situational xg into one
    xg_situational = xg.merge(xg_per_situation, on="ShooterId",
                                            suffixes=("", "_Situational")).pivot(
        index=["ShooterId", "xG"], columns="Situation", values="xG_Situational").fillna(0).reset_index()
    
    # Combine goals and xG
    situational = goals_situational.merge(xg_situational, on="ShooterId", 
                                          suffixes=("_Goals", "_xG"))
                                                
    # Reorder columns
    situational = situational[["ShooterId", "Goals", "xG", "EV_Goals", "EV_xG",
                               "PP_Goals", "PP_xG", "SH_Goals", "SH_xG", "EN_Goals", "EN_xG"]]
    
    # Add player names
    xg_name = players.merge(situational, left_on="id", right_on="ShooterId").drop(
        "ShooterId", axis=1).rename(columns={"id": "PlayerId"}).sort_values("xG", ascending=False)
    
    # Save as csv file
    xg_name.to_csv("../Data/xg.csv", index=False)
    
    # Combine all shots
    xg_shots = pd.concat([ev_xg.assign(Situation="EV"), pp_xg.assign(Situation="PP"), 
                          sh_xg.assign(Situation="SH"), en_xg.assign(Situation="EN")])
    
    # Save as csv file
    xg_shots.to_csv("../Data/xg_shots.csv", index=False)
    
    # Combine xG with play by play
    xg_pbp = pbp.loc[pbp.GameId.astype(str).str.startswith("2021")].merge(xg_shots.drop(
        ["ScoreDifferential", "ManpowerSituation"], axis=1).rename(columns={"X": "X_adj"}), how="left")
    
    # Save as csv file
    xg_pbp.to_csv("../Data/xg_pbp.csv", index=False)
    
    return xg_name


def plot_feature_importance(model, shot_data: DataFrame, suffix: str, split_data):
    """
    Visualize feature importance for a given model.

    Parameters
    ----------
    model : xgb.sklearn.XGBClassifier
        The XGBoost model.
    shot_data : DataFrame
        Data for the model. 
    suffix : str
        The suffix of the figure.
    split_data : function
        Function as defined in xgModel.py

    Returns
    -------
    None. Instead a plot is returned and saved to disk.

    """
    # Get training data
    X_train, _, _, _, _, _, _ = split_data(shot_data)
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 8))       
    
    # Remove padding
    ax.margins(y=0.01)
    
    # Get the importance of the features, sorted from largest to smallest
    sorted_idx = model.feature_importances_.argsort()
    
    # Plot the feature importance
    ax.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx], 
            color="#FF9913")
          
    # Change plot spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#C0C0C0")
    ax.spines["bottom"].set_color("#C0C0C0")
    
    # Change ticks
    ax.tick_params(axis="both", labelsize=11, color="#C0C0C0")
    
    # Specify axis labels
    ax.set_xlabel("Feature importance", fontsize=14)
    ax.set_ylabel("", fontsize=14)
    
    # Change axis limits    
    ax.set_xlim(0, 0.2)
    
    # Specify axis tick spacing
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    # Add title
    ax.set_title(suffix)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f"../Figures/feature_importance_{suffix}.png", dpi=300)
