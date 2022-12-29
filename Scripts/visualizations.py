#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from preprocessShots import get_shots
from matplotlib import font_manager
from matplotlib import rcParams

# Find all fonts
font_files = font_manager.findSystemFonts(fontpaths="../Fonts")
   
# Add all fonts
for font_file in font_files:
   font_manager.fontManager.addfont(font_file)
   
# Set Roboto as default font
rcParams["font.family"] = "Roboto"

# Get all shots
_, shots = get_shots(2022, fenwick_events=["GOAL", "SHOT", "MISSED SHOT"])

# Create a column to indicate if a goal was scored or not
shots["Goal"] = np.select([shots.EventType.eq("GOAL")], [1], default=0)

# Round distance and angle
shots["Distance"] = np.around(shots["Distance"])
shots["Angle"] = np.around(shots["Angle"])

# Round speed events
shots["DistFromLastEvent"] = np.around(shots["DistFromLastEvent"])
shots["SpeedFromLastEvent"] = np.around(shots["SpeedFromLastEvent"])
shots["AngleChangeSpeed"] = np.around(shots["AngleChangeSpeed"])


def plot_proportion_over_variable(var, ylims=(0, 1), tickspacing=0.2):
    # Compute the number of goals and no goals per variable
    variable = shots.groupby([var, "Goal"], as_index=False).size().pivot(
        index=var, values="size", columns="Goal").fillna(0).rename(
        columns={0: "NoGoal", 1: "Goal"}).reset_index()
    
    # Compute the proportion of goals per variable
    variable["Prop"] = variable["Goal"] / (variable["NoGoal"] + variable["Goal"])
    
    # Initialize a figure
    fig, ax = plt.subplots(figsize=(12, 8))       
    
    # Create the initial plot
    ax.plot(variable[var], variable["Prop"], color="#FF9913", linewidth=2.5)
          
    # Change plot spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#C0C0C0")
    ax.spines["bottom"].set_color("#C0C0C0")
    
    # Change ticks
    ax.tick_params(axis="both", labelsize=12, color="#C0C0C0")
    
    # Specify axis labels
    ax.set_xlabel(var, fontsize=14)
    ax.set_ylabel("Percentage of shots that become a goal", fontsize=14)

    # Change axis limits    
    ax.set_ylim(ylims[0], ylims[1])
    
    # Specify axis tick spacing
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tickspacing))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f"../Figures/goal_proportion_{var}.png", dpi=300)

    
plot_proportion_over_variable("Distance")
plot_proportion_over_variable("Angle", ylims=(0, 0.2), tickspacing=0.05)


# Compute the proporotion of goals per variable and value for binary variables
binary_vars_goal_prop = pd.concat(
    [shots.groupby(["OffWing"], as_index=False)["Goal"].mean().assign(
        Variable="OffWing").rename(columns={"OffWing": "Value"}),
    shots.groupby(["IsForward"], as_index=False)["Goal"].mean().assign(
        Variable="IsForward").rename(columns={"IsForward": "Value"}),
    shots.groupby(["BehindNet"], as_index=False)["Goal"].mean().assign(
        Variable="BehindNet").rename(columns={"BehindNet": "Value"}),
    shots.groupby(["EmptyNet"], as_index=False)["Goal"].mean().assign(
        Variable="EmptyNet").rename(columns={"EmptyNet": "Value"})
    ]).replace({True: 1, False: 0})

# Proportion of goals for non-binary variables
non_binary_vars_goal_prop = pd.concat(
    [shots.replace({-3: r"$\leq$-3", 3: r"$\geq$3"}).groupby(["ScoreDifferential"], 
                                                             as_index=False)["Goal"].mean().assign(
        Variable="ScoreDifferential").rename(columns={"ScoreDifferential": "Value"}),
    shots.groupby(["ShotType"], as_index=False)["Goal"].mean().assign(
        Variable="ShotType").rename(columns={"ShotType": "Value"}),
    shots.replace(
        {"6v5": "PP1", "5v4": "PP1", "4v3": "PP1",
         "6v4": "PP2", "5v3": "PP2", "6v3": "PP2",
         "5v6": "SH", "4v5": "SH", "4v6": "SH", 
         "3v4": "SH", "3v5": "SH", "3v6": "SH",
         "5v5": "EV", "4v4": "EV", "3v3": "EV"}).groupby(["ManpowerSituation"], 
                                                         as_index=False)["Goal"].mean().assign(
        Variable="ManpowerSituation").rename(columns={"ManpowerSituation": "Value"})]).replace(
            {"ManpowerSituation": "Manpower", "ScoreDifferential": "Goal differential",
             "ShotType": "Shot type"})

# Convert value to string             
non_binary_vars_goal_prop["Value"] = non_binary_vars_goal_prop["Value"].astype(str).replace(
    "\.0", "", regex=True)

# Specify colors
color_list = {r"$\leq$-3": '#6EA6CD', "-2": '#98CAE1', "-1": '#C2E4EF', 
              "0": '#EAECCC', "1": '#FEDA8B', "2": '#FDB366', r"$\geq$3": '#F67E4B', 
              "EV": "#E7D4E8", "PP1": "#C2A5CF", "PP2": "#9970AB", "SH": "#762A83",
              "Slap": "#C6DBED", "Wrap-around": "#9DCCEF", "Wrist": "#60BCE9",
              "Snap": "#42A7C6", "Backhand": "#238F9D", "Deflected": "#00767B", 
              "Tip-In": "#125A56"
              }

# Specify order of factor
factor_order = [r"$\leq$-3", "-2", "-1", "0", "1", "2", r"$\geq$3", 
                "EV", "PP1", "PP2", "SH", 
                "Slap", "Wrap-around", "Wrist", 
                "Snap", "Backhand", "Deflected", "Tip-In"]

# Create an order column
non_binary_vars_goal_prop["Order"] = pd.Categorical(non_binary_vars_goal_prop.Value, 
                                                    categories=factor_order, 
                                                    ordered=True)

# Initialize figure
ax = plt.figure(figsize=(12, 8))

# Create a plot for the number of players per cluster and team
grid = sns.catplot(data=non_binary_vars_goal_prop.sort_values(["Variable", "Order"]), 
                   x="Value", y="Goal", hue="Value",
                   col="Variable", dodge=False, legend=False, 
                   kind='bar', col_wrap=3, sharex=False, palette=color_list)

# Specify limits
grid.set(ylim=(0, 0.2))
   
# Specify axis labels
grid.set_axis_labels(x_var="", 
                     y_var="Percentage of shots that become a goal")
   
# Specify plot style
sns.set_style("ticks", {"axes.edgecolor": "C0C0C0",
                        "patch.edgecolor": "black"
                        })
   
# Remove spines
sns.despine()

# Specify facet title
grid.set_titles("{col_name}")

# Specify axis tick spacing
for axis in grid.axes:
    axis.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

# Save plot
plt.savefig("../Figures/goal_proportion_manpower_GD.png", dpi=300)
