#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import numpy as np
from math import pi
from typing import Tuple, List
from pandera.typing import DataFrame


# All fenwick events 
fenwick_events = ["SHOT", "MISSED SHOT", "GOAL"]


def read_data(end_season: int=2022, fenwick_events: List=fenwick_events) -> Tuple[DataFrame, DataFrame]:
    """
    Read data from all seasons, currently 2010-2011 until 2021-2022.

    Parameters
    ----------
    end_season : int, default is 2022
        The final season to consider.
    fenwick_events : List 
        List of all fenwick events.
        
    Returns
    -------
    pbp : DataFrame
        Data frame of all play by play events from all seasons.

    """

    # Read all data from all seasons
    pbp = pd.concat([pd.read_csv(f"../Data/GamePBPMerged/pbp_merged_{season}.csv", low_memory=False)
                     for season in range(2010, end_season)])
    
    # Read player data to get player shooting information
    players = pd.read_csv("../Data/players.csv", low_memory=False)
    
    # Find the player who shot the puck
    pbp["ShooterId"] = np.select([pbp.EventType.eq("BLOCKED SHOT"), 
                                  pbp.EventType.isin(fenwick_events)],
                                 [pbp.Player2, pbp.Player1], 
                                 default=np.nan)
    
    # Add information regarding player position and shooting style
    pbp = pbp.merge(players[["id", "position", "shootsCatches"]].rename(
        columns={"id": "ShooterId", "position": "Position", "shootsCatches": "Shoots"}),
        on="ShooterId", how="left")
    
    # Remove any events without location data
    pbp = pbp.loc[pbp.X.notna() & pbp.Y.notna()].copy()

    return pbp


def add_prior_events(pbp: DataFrame) -> DataFrame:
    """
    Add information regarding the prior event.

    Parameters
    ----------
    pbp : DataFrame
        Data frame of all play by play events from all seasons.

    Returns
    -------
    pbp_modified : DataFrame
        Modified data frame with additional columns describing the prior event.

    """
    
    # Copy to avoid changing in-place
    pbp_modified = pbp.copy()
        
    # Last event
    pbp_modified["LastEvent"] = pbp_modified.EventType.shift(1, fill_value="")
    
    # The time of the previous event
    pbp_modified["LastEventTime"] = pbp_modified.TotalElapsedTime.shift(1, fill_value=0)
    
    # Time since last event
    pbp_modified["TimeSinceLastEvent"] = pbp_modified.TotalElapsedTime - pbp_modified.LastEventTime
    
    # Last event attributes
    pbp_modified["LastTeam"] = pbp_modified.Team.shift(1, fill_value=False)
    
    # If the previous team is the same as current team
    pbp_modified["SameTeamAsPrevEvent"] = (pbp_modified["Team"] == pbp_modified["LastTeam"]).astype(int)
    
    # Determine the X-coordinate of the previous event
    pbp_modified["LastX"] = np.select(
        [pbp_modified.SameTeamAsPrevEvent.astype(bool), ~pbp_modified.SameTeamAsPrevEvent.astype(bool)],
        [pbp_modified.X.shift(1, fill_value=0),         -1 * pbp_modified.X.shift(1, fill_value=0)])
    
    # Determine the Y-coordinate of the previous event
    pbp_modified["LastY"] = np.select(
        [pbp_modified.SameTeamAsPrevEvent.astype(bool), ~pbp_modified.SameTeamAsPrevEvent.astype(bool)],
        [pbp_modified.Y.shift(1, fill_value=0),         -1 * pbp_modified.Y.shift(1, fill_value=0)])

    # Compute the distance from the last event
    pbp_modified["DistFromLastEvent"] = np.sqrt((pbp_modified.X - pbp_modified.LastX)**2 + 
                                                (pbp_modified.Y - pbp_modified.LastY)**2)
    
    # Compute the speed from the last event
    pbp_modified["SpeedFromLastEvent"] = (pbp_modified["DistFromLastEvent"] /
                                          pbp_modified["TimeSinceLastEvent"].replace(0, 1)).replace(0, 1)

    return pbp_modified


def add_extra_cols(pbp: DataFrame, fenwick_events: List=fenwick_events) -> DataFrame:
    """
    Add additional columns regarding e.g., home team, zone, and if a shot was a rush.

    Parameters
    ----------
    pbp : DataFrame
        Data frame of all play by play events from all seasons.
    fenwick_events : List 
        List of all fenwick events.
        
    Returns
    -------
    pbp_modified : DataFrame
        Modified data frame with additional columns added.

    """
    # Copy to avoid changing in-place
    pbp_modified = pbp.copy()
    
    # If the team is the home team
    pbp_modified["IsHome"] = (pbp_modified["Team"] == pbp_modified["HomeTeamName"]).astype(int)
    
    # Adjust goals for/against of of goal events to avoid data leakage
    pbp_modified.loc[pbp_modified.EventType.eq("GOAL") &
                  pbp_modified.IsHome.astype(bool), "GoalsFor"] -= 1
    
    pbp_modified.loc[pbp_modified.EventType.eq("GOAL") &
                  ~pbp_modified.IsHome.astype(bool), "GoalsAgainst"] -= 1
    
    # Get the distance from the pbp data
    pbp_modified["pbpDistance"] = pbp_modified.Description.str.extract("(\d+)(?= ft.)").astype(float)
    
    # Extract zone
    pbp_modified["Zone"] = pbp_modified.Description.str.extract("(?<=, )([A-z]{3})(?=\. Zone)")
    
    # Correct zone if wrong
    pbp_modified["Zone"] = np.select(
        [pbp_modified.Zone.eq("Def") & pbp_modified.EventType.eq("BLOCKED SHOT"),
         pbp_modified.EventType.isin(fenwick_events) & pbp_modified.Zone.eq("Def") & 
         pbp_modified.pbpDistance.le(64)],
              ["Off", "Off"], default=pbp_modified["Zone"])

    # Determine if a shot was a rush shot
    pbp_modified["IsRush"] = np.where(
        ((pbp_modified["TotalElapsedTime"] - pbp_modified["LastEventTime"]).le(4) & 
         ((pbp_modified["Zone"].ne(pbp_modified.Zone.shift(1, fill_value="Neu"))) & 
        (pbp_modified.Zone.shift(1, fill_value="Neu").ne("Neu")))) |
        ((pbp_modified["TotalElapsedTime"] - pbp_modified["LastEventTime"]).le(4) & 
         (pbp_modified["LastEvent"].isin(["TAKEAWAY", "GIVEAWAY"]))),
        1, 0
    )

    return pbp_modified

        
def add_players_on_ice(pbp: DataFrame) -> DataFrame:
    """
    Compute the number of skaters per ice at any given time. Also determine if
    the net of the opposition was empty and the current manpower situation.

    Parameters
    ----------
    pbp : DataFrame
        Data frame of all play by play events from all seasons.

    Returns
    -------
    pbp_modified : DataFrame
        Modified data frame with information of number of skaters/goalie added.

    """
    # Copy to avoid changing in-place
    pbp_modified = pbp.copy()
    
    # Fill NA with True for goalie on ice 
    pbp_modified[["AwayGoalieOnIce", "HomeGoalieOnIce"]] = pbp_modified[
        ["AwayGoalieOnIce", "HomeGoalieOnIce"]].fillna(False)
    
    # Convert goalie indicator columns to boolean
    pbp_modified[["AwayGoalieOnIce", "HomeGoalieOnIce"]] = pbp_modified[
        ["AwayGoalieOnIce", "HomeGoalieOnIce"]].astype(bool)
    
    # Compute the number of skaters on the ice for the home team
    pbp_modified["HomeSkatersOnIce"] = np.select(
        [pbp_modified.HomeGoalieOnIce, ~pbp_modified.HomeGoalieOnIce], 
        [pbp_modified[[f"HomePlayer{i}" for i in range(1, 7)]].notna().sum(axis=1) - 1,
         pbp_modified[[f"HomePlayer{i}" for i in range(1, 7)]].notna().sum(axis=1)],
        default=np.nan)
    
    # Compute the number of skaters on the ice for the away team
    pbp_modified["AwaySkatersOnIce"] = np.select(
        [pbp_modified.AwayGoalieOnIce, ~pbp_modified.AwayGoalieOnIce], 
        [pbp_modified[[f"AwayPlayer{i}" for i in range(1, 7)]].notna().sum(axis=1) - 1,
         pbp_modified[[f"AwayPlayer{i}" for i in range(1, 7)]].notna().sum(axis=1)],
        default=np.nan)

    # Determine if the net was empty from the shooting team's perspective
    pbp_modified["EmptyNet"] = np.select(
        [pbp_modified.IsHome.astype(bool), ~pbp_modified.IsHome.astype(bool)], 
        [~pbp_modified.AwayGoalieOnIce,    ~pbp_modified.HomeGoalieOnIce])
    
    # Compute the difference in skaters between the two teams
    pbp_modified["SkaterDifferential"] = np.select(
        [pbp_modified.IsHome.astype(bool), ~pbp_modified.IsHome.astype(bool)],
        [pbp_modified.HomeSkatersOnIce - pbp_modified.AwaySkatersOnIce, 
         pbp_modified.AwaySkatersOnIce - pbp_modified.HomeSkatersOnIce])
    
    # Compute the number of skaters on the opposition team
    pbp_modified["OppSkatersOnIce"] = np.select(
        [pbp_modified.IsHome.astype(bool), ~pbp_modified.IsHome.astype(bool)],
        [pbp_modified.AwaySkatersOnIce,     pbp_modified.HomeSkatersOnIce])

    # Determine the manpower situation at a given event
    pbp_modified["ManpowerSituation"] = np.select(
        [pbp_modified.IsHome.astype(bool), ~pbp_modified.IsHome.astype(bool)],
        [pbp_modified[["AwaySkatersOnIce", "HomeSkatersOnIce"]].astype(int).astype(str).apply(
            lambda x: "v".join([x.HomeSkatersOnIce, x.AwaySkatersOnIce]), axis=1), 
         pbp_modified[["AwaySkatersOnIce", "HomeSkatersOnIce"]].astype(int).astype(str).apply(
             lambda x: "v".join([x.AwaySkatersOnIce, x.HomeSkatersOnIce]), axis=1)])

    return pbp_modified


def get_distance_and_angle(shots: DataFrame) -> DataFrame:
    """
    Compute distance and angle of shots. Also compute the change in angle if
    the previous event also was a shot.

    Parameters
    ----------
    shots : DataFrame
        Data frame containing all (fenwick) shots.

    Returns
    -------
    shots_modified : DataFrame
        Modified data frame with distance and angle related columns added.

    """
    
    # Copy to avoid changing in-place
    shots_modified = shots.copy()
    
    # Adjust shooting coordinates for shots that are in the goal
    shots_modified.loc[shots_modified.X.eq(89), "X"] -= 1
    shots_modified.loc[shots_modified.X.eq(-89), "X"] += 1

    # Compute the distance of the shot
    shots_modified["Distance"] = np.sqrt((89 - shots_modified.X)**2 + (shots_modified.Y)**2)

    # Compute the angle of the shot
    shots_modified["Angle"] = np.arctan(shots_modified.Y / (89 - np.abs(shots_modified.X))) * (180 / pi)

    # Special case where the angle is 0 degrees
    shots_modified.loc[shots_modified.Y.eq(0), "Angle"] = 0
    
    # Determine if a shot was a rebound or not
    shots_modified["IsRebound"] = np.where((shots_modified["LastEvent"].isin(["MISSED SHOT", 
                                                                "SHOT"])) & 
                                           # (shots_modified["TimeSinceLastEvent"].le(2)) & 
                                           (shots_modified["SameTeamAsPrevEvent"]), 1, 0)
    
    # Get the previous shot angle
    shots_modified["PreviousAngle"] = shots_modified.Angle.shift(1, fill_value=0)
    
    # Get the angle change for rebound shots_modified
    shots_modified["AngleChange"] = np.select(
        [shots_modified.IsRebound.astype(bool) & 
         (np.sign(shots_modified.Angle) == np.sign(shots_modified.PreviousAngle)),
         shots_modified.IsRebound.astype(bool) & 
         (np.sign(shots_modified.Angle) != np.sign(shots_modified.PreviousAngle))
         ], 
        [np.abs(shots_modified.Angle - shots_modified.PreviousAngle),
         np.abs(shots_modified.Angle + shots_modified.PreviousAngle)],
        default=0)
    
    # Compute the time difference from the last shot
    shots_modified["TimeSinceLastShot"] = shots_modified.TotalElapsedTime - shots_modified.TotalElapsedTime.shift(1, fill_value=0)
    
    # Compute the change in angle for rebound shots_modified
    shots_modified["AngleChangeSpeed"] = shots_modified["AngleChange"] / shots_modified["TimeSinceLastShot"].replace(0, 1)
    
    # Save absolute value for angle
    shots_modified["Angle"] = np.abs(shots_modified["Angle"])

    return shots_modified


def get_shots(end_season: int=2022, fenwick_events: List=fenwick_events) -> Tuple[DataFrame, 
                                                                                  DataFrame]:
    """
    Get all (fenwick) shots from play by play data.

    Parameters
    ----------
    end_season : int, default is 2022.
        The final season to consider.

    Returns
    -------
    pbp : DataFrame
        Data frame containing all play by play events.
    shots : DataFrame
        Data frame containing all (fenwick) shots.

    """
    
    # Read all data
    pbp = read_data(end_season, fenwick_events)
    
    # Add the prior events
    pbp = add_prior_events(pbp)
    
    # Add and correct additional columns
    pbp = add_extra_cols(pbp, fenwick_events)
    
    # Add columns for number of players/skaters on the ice
    pbp = add_players_on_ice(pbp)
    
    # Get all shooting events
    shots = pbp.loc[pbp.EventType.isin(fenwick_events) & 
                    pbp.PeriodNumber.le(4)].copy()
    
    # Remove penalty shots and shots by goalies
    shots = shots.loc[~shots.Description.str.contains("Penalty") &
                      shots.Position.ne("G")]
    
    # Compute the score difference from the shooting team's perspective
    shots["ScoreDifferential"] = np.select([shots.IsHome.astype(bool), ~shots.IsHome.astype(bool)],
                                             [shots["GoalsFor"] - shots["GoalsAgainst"],
                                              shots["GoalsAgainst"] - shots["GoalsFor"]])
    
    # Add special cases for when |goal difference| > 3
    shots["ScoreDifferential"] = np.select([shots.ScoreDifferential > 3, shots.ScoreDifferential < -3],
                                             [3, -3], default=shots.ScoreDifferential)
    
    
    # Remove shots where there are too few players on the ice for either team
    shots = shots.loc[shots.AwaySkatersOnIce.ge(3) & 
                      shots.HomeSkatersOnIce.ge(3)].copy()
    
    # Compute distance and angle of shots
    shots = get_distance_and_angle(shots)
    
    # Determine if the shot was taken of the "wrong" wing based on the shooter's handedness
    shots["OffWing"] = np.select(
        [shots.Shoots.eq("L") & shots.Y.lt(0),
         shots.Shoots.eq("L") & shots.Y.ge(0),
         shots.Shoots.eq("R") & shots.Y.lt(0),
         shots.Shoots.eq("R") & shots.Y.ge(0)],
        [1, 0, 0, 1])
    
    # Determine if the shooter was a forward
    shots["IsForward"] = np.select([shots.Position.isin(["D"])], [False], default=True)
    
    # Create a grouping for the category of the last event 
    shots["LastEventGroup"] = np.select(
        [shots.LastEvent.eq("FACEOFF"),
         shots.LastEvent.isin(["MISSED SHOT", "BLOCKED SHOT"]),
         shots.LastEvent.eq("SHOT")],
        ["LastEventFaceoff", "LastEventShotNotOnGoal", "LastEventShotOnGoal"],
        default="LastEventNonShot")
    
    # Create a variable for if a shot was taken behind the net
    shots["BehindNet"] = shots.X.gt(89)
    
    return pbp, shots


def create_dummy_vars(shots: DataFrame) -> List:
    """
    Create dummy variables for categorical columns to use in modeling. 
    Dummies are created for shot type, manpower situation, score differential,
    team who performed the last event, and zone.

    Parameters
    ----------
    shots : DataFrame
        Data frame containing all (fenwick) shots.

    Returns
    -------
    List
        List of all dummy variables, stored as numpy arrays.

    """
    # Create dummy variables for the shot type
    shot_type_cat = pd.get_dummies(shots.ShotType)
    
    # Create all the manpower situations
    manpower_situation = shots.ManpowerSituation.replace(
        {"6v5": "PP1", "5v4": "PP1", "4v3": "PP1",
         "6v4": "PP2", "5v3": "PP2", "6v3": "PP2",
         "5v6": "SH", "4v5": "SH", "4v6": "SH", "3v4": "SH", "3v5": "SH", "3v6": "SH"
          })
    
    # Create dummy variables for the manpower situtation
    manpower_cat = pd.get_dummies(manpower_situation)
    
    # Create dummy variables for the score differential
    score_differential_cat = pd.get_dummies(shots.ScoreDifferential, prefix="GD")

    # Create dummy variables for the previous event 
    last_team_event_cat = pd.get_dummies(
        shots[["LastEventGroup", "SameTeamAsPrevEvent"]].replace(
            {0: "Opp", 1: "Team"}).apply(lambda x: "".join(x.astype(str)), axis=1))
    
    # Create dummy variables for zone
    zone_cat = pd.get_dummies(shots.Zone.replace({"Off": "OffZone", 
                                                  "Neu": "NeuZone", 
                                                  "Def": "DefZone"}))
    
    # Save all variables in a list
    dummy_vars_list = [manpower_cat, shot_type_cat, score_differential_cat, 
                       last_team_event_cat, zone_cat]
    
    return dummy_vars_list
