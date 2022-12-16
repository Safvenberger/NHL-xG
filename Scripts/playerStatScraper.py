#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg

import pandas as pd
import re
from tqdm import tqdm
from typing import List
from pandera.typing import DataFrame


def get_player_year_stats(player_id: int) -> DataFrame:
    """
    Get the available stats for a list of players. Note that the available
    stats vary per year/season.

    Parameters
    ----------
    player_id : int
        The id of the player to get stats for.
        
    Returns
    -------
    player_stats : DataFrame
        Data frame of a player's stats, one row per player and season.

    """
    
    # Url for a stats per season for a given player
    url = f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=yearByYear"
    
    # Get player stats per season as a json document
    player_json = pd.read_json(url)
    
    # Extract player stats
    player_stats = pd.json_normalize(player_json["stats"].iloc[0]["splits"])
    
    # Remove columns
    player_stats.drop([col for col in ["team.link", "league.link"] if col in player_stats.columns], 
                      axis=1, inplace=True)
    
    # Rename columns
    player_stats = player_stats.rename(columns=lambda x: re.sub("\.name|stat\.", "" , x))
    player_stats = player_stats.rename(columns=lambda x: re.sub("\.id", "Id" , x))

    # Capitalize the first letter of each columns
    player_stats.columns = [col[:1].upper() + col[1:] for col in player_stats.columns]
    
    return player_stats


def get_stats_for_all_players(player_id_list: List) -> DataFrame:
    """
    Get the available stats for a list of players. Note that the available
    stats vary per year/season.

    Parameters
    ----------
    player_id_list : List 
        List of all players to get stats for.
        
    Returns
    -------
    player_stats_all : DataFrame
        Data frame of all player stats, one row per player and season.

    """
    
    # Create a storage for results
    player_stats_dict = {}
    
    # Loop over all unique player ids
    for player_id in tqdm(player_id_list):
        
        # Get stats for the player
        player_stats = get_player_year_stats(player_id)
        
        # Add a column for player id
        player_stats.insert(0, "PlayerId", player_id)
        
        # Save in the dictionary
        player_stats_dict[player_id] = player_stats
        
    # Get player stats for all players
    player_stats_all = pd.concat(player_stats_dict).reset_index(drop=True)
    
    return player_stats_all 


if __name__ == "__main__":
    
    # Get all players
    players = pd.read_csv("../Data/players.csv", low_memory=False)[["fullName", "id", 
                                                                    "centralRegistryPosition",
                                                                    "shootsCatches"]]

    # Get all player stats
    player_stats_all = get_stats_for_all_players(players.id.to_list())
    
    # Add player name
    players_stats_all_named = player_stats_all.merge(players.rename(
        columns={"id": "PlayerId", "fullName": "Name", "shootsCatches": "Shoots", 
                 "centralRegistryPosition": "Position"}), how="left")
    
    # Save to .csv files
    players_stats_all_named.to_csv("../Data/player_stats_all.csv", index=False)
    
    # Only NHL stats
    players_stats_all_named.loc[player_stats_all.League.eq("National Hockey League")].to_csv(
        "../Data/player_stats_NHL.csv", index=False)
    
    # Only AHL stats
    players_stats_all_named.loc[player_stats_all.League.eq("AHL")].to_csv(
        "../Data/player_stats_AHL.csv", index=False)
