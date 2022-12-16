#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Rasmus Säfvenberg


# =============================================================================
# Even strength
# =============================================================================
ev_params = {'alpha': 48.0,
             'colsample_bytree': 0.5,
             'gamma': 0.0,
             'lambda': 0.0,
             'learning_rate': 0.2,
             'max_delta_step': 3.0,
             'max_depth': 9.0,
             'min_child_weight': 4.0,
             'subsample': 0.85}

# The baseline log loss is: 2.163072499771719
# The baseline AUC is: 0.5
# The log loss is: 0.2048906972821979
# The AUC is: 0.7715389260407994

# Percentage decrease in log loss: 0.9053
# Percentage increase of AUC: 0.5433

# =============================================================================
# Powerplay
# =============================================================================
pp_params = {'alpha': 0.0,
             'colsample_bytree': 0.7,
             'gamma': 3.0,
             'lambda': 0.0,
             'learning_rate': 0.05,
             'max_delta_step': 8.0,
             'max_depth': 8.0,
             'min_child_weight': 4.0,
             'subsample': 0.75}

# The baseline log loss is: 3.3431503287196693
# The baseline AUC is: 0.5
# The log loss is: 0.29644149967619504
# The AUC is: 0.6922239059164517

# Percentage decrease in log loss: 0.9113
# Percentage increase of AUC: 0.3844

# =============================================================================
# Short-handed
# =============================================================================
sh_params = {'alpha': 0.0,
             'colsample_bytree': 0.75,
             'gamma': 0.0,
             'lambda': 0.0,
             'learning_rate': 0.15,
             'max_delta_step': 5.0,
             'max_depth': 2.0,
             'min_child_weight': 0.0,
             'subsample': 0.7}

# The baseline log loss is: 2.3623505426192866
# The baseline AUC is: 0.5
# The log loss is: 0.20096607231803904
# The AUC is: 0.8273362681495211

# Percentage decrease in log loss: 0.9149
# Percentage increase of AUC: 0.6546

# =============================================================================
# Empty net
# =============================================================================
en_params = {'alpha': 0.0,
             'colsample_bytree': 0.75,
             'gamma': 0.0,
             'lambda': 0.0,
             'learning_rate': 0.15,
             'max_delta_step': 5.0,
             'max_depth': 2.0,
             'min_child_weight': 0.0,
             'subsample': 0.7}

# The baseline log loss is: 11.910198618048579
# The baseline AUC is: 0.5
# The log loss is: 0.5744872317630155
# The AUC is: 0.6889178947368422

# Percentage decrease in log loss: 0.9518
# Percentage increase of AUC: 0.3778
