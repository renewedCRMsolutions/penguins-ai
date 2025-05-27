# File: features/nhl_features.py
class NHLFeatureEngine:
    def __init__(self):
        self.feature_groups = {
            'basic': ['shot_distance', 'shot_angle', 'period'],
            'advanced': ['rush_shot', 'rebound', 'cross_ice_pass'],
            'contextual': ['score_differential', 'time_remaining', 'home_away'],
            'player': ['shooter_goals_60', 'shooter_handedness'],
            'goalie': ['goalie_save_pct', 'shots_faced_period'],
            'team': ['powerplay', 'pulled_goalie', 'team_corsi']
        }