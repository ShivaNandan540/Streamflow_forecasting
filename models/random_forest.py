from sklearn.ensemble import RandomForestRegressor

def build_random_forest(random_state=42):
    return RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )
