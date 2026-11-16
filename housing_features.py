import numpy as np
import pandas as pd


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of housing predictors with ratio and interaction features."""
    data = data.copy()
    data["rooms_per_household"] = (
        data["total_rooms"] / data["households"].replace(0, np.nan)
    )
    data["bedrooms_per_room"] = (
        data["total_bedrooms"] / data["total_rooms"].replace(0, np.nan)
    )
    data["population_per_household"] = (
        data["population"] / data["households"].replace(0, np.nan)
    )
    data["income_x_rooms_per_household"] = (
        data["median_income"] * data["rooms_per_household"]
    )
    data["latitude_x_longitude"] = data["latitude"] * data["longitude"]
    return data
