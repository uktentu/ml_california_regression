from pydantic import BaseModel, Field
from typing import Literal

class HouseFeatures(BaseModel):
    longitude: float = Field(..., ge=-125, le=-113)
    latitude: float = Field(..., ge=32, le=43)
    housing_median_age: float = Field(..., ge=0)
    total_rooms: float = Field(..., ge=0)
    total_bedrooms: float = Field(..., ge=0)
    population: float = Field(..., ge=0)
    households: float = Field(..., ge=0)
    median_income: float = Field(..., ge=0)
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
