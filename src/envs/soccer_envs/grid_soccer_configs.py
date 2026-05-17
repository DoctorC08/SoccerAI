from pydantic import BaseModel, model_validator, Field
from typing import List, Tuple, Dict
from enum import Enum

class DataSourceTypeEnum(str, Enum):
    ONES = "ones"
    TWOS = "twos"
    ONEVTWO = "onevtwo"
    CUSTOM = "custom"


class RewardConfig(BaseModel):


    @model_validator(mode='after')
    def 


class GridSoccerConfigs(BaseModel):
    preset_env: DataSourceTypeEnum
    n_players: List[int] 
    size: List[int] = Field(
        default=[30, 20], 
        description="size of the field: [width, height] of grid environment"
    )
    starting_location: Dict[int | List[int]] = Field(
        default=None, 
        description="starting location of players for custom env: Dict[int | List[int]] "
        "- {0 : [+/- x, +/- y], team num : [player1_x, player1_y], team num : [player2_x, player2_y], ...} - team num must be in range [0, 2] use 0 to define random starting range"
    )

    @model_validator(mode='after')
    def validate_env(self):
        assert self.preset_env is not None, "preset_env must be defined"

        if self.preset_env == DataSourceTypeEnum.CUSTOM:
            assert self.n_players is not None and self.n_players > 0, "n_players must be defined for custom env"
            assert len(self.n_players) == 2, "n_players length must be 2"
            

        



