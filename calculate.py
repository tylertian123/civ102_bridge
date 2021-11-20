import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# A list of point loads [(loc1, load1), (loc2, load2), ...]
# where loc1, loc2, ... are the distances of the loads from the left edge
LoadingCondition = List[Tuple[float, float]]


# A class to hold all the bridge and loading geometry constants
class Geometry:
    def __init__(self, values: Dict[str, Any]):
        self.train_wheel_load = values["train"]["totalWeight"] / 3 / 2 # Load on each wheel
        self.train_wheel_distance = values["train"]["wheelDistance"] # Distance between two wheels on the same carriage
        self.train_wheel_edge_distance = values["train"]["wheelEdgeDistance"] # Distance between the wheel and edge of the carriage
        self.train_car_distance = values["train"]["carDistance"] # Distance between the edges of two train carriages

        self.bridge_length = values["bridge"]["length"] # Length of the entire bridge

    @classmethod
    def from_yaml(cls, file: str) -> "Geometry":
        return Geometry(yaml.load(open(file, "r", encoding="utf-8"), Loader))


def load_train(geo: Geometry, dist: float) -> LoadingCondition:
    """
    Create loading condition for the train, with the right edge of the train at distance dist.
    """
    loads = []
    # Create 2 wheel point loads for each car
    for i in range(3):
        offset = i * (geo.train_wheel_distance + 2 * geo.train_wheel_edge_distance + geo.train_car_distance)
        loads.append(dist - offset - geo.train_wheel_edge_distance)
        loads.append(dist - offset - geo.train_wheel_edge_distance - geo.train_wheel_distance)
    # Sort loads in ascending order of location and exclude loads not on the bridge
    return [(loc, geo.train_wheel_load) for loc in sorted(loads) if 0 <= loc <= geo.bridge_length]


if __name__ == "__main__":
    geo = Geometry.from_yaml("design0.yaml")
    print(load_train(geo, 960))
