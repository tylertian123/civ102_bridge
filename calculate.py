import numpy as np
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt, patches

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# A list of point loads [(loc1, load1), (loc2, load2), ...]
# where loc1, loc2, ... are the distances of the loads from the left edge
Forces = List[Tuple[float, float]]


# A class to hold all the bridge and loading geometry constants
class Geometry:
    def __init__(self, values: Dict[str, Any]) -> None:
        self.train_wheel_load = values["train"]["totalWeight"] / 3 / 2 # Load on each wheel
        self.train_wheel_distance = values["train"]["wheelDistance"] # Distance between two wheels on the same carriage
        self.train_wheel_edge_distance = values["train"]["wheelEdgeDistance"] # Distance between the wheel and edge of the carriage
        self.train_car_distance = values["train"]["carDistance"] # Distance between the edges of two train carriages

        self.bridge_length = values["bridge"]["length"] # Length of the entire bridge
        self.bridge_supports = values["bridge"]["supports"] # Location of all the supports

        self.bridge_cross_sections = [CrossSection(d) for d in values["bridge"]["crossSections"]]
    @classmethod
    def from_yaml(cls, file: str) -> "Geometry":
        return Geometry(yaml.load(open(file, "r", encoding="utf-8"), Loader))


# A class to hold and compute properties of cross sections
class CrossSection:
    def __init__(self, values: Dict[str, Any]) -> None:
        self.geometry = values["geometry"]

        # Compute properties
        self.ybar = sum(w * h * (y + h / 2) for x, y, w, h in self.geometry) / sum(w * h for _, _, w, h in self.geometry)
        # Parallel axis theorem, with I of each piece being bh^3/12
        self.i = sum(w * h ** 3 / 12 + (y + h / 2 - self.ybar) ** 2 for x, y, w, h in self.geometry)
    
    def visualize(self, ax) -> None:
        """
        Draw the cross section onto a matplotlib plot to visualize it.
        """
        for x, y, w, h in self.geometry:
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="b", facecolor="none"))
        ax.set_xlim(min(x for x, _, _, _ in self.geometry) - 10, max(x + w for x, _, w, _ in self.geometry) + 10)
        ax.set_ylim(min(y for _, y, _, _ in self.geometry) - 10, max(y + h for _, y, _, h in self.geometry) + 10)
        ax.set_aspect("equal")

def load_train(geo: Geometry, dist: float) -> Forces:
    """
    Create loading condition for the train, with the right edge of the train at distance dist.
    """
    loads = []
    # Create 2 wheel point loads for each car
    for i in range(3):
        offset = i * (geo.train_wheel_distance + 2 * geo.train_wheel_edge_distance + geo.train_car_distance)
        loads.append(dist - offset - geo.train_wheel_edge_distance)
        loads.append(dist - offset - geo.train_wheel_edge_distance - geo.train_wheel_distance)
    # Sort loads in ascending order of location and exclude loads not on the bridge (negative because load is down)
    return [(loc, -geo.train_wheel_load) for loc in sorted(loads) if 0 <= loc <= geo.bridge_length]


def reaction_forces(geo: Geometry, loads: Forces) -> Forces:
    """
    Compute the two reaction forces and add them to the forces.
    """
    # Sum of moments
    ma = sum(load * (loc - geo.bridge_supports[0]) for loc, load in loads)
    fb = (0 - ma) / (geo.bridge_supports[1] - geo.bridge_supports[0])
    fa = -sum(load for _, load in loads) - fb
    forces = loads + [(geo.bridge_supports[0], fa), (geo.bridge_supports[1], fb)]
    forces.sort()
    return forces


def make_sfd(geo: Geometry, loads: Forces) -> np.ndarray:
    """
    Compute the Shear Force Diagram from loads.
    """
    shear = [0] * (geo.bridge_length + 1) # One point per mm
    # Accumulate point loads
    s = x = i = 0
    while x < len(shear) and i < len(loads):
        # For every point load reached, add its value
        if x <= loads[i][0] and x + 1 > loads[i][0]:
            s += loads[i][1]
        shear[x] = s
        # Increment x value and next point load
        x += 1
        if x > loads[i][0]:
            i += 1
    if abs(shear[-1]) > 1e-5:
        raise ValueError("Final shear not equal to zero!")
    return np.array(shear)


def make_bmd(geo: Geometry, sfd: np.ndarray) -> np.ndarray:
    """
    Compute the Bending Moment Diagram from the Shear Force Diagram.
    """
    bmd = [0] * len(sfd)
    moment = 0
    for x in range(len(sfd)):
        moment += sfd[x]
        bmd[x] = moment
    return np.array(bmd)


if __name__ == "__main__":
    geo = Geometry.from_yaml("design0.yaml")
    #forces = reaction_forces(geo, load_train(geo, 960))

    #plt.plot(np.arange(0, geo.bridge_length + 1, 1), make_bmd(geo, make_sfd(geo, forces)))
    #plt.show()

    geo.bridge_cross_sections[0].visualize(plt.gca())
    plt.show()
