import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt, patches
from matplotlib.axes import Axes

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# A list of point loads [(loc1, load1), (loc2, load2), ...]
# where loc1, loc2, ... are the distances of the loads from the left edge
Forces = List[Tuple[float, float]]


def floatgeq(a: float, b: float) -> bool:
    """
    Test floats for greater than or almost-equality.
    """
    return a > b or math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)


def floatleq(a: float, b: float) -> bool:
    """
    Test floats for less than or almost-equality.
    """
    return a < b or math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)


# A class to hold and compute properties of cross sections
class CrossSection:
    def __init__(self, values: Dict[str, Any]) -> None:
        # Geometry consists of a list of rectangles in the form of [x, y, width, height]
        self.geometry = values["geometry"]
        self.min_b_height = values["minBHeight"]

        # Compute properties
        self.ytop = max(y + h for _, y, _, h in self.geometry)
        self.ybot = min(y for _, y, _, h in self.geometry)
        self.area = sum(w * h for _, _, w, h in self.geometry)
        self.ybar = sum(w * h * (y + h / 2) for _, y, w, h in self.geometry) / self.area
        # Parallel axis theorem, with I of each piece being bh^3/12
        self.i = sum(w * h ** 3 / 12 + (y + h / 2 - self.ybar) ** 2 for _, y, w, h in self.geometry)
    
    def calculate_b(self, y0: float, above: Optional[bool] = None) -> None:
        """
        Calculate the total cross-sectional width b at depth y0.

        above specifies boundary behaviour. If it's true, then when y0 is at a boundary, the piece's width
        is only counted if it's above y0; if it's false, then the piece's width is only counted if it's below y0.
        If it's None, then the lesser of the two will be returned.
        """
        if above is None:
            return min(self.calculate_b(y0, True), self.calculate_b(y0, False))
        # Consider the piece if y0 is in the middle of it
        # That is, y + h should be above y0, and y should be below y0
        # Note float comparisions...
        return sum(w for _, y, w, h in self.geometry
                if floatgeq(y + h, y0) and floatleq(y, y0)
                # And then consider whether the rest of the piece is above or below y0
                and ((above and y + h > y0) or (not above and y < y0)))

    def calculate_q(self, y0: float) -> None:
        """
        Calculate the first moment of area about the centroid, Q, for a given depth y0 (relative to the bottom).
        """
        # Integrate from the top
        q = 0
        for _, y, w, h in self.geometry:
            # Do not consider pieces entirely below y0
            # We don't need to worry about direct float comparision here
            # If y + h is very close to y0, then the amount of this piece above y0 is very small
            # So most of it will be cut out in the following calculations
            if y + h < y0:
                continue
            # If the bottom of the piece is below y0, cut it off at y0
            bottom = max(y, y0)
            # Update height according to new bottom
            h = y + h - bottom
            # Add up the pieces
            # bottom + h / 2 is the local centroid of the piece
            q += w * h * ((bottom + h / 2) - self.ybar)
        # Since the distance from ybar is signed, the resulting Q might be negative
        # But shear stress doesn't really have a direction so Q is taken to be the absolute value
        return abs(q)
    
    def calculate_shear_failure_force(self, tau: float) -> float:
        """
        Calculate the shear force Vfail that causes shear failure of the matboard.

        tau is the shear strength of the matboard.
        """
        # tau = VQ/Ib => V = tau*Ib/Q
        # Consider two points: the centroid, where Q is maximized, and the depth where b is minimized
        v = tau * self.i * self.calculate_b(self.ybar, above=None) / self.calculate_q(self.ybar)
        if self.min_b_height is not None:
            v = min(v, tau * self.i * self.calculate_b(self.min_b_height, above=None) / self.calculate_q(self.min_b_height))
        return v

    def visualize(self, ax: Axes) -> None:
        """
        Draw the cross section onto a matplotlib plot to visualize it.
        """
        for x, y, w, h in self.geometry:
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="b", facecolor="none"))
        xmin = min(x for x, _, _, _ in self.geometry)
        xmax = max(x + w for x, _, w, _ in self.geometry)
        ymin = min(y for _, y, _, _ in self.geometry)
        ymax = max(y + h for _, y, _, h in self.geometry)
        ax.set_xlim(xmin - 10, xmax + 10)
        ax.set_ylim(ymin - 10, ymax + 10)
        ax.axhline(self.ybar, c="r")
        ax.set_aspect("equal")
        print(f"Max y: {self.ytop}\nMin y: {self.ybot}\nArea: {self.area}\nCentroid: {self.ybar}\nI: {self.i}")


# A class to hold all the bridge and loading geometry constants
class Bridge:
    def __init__(self, values: Dict[str, Any]) -> None:
        self.train_wheel_load = values["loading"]["train"]["totalWeight"] / 3 / 2 # Load on each wheel
        self.train_wheel_distance = values["loading"]["train"]["wheelDistance"] # Distance between two wheels on the same carriage
        self.train_wheel_edge_distance = values["loading"]["train"]["wheelEdgeDistance"] # Distance between the wheel and edge of the carriage
        self.train_car_distance = values["loading"]["train"]["carDistance"] # Distance between the edges of two train carriages
        self.point_load_locations = values["loading"]["points"]

        self.length = values["bridge"]["length"] # Length of the entire bridge
        self.supports = values["bridge"]["supports"] # Location of all the supports

        self.sigmat = values["bridge"]["material"]["sigmat"] # Ultimate tensile stress
        self.sigmac = values["bridge"]["material"]["sigmac"] # Ultimate compressive stress
        self.tau = values["bridge"]["material"]["tau"] # Max shear
        self.glue_tau = values["bridge"]["material"]["glueTau"]
        self.e = values["bridge"]["material"]["e"] # Young's modulus
        self.nu = values["bridge"]["material"]["nu"] # Poisson's ratio

        self.cross_sections = [(d["start"], d["stop"], CrossSection(d)) for d in values["bridge"]["crossSections"]]
    
    @classmethod
    def from_yaml(cls, file: str) -> "Bridge":
        return Bridge(yaml.load(open(file, "r", encoding="utf-8"), Loader))

    def load_train(self, dist: float) -> Forces:
        """
        Create loading condition for the train, with the right edge of the train at distance dist.
        """
        loads = []
        # Create 2 wheel point loads for each car
        for i in range(3):
            offset = i * (self.train_wheel_distance + 2 * self.train_wheel_edge_distance + self.train_car_distance)
            loads.append(dist - offset - self.train_wheel_edge_distance)
            loads.append(dist - offset - self.train_wheel_edge_distance - self.train_wheel_distance)
        # Sort loads in ascending order of location and exclude loads not on the bridge (negative because load is down)
        return [(loc, -self.train_wheel_load) for loc in sorted(loads) if 0 <= loc <= self.length]

    def load_points(self, p: float) -> Forces:
        """
        Create loading condition for applying loads P at each location in the geometry.
        """
        return [(loc, -p) for loc in self.point_load_locations]

    def reaction_forces(self, loads: Forces) -> Forces:
        """
        Compute the two reaction forces and add them to the forces.
        """
        # Sum of moments
        ma = sum(load * (loc - self.supports[0]) for loc, load in loads)
        # Sum of moments + Fb * distance of b = 0
        fb = (0 - ma) / (self.supports[1] - self.supports[0])
        # Sum of forces + Fb + Fa = 0
        fa = -sum(load for _, load in loads) - fb
        # Add forces to array
        forces = loads + [(self.supports[0], fa), (self.supports[1], fb)]
        forces.sort()
        return forces

    def make_sfd(self, loads: Forces) -> np.ndarray:
        """
        Compute the Shear Force Diagram from loads.
        """
        shear = [0] * (self.length + 1) # One point per mm
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

    def make_bmd(self, sfd: np.ndarray) -> np.ndarray:
        """
        Compute the Bending Moment Diagram from the Shear Force Diagram.
        """
        bmd = [0] * len(sfd)
        moment = 0
        for x in range(len(sfd)):
            moment += sfd[x]
            bmd[x] = moment
        return np.array(bmd)

    def make_curvature_diagram(self, bmd: np.ndarray) -> np.ndarray:
        """
        Compute the Curvature Diagram from the Bending Moment Diagram.
        """
        i = 0
        cs = self.cross_sections[i]
        phi = []
        for x in range(len(bmd)):
            # Find the right cross section
            while x > cs[1]:
                i += 1
                cs = self.cross_sections[i]
            # Curvature phi = M / EI
            phi.append(bmd[x] / (self.e * cs[2].i))
        return np.array(phi)

    def calculate_tensile_failure_moment(self) -> float:
        """
        Calculate the bending moment Mfail that would cause a matboard flexural tensile failure.

        Returns a signed quantity. The absolute value of the return value is required magnitude of the bending moment;
        the sign indicates the direction (positive - positive moment, bottom in tension;
        negative - negative moment, top in tension)
        """
        # Need to do this for each cross section since they're all different
        # sigma = M*y/I => M = sigma*I/y
        # sigma1 is for when the top is in tension, i.e. negative curvature
        sigma1 = min(self.sigmat * cs.i / (cs.ytop - cs.ybar) for _, _, cs in self.cross_sections)
        # When the bottom is in tension, i.e. positive curvature
        sigma2 = min(self.sigmat * cs.i / (cs.ybar - cs.ybot) for _, _, cs in self.cross_sections)
        if sigma1 < sigma2:
            return -sigma1
        return sigma2
    
    def calculate_compressive_failure_moment(self) -> float:
        """
        Calculate the bending moment Mfail that would cause a matboard flexural compressive failure.

        Returns a signed quantity. The absolute value of the return value is required magnitude of the bending moment;
        the sign indicates the direction (positive - positive moment, top in compression;
        negative - negative moment, bottom in compression)
        """
        # See logic above
        # sigma1 is for when the top is in compression, i.e. positive curvature
        sigma1 = min(self.sigmac * cs.i / (cs.ytop - cs.ybar) for _, _, cs in self.cross_sections)
        # When the bottom is in compression, i.e. negative curvature
        sigma2 = min(self.sigmac * cs.i / (cs.ybar - cs.ybot) for _, _, cs in self.cross_sections)
        # This time return sigma1 as positive and sigma2 as negative
        if sigma1 < sigma2:
            return sigma1
        return -sigma2

    def calculate_shear_failure_force(self) -> float:
        """
        Calculate the shear force Vfail that would result in matboard shear failure.
        """
        return min(cs.calculate_shear_failure_force(self.tau) for _, _, cs in self.cross_sections)


def main():
    bridge = Bridge.from_yaml("design0.yaml")
    loads = bridge.load_points(200)
    #loads = bridge.load_train(960)
    forces = bridge.reaction_forces(loads)

    sfd = bridge.make_sfd(forces)
    bmd = bridge.make_bmd(sfd)
    phi = bridge.make_curvature_diagram(bmd)

    print("Tensile Mfail:", bridge.calculate_tensile_failure_moment())
    print("Compressive Mfail:", bridge.calculate_compressive_failure_moment())
    print("Matboard Vfail:", bridge.calculate_shear_failure_force())

    x = np.arange(0, bridge.length + 1, 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.scatter([f[0] for f in forces], [f[1] for f in forces])
    ax1.set_title("Loads")
    ax2.plot(x, sfd)
    ax2.set_title("Shear Force")
    ax3.plot(x, bmd)
    ax3.set_title("Bending Moment")
    ax4.plot(x, phi)
    ax4.set_title("Curvature")
    plt.show()

    bridge.cross_sections[0][2].visualize(plt.gca())
    plt.show()


if __name__ == "__main__":
    main()
