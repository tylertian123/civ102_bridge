import numpy as np
import math
import itertools
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union
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

# Rectangles are defined by [x, y, width, height]
Rect = Union[List[float], Tuple[float, float, float, float]]

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


BucklingModes = namedtuple("BucklingModes", ("two_edge", "one_edge", "linear_stress", "shear"))


# A class to hold and compute properties of cross sections
class CrossSection:
    def __init__(self, values: Dict[str, Any]) -> None:
        # Geometry consists of a list of rectangles in the form of [x, y, width, height]
        self.geometry = list(values["geometry"].values())
        self.min_b_height = values["minBHeight"]
        # For each item listed in the pieces, it could either be a rectangle or the name of another piece
        rect_list = lambda l: [self.parse_rect(values["geometry"], rect) for rect in l]
        # An array of glued components, with form (geom, b)
        # Where geom is a list of rectangles that make up the component and b is the glue area
        self.glued_components = [(rect_list(c["pieces"]), c["glueArea"]) for c in values["gluedComponents"]]
        # Specifications of the scenarios for all the plate buckling modes
        self.buckling_modes = BucklingModes(two_edge=rect_list(values["bucklingModes"].get("twoEdge", [])),
            one_edge=rect_list(values["bucklingModes"].get("oneEdge", [])),
            linear_stress=rect_list(values["bucklingModes"].get("linearStress", [])),
            shear=rect_list(values["bucklingModes"].get("shear", [])))
        self.diaphragm_distance = values.get("diaphragmDistance", math.inf)

        # Compute properties
        self.ytop = max(y + h for _, y, _, h in self.geometry)
        self.ybot = min(y for _, y, _, h in self.geometry)
        self.area = sum(w * h for _, _, w, h in self.geometry)
        self.ybar = sum(w * h * (y + h / 2) for _, y, w, h in self.geometry) / self.area
        # Parallel axis theorem: sum wh^3/12 + Ad^2
        self.i = sum(w * h ** 3 / 12 + w * h * (y + h / 2 - self.ybar) ** 2 for _, y, w, h in self.geometry)
    
    def parse_rect(self, known_rects: Dict[str, Rect], rect: Union[str, Rect]) -> Rect:
        """
        Parse a rect from the YAML into a proper Rect (list or tuple of [x, y, width, height]).

        If rect is already a tuple or a list, it will be returned directly.
        If it's a string, then it will be interpreted with the following syntax:

        name[w=wstart:wstop, h=hstart:hstop]:newname

        where name is the name of an existing rect in known_rects, wstart and wstop are the start and end of the
        width-wise (horizontal) slice (relative to the x value of the rect), hstart and hstop are the start and end of
        of the height-wise (vertical) slice (relative to the y value of the rect), and newname is an optional new name
        to give to this rect, and if given, the new rect produced by the slice will be stored in known_rects.

        Both w and h type slices can be used together, or only one or none can be used. If wstart/wstop/hstart/hstop
        are negative, they are to be interpreted as starting from the far edge (like how Python list slicing works).
        wstart and hstart can be omitted and default to 0. wstop and hstop can be omitted and default to the width
        and height of the rect to be sliced respectively.
        """
        if isinstance(rect, list) or isinstance(rect, tuple):
            return rect
        if rect.isalnum():
            return known_rects[rect]

        try:
            open_idx = rect.index("[")
            close_idx = rect.index("]")
            rect_slice = rect[open_idx + 1:close_idx]
            new_rect = known_rects[rect[:open_idx]].copy()
            for s in rect_slice.split(","):
                s = s.strip()
                slice_type, slice_range = s.split("=")
                start, stop = slice_range.split(":")
                start = float(start) if start else 0
                stop = float(stop) if stop else None
                # Width slice
                if slice_type == "w":
                    # Default for stop is the end of the rect
                    if stop is None:
                        stop = new_rect[2]
                    if stop < 0:
                        stop += new_rect[2]
                    if start < 0:
                        start += new_rect[2]
                    # Shrink from the right first
                    new_rect[2] = stop
                    # Shrink from the left
                    new_rect[0] += start
                    new_rect[2] -= start
                elif slice_type == "h":
                    if stop is None:
                        stop = new_rect[3]
                    if stop < 0:
                        stop += new_rect[3]
                    if start < 0:
                        start += new_rect[3]
                    # Shrink from the top
                    new_rect[3] = stop
                    # Shrink from the bottom
                    new_rect[1] += start
                    new_rect[3] -= start
                else:
                    raise ValueError(f"Unknown slice type: {slice_type}")
            tail = rect[close_idx + 2:]
            if tail:
                if not tail.isalnum():
                    raise ValueError(f"Invalid name: {tail}")
                known_rects[tail] = new_rect
            return new_rect
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid syntax: {rect}") from e
        except KeyError as e:
            raise ValueError(f"Unknown name: {rect[:open_idx]}") from e
    
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
    
    def calculate_matboard_vfail(self, tau: float) -> float:
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
    
    def calculate_glue_vfail(self, tau: float) -> float:
        """
        Calculate the shear force Vfail that causes shear failure of the glue.
        
        tau is the shear strength of the glue.
        """
        # Calculate for each glued component
        # V = tau*Ib/Q for each component like above
        # The b of each piece is provided in the cross section specifications
        # The Q is equal to the absolute value of the sum of area times centroidal distance
        return tau * self.i * min(b / abs(sum(w * h * (y + h / 2 - self.ybar) for _, y, w, h in geom))
            for geom, b in self.glued_components)

    def calculate_two_edge_mfail(self, e: float, nu: float) -> Tuple[float, float]:
        """
        Calculate the bending moment Mfail that causes thin-plate buckling where both edges are restrained and
        the piece is under constant flexural stress.

        Returns an upper bound and a lower bound. If the bending moment is higher than the upper bound or more
        negative than the lower bound then the structure will fail. Note that if the upper/lower bound does not
        exist for this mode then math.inf will be returned.
        """
        mfailu = math.inf
        mfaill = math.inf
        for _, y, w, h in self.buckling_modes.two_edge:
            # sigma = (4pi^2*E)/(12(1-nu^2))(t/b)^2 = My/I => M = sigma*I/y
            # Since the piece is under constant flexural stress it must have a constant y value so it must be horizontal
            # This means that thickness t is the piece's height and width b is the piece's width
            sigma = (4 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * (h / w) ** 2
            # Assume piece is either entirely above or entirely below centroid
            if y + h > self.ybar:
                # If the piece is entirely above centroid, then it's only under compression for positive bending moment
                # Use the top edge as y for max stress
                mfail = self.i * sigma / (y + h - self.ybar)
                mfailu = min(mfailu, mfail)
            else:
                # If the piece is entirely below centroid, then it's only under compression for negative bending moment
                mfail = self.i * sigma / (self.ybar - y)
                mfaill = min(mfaill, mfail)
        return mfailu, -mfaill
    
    def calculate_one_edge_mfail(self, e: float, nu: float) -> Tuple[float, float]:
        """
        Calculate the bending moment Mfail that causes thin-plate buckling where only one edge is restrained and
        the piece is under constant flexural stress.

        Returns an upper bound and a lower bound. If the bending moment is higher than the upper bound or more
        negative than the lower bound then the structure will fail. Note that if the upper/lower bound does not
        exist for this mode then math.inf will be returned.
        """
        mfailu = math.inf
        mfaill = math.inf
        for _, y, w, h in self.buckling_modes.one_edge:
            # sigma = (0.425pi^2*E)/(12(1-nu^2))(t/b)^2 = My/I => M = sigma*I/y
            # Since the piece is under constant flexural stress it must have a constant y value so it must be horizontal
            # This means that thickness t is the piece's height and width b is the piece's width
            sigma = (0.425 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * (h / w) ** 2
            # Assume piece is either entirely above or entirely below centroid
            if y > self.ybar:
                # If the piece is entirely above centroid, then it's only under compression for positive bending moment
                # Use the top edge as y for max stress
                mfail = self.i * sigma / (y + h - self.ybar)
                mfailu = min(mfailu, mfail)
            else:
                # If the piece is entirely below centroid, then it's only under compression for negative bending moment
                mfail = self.i * sigma / (self.ybar - y)
                mfaill = min(mfaill, mfail)
        return mfailu, -mfaill

    def calculate_linear_stress_mfail(self, e: float, nu: float) -> Tuple[float, float]:
        """
        Calculate the bending moment Mfail that causes thin-plate buckling where both edges are restrained and
        the piece is under linear flexural stress.

        Returns an upper bound and a lower bound. If the bending moment is higher than the upper bound or more
        negative than the lower bound then the structure will fail. Note that if the upper/lower bound does not
        exist for this mode then math.inf will be returned.
        """
        mfailu = math.inf
        mfaill = math.inf
        for _, y, w, h in self.buckling_modes.linear_stress:
            # sigma = (4pi^2*E)/(12(1-nu^2))(t/b)^2 = My/I => M = sigma*I/y
            # If the piece is under linear flexural stress, then it must be vertical
            # Thickness t is the piece's width and width b is the piece's height
            # Split into two cases: Above centroid and below centroid (split piece between both if piece is on both sides)
            if y + h > self.ybar:
                # New height is y + h - ybar, the amount that's above ybar
                sigma = (6 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * (w / (y + h - self.ybar)) ** 2
                # Above centroid is under compression only for positive bending moment
                # Maximum stress occurs at top edge
                mfail = self.i * sigma / (y + h - self.ybar)
                mfailu = min(mfailu, mfail)
            if y < self.ybar:
                # New height is ybar - y
                sigma = (6 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * (w / (self.ybar - y)) ** 2
                # Below centroid is under compression only for negative bending moment
                # Max stress occurs at bottom edge
                mfail = self.i * sigma / (self.ybar - y)
                mfaill = min(mfaill, mfail)
        return mfailu, -mfaill
    
    def calculate_buckling_vfail(self, e: float, nu: float) -> float:
        """
        Calculate the shear force Vfail that would cause shear buckling.
        """
        vfail = math.inf
        for _, y, w, h in self.buckling_modes.shear:
            # tau = (5pi^2E)/(12(1 - nu^2))((t/h)^2 + (t/a)^2)
            # t used in equation is the shorter of the two, and h is the longer of the two
            t = min(w, h)
            h2 = max(w, h)
            tau = (5 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * ((t / h2) ** 2 + (t / self.diaphragm_distance) ** 2)

            # tau = VQ/Ib => V = tau*Ib/Q, and the width b is the same as the t above
            # To get max shear stress Q needs to be maximized so the depth should be chosen to be as close to ybar as possible
            # First take the max of ybar and y to get either ybar if it's above the bottom or y if ybar is below the bottom
            # and then take the min of the last result and the top y value to clamp it again
            v = tau * self.i * t / self.calculate_q(min(max(self.ybar, y), y + h))
            vfail = min(vfail, v)
        return vfail

    def visualize(self, ax: Axes, show_glued_components: bool = False, show_buckling_modes: bool = False) -> None:
        """
        Draw the cross section onto a matplotlib plot to visualize it.
        """
        for x, y, w, h in self.geometry:
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="k", facecolor="none"))
        if show_glued_components:
            for (geom, _), color in zip(self.glued_components, itertools.cycle("bgcmy")):
                for x, y, w, h in geom:
                    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none"))
        if show_buckling_modes:
            label = "Two-edge Buckling"
            for x, y, w, h in self.buckling_modes.two_edge:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:orange", edgecolor="none", label=label))
                label = None
            label = "One-edge Buckling"
            for x, y, w, h in self.buckling_modes.one_edge:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:olive", edgecolor="none", label=label))
                label = None
            label = "Linear-stress buckling"
            for x, y, w, h in self.buckling_modes.linear_stress:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:brown", edgecolor="none", label=label))
                label = None
            label = "Shear buckling"
            for x, y, w, h in self.buckling_modes.shear:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:pink", edgecolor="none", label=label))
                label = None
        xmin = min(x for x, _, _, _ in self.geometry)
        xmax = max(x + w for x, _, w, _ in self.geometry)
        ymin = min(y for _, y, _, _ in self.geometry)
        ymax = max(y + h for _, y, _, h in self.geometry)
        ax.set_xlim(xmin - 10, xmax + 10)
        ax.set_ylim(ymin - 10, ymax + 10)
        ax.axhline(self.ybar, c="r", label="Centroid")
        ax.set_aspect("equal")
        ax.legend(loc="best")


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
    def from_yaml(cls, file) -> "Bridge":
        return Bridge(yaml.load(file, Loader))

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
        while x < len(shear):
            # For every point load reached, add its value
            while i < len(loads) and x <= loads[i][0] and x + 1 > loads[i][0]:
                s += loads[i][1]
                i += 1
            shear[x] = s
            # Increment x value
            x += 1
        shear = np.array(shear)
        if abs(shear[-1]) > 1e-7:
            print(f"Warning: Nonzero final shear! Final value {shear[-1]} is {abs(shear[-1]) / max(abs(shear.max()), abs(shear.min()))} times the maximum absolute shear.")
        return shear

    def make_bmd(self, sfd: np.ndarray) -> np.ndarray:
        """
        Compute the Bending Moment Diagram from the Shear Force Diagram.
        """
        bmd = [0] * len(sfd)
        moment = 0
        for x in range(len(sfd)):
            moment += sfd[x]
            bmd[x] = moment
        bmd = np.array(bmd)
        if abs(bmd[-1]) > 1e-7:
            print(f"Warning: Nonzero final bending moment! Final value {bmd[-1]} is {abs(bmd[-1]) / max(abs(bmd.max()), abs(bmd.min()))} times the maximum absolute shear.")
        return bmd

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
    
    def max_sfd_train(self, step: Optional[int] = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the maximum shear force at every point from all possible train locations.
        """
        upper = np.zeros((self.length + 1,))
        lower = np.zeros((self.length + 1,))
        # Sample locations from the front of the train being at the left end of the bridge,
        # to when the front of the train is at the length of the bridge plus the length of the train
        lend = self.train_wheel_edge_distance
        rend = self.length + 4 * self.train_wheel_edge_distance + 3 * self.train_wheel_distance + 2 * self.train_car_distance
        locations = np.arange(lend, rend + 1, step)
        for location in locations:
            sfd = self.make_sfd(self.reaction_forces(self.load_train(location)))
            upper = np.maximum(upper, sfd)
            lower = np.minimum(lower, sfd)
        return upper, lower
    
    def max_bmd_train(self, step: Optional[int] = 4) -> Tuple[np.ndarray, np.ndarray]:
        upper = np.zeros((self.length + 1,))
        lower = np.zeros((self.length + 1,))
        # Same logic as above
        lend = self.train_wheel_edge_distance
        rend = self.length + 4 * self.train_wheel_edge_distance + 3 * self.train_wheel_distance + 2 * self.train_car_distance
        locations = np.arange(lend, rend + 1, step)
        for location in locations:
            bmd = self.make_bmd(self.make_sfd(self.reaction_forces(self.load_train(location))))
            upper = np.maximum(upper, bmd)
            lower = np.minimum(lower, bmd)
        return upper, lower

    def calculate_tensile_mfail(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bending moment Mfail that would cause a matboard flexural tensile failure.

        Returns two numpy arrays, consisting of an upper and lower bound (lower bound will be negative).
        If the bending moment is higher than the upper bound or more negative than the lower bound then the
        structure will fail.
        """
        upper = []
        lower = []
        for start, stop, cs in self.cross_sections:
            # sigma = M*y/I => M = sigma*I/y
            # sigma1 is for when the top is in tension, i.e. negative moment
            # sigma1 will come out negative since ybar < ytop
            sigma1 = self.sigmat * cs.i / (cs.ybar - cs.ytop)
            # When the bottom is in tension, i.e. positive moment
            sigma2 = self.sigmat * cs.i / (cs.ybar - cs.ybot)
            upper.extend([sigma2] * (stop - start + 1))
            lower.extend([sigma1] * (stop - start + 1))
        return np.array(upper), np.array(lower)
    
    def calculate_compressive_mfail(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bending moment Mfail that would cause a matboard flexural compressive failure.

        Returns two numpy arrays, consisting of an upper and lower bound (lower bound will be negative).
        If the bending moment is higher than the upper bound or more negative than the lower bound then the
        structure will fail.
        """
        # See logic above
        upper = []
        lower = []
        for start, stop, cs in self.cross_sections:
            # sigma1 is for when the top is in compression, i.e. positive curvature
            sigma1 = self.sigmac * cs.i / (cs.ytop - cs.ybar)
            # When the bottom is in compression, i.e. negative curvature
            # ybot - ybar is negative so this will come out negative
            sigma2 = self.sigmac * cs.i / (cs.ybot - cs.ybar)
            upper.extend([sigma1] * (stop - start + 1))
            lower.extend([sigma2] * (stop - start + 1))
        return np.array(upper), np.array(lower)

    def calculate_matboard_vfail(self) -> np.ndarray:
        """
        Calculate the shear force Vfail that would result in matboard shear failure.
        """
        vfail = []
        for start, stop, cs in self.cross_sections:
            vfail.extend([cs.calculate_matboard_vfail(self.tau)] * (stop - start + 1))
        return np.array(vfail)
    
    def calculate_glue_vfail(self) -> np.ndarray:
        """
        Calculate the shear force Vfail that causes shear failure of the glue.
        """
        vfail = []
        for start, stop, cs in self.cross_sections:
            vfail.extend([cs.calculate_glue_vfail(self.glue_tau)] * (stop - start + 1))
        return np.array(vfail)
    
    def calculate_two_edge_mfail(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bending moment Mfail that causes thin-plate buckling where both edges are restrained and
        the piece is under constant flexural stress.

        Returns two numpy arrays, consisting of an upper and lower bound (lower bound will be negative).
        If the bending moment is higher than the upper bound or more negative than the lower bound then the
        structure will fail.
        """
        upper = []
        lower = []
        for start, stop, cs in self.cross_sections:
            u, l = cs.calculate_two_edge_mfail(self.e, self.nu)
            upper.extend([u] * (stop - start + 1))
            lower.extend([l] * (stop - start + 1))
        return np.array(upper), np.array(lower)
    
    def calculate_one_edge_mfail(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bending moment Mfail that causes thin-plate buckling where only one edge is restrained and
        the piece is under constant flexural stress.

        Returns two numpy arrays, consisting of an upper and lower bound (lower bound will be negative).
        If the bending moment is higher than the upper bound or more negative than the lower bound then the
        structure will fail.
        """
        upper = []
        lower = []
        for start, stop, cs in self.cross_sections:
            u, l = cs.calculate_one_edge_mfail(self.e, self.nu)
            upper.extend([u] * (stop - start + 1))
            lower.extend([l] * (stop - start + 1))
        return np.array(upper), np.array(lower)
    
    def calculate_linear_stress_mfail(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the bending moment Mfail that causes thin-plate buckling where both edges are restrained and
        the piece is under linearly varying flexural stress.

        Returns two numpy arrays, consisting of an upper and lower bound (lower bound will be negative).
        If the bending moment is higher than the upper bound or more negative than the lower bound then the
        structure will fail.
        """
        upper = []
        lower = []
        for start, stop, cs in self.cross_sections:
            u, l = cs.calculate_linear_stress_mfail(self.e, self.nu)
            upper.extend([u] * (stop - start + 1))
            lower.extend([l] * (stop - start + 1))
        return np.array(upper), np.array(lower)
    
    def calculate_buckling_vfail(self) -> np.ndarray:
        """
        Calculate the shear force Vfail that causes shear buckling.
        """
        vfail = []
        for start, stop, cs in self.cross_sections:
            vfail.extend([cs.calculate_buckling_vfail(self.e, self.nu)] * (stop - start + 1))
        return np.array(vfail)
    
    def calculate_shear_fos(self, sfd: np.ndarray, fail_shear: List[np.ndarray]) -> float:
        """
        Compute the Factor of Safety between the internal shear force in sfd and the failure shear forces
        in fail_shear.
        """
        # Find the minimum shear failure force by taking the minimum of all the failure shear forces
        min_fail_shear = [min(v) for v in zip(*fail_shear)]
        return min((cap / abs(dem) for cap, dem in zip(min_fail_shear, sfd) if dem != 0), default=math.inf)
    
    def calculate_moment_fos(self, bmd: np.ndarray, fail_moment_upper: List[np.ndarray], fail_moment_lower: List[np.ndarray]) -> float:
        """
        Compute the Factor of Safety between the internal bending moment in bmd and the failure bending moments
        in fail_moment_upper and fail_moment_lower.
        """
        # Find the max and min failure bending moment
        upper = [min(m) for m in zip(*fail_moment_upper)]
        lower = [max(m) for m in zip(*fail_moment_lower)]
        return min(min((cap / dem for cap, dem in zip(upper, bmd) if dem > 0), default=math.inf),
            min((cap / dem for cap, dem in zip(lower, bmd) if dem < 0), default=math.inf))
    
    def calculate_tangential_deviation(self, phi: np.ndarray, a: int, b: int) -> float:
        """
        Calculate the tangential deviation of b from a tangent drawn at a, delta_BA.
        """
        area = sum(phi[x] for x in range(a, b + 1))
        xbar = sum(x * phi[x] for x in range(a, b + 1)) / area
        # Use second moment area theorem
        return area * (b - xbar)

    def calculate_deflection(self, phi: np.ndarray, x: int) -> float:
        """
        Calculate the deflection at a location x from the left end of the bridge.

        x should be in-between the two supports. Otherwise, the behaviour is undefined.
        """
        delta_ba = self.calculate_tangential_deviation(phi, self.supports[0], self.supports[1])
        # By similar triangles, delta_BA / AB = Delta_XA' / AX
        # Delta_XA is composed of the tangential deviation delta_XA and the true deflection
        delta_xa = self.calculate_tangential_deviation(phi, self.supports[0], x)
        return (delta_ba / (self.supports[1] - self.supports[0])) * (x - self.supports[0]) - delta_xa

