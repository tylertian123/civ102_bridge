import numpy as np
import math
import itertools
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union
from matplotlib import patches, pyplot as plt
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


LocalBuckling = namedtuple("LocalBuckling", ("two_edge", "one_edge", "linear_stress", "shear"))


class CrossSection:
    """
    A class to hold and compute properties of cross sections.
    """

    ALL_NAMED_RECTS = {} # type: Dict[str, Dict[str, Rect]]

    def __init__(self, values: Dict[str, Any]) -> None:
        self.name = values["name"]
        CrossSection.ALL_NAMED_RECTS[self.name] = {}
        # Geometry consists of a list of rectangles in the form of [x, y, width, height]
        self.geometry = [self.parse_rect(rect) for rect in values["geometry"]]
        self.min_b_height = values.get("minBHeight")
        # An array of glued components, with form (geom, b)
        # Where geom is a list of rectangles that make up the component and b is the glue area
        self.glued_components = [([self.parse_rect(r) for r in c["pieces"]], c["glueArea"])
            for c in values["gluedComponents"]]
        # Specifications of the scenarios for all the plate buckling modes
        self.local_buckling = LocalBuckling(two_edge=[self.parse_rect(r) for r in values["localBuckling"].get("twoEdge", [])],
            one_edge=[self.parse_rect(r) for r in values["localBuckling"].get("oneEdge", [])],
            linear_stress=[self.parse_rect(r) for r in values["localBuckling"].get("linearStress", [])],
            # Shear buckling is more complicated and consists of tuples of (rect, min_b_height)
            # This is because b might vary over the depth of the rect since b is calculated for the whole structure
            shear=[(self.parse_rect(d["piece"]), d["minBHeight"]) for d in values["localBuckling"].get("shear", [])])
        self.diaphragm_distance = values.get("diaphragmDistance", math.inf)

        # Compute properties
        self.ytop = max(y + h for _, y, _, h in self.geometry)
        self.ybot = min(y for _, y, _, h in self.geometry)
        self.area = sum(w * h for _, _, w, h in self.geometry)
        self.ybar = sum(w * h * (y + h / 2) for _, y, w, h in self.geometry) / self.area
        # Parallel axis theorem: sum wh^3/12 + Ad^2
        self.i = sum(w * h ** 3 / 12 + w * h * (y + h / 2 - self.ybar) ** 2 for _, y, w, h in self.geometry)

    def parse_rect(self, rect: Union[str, Rect]) -> Rect:
        """
        Parse a rect from the YAML into a proper Rect (list or tuple of [x, y, width, height]).

        If rect is already a tuple or a list, it will be returned directly.
        If it's a string, then it will be interpreted as one of the two following syntaxes:

        name:[x, y, width, height]

        where x, y, width, height are the rect location and dimensions, and name is the name given to the rect, or

        cross_section:name[w=wstart:wstop, h=hstart:hstop]:newname

        where cross_section is the optional name of another cross section, name is the name of the rect to slice,
        wstart and wstop are the start and end of the width-wise (horizontal) slice (relative to the x value of the rect),
        hstart and hstop are the start and end of the height-wise (vertical) slice (relative to the y value of the rect),
        and newname is an optional new name to give to this rect.

        Both w and h type slices can be used together, or only one or none can be used. If wstart/wstop/hstart/hstop
        begin with a star, they are to be interpreted as starting from the far edge (like how Python list slicing works
        with negative indices).
        wstart and hstart can be omitted and default to 0. wstop and hstop can be omitted and default to the width
        and height of the rect to be sliced respectively.
        """
        # Return rect if already the right type
        if isinstance(rect, list) or isinstance(rect, tuple):
            return rect
        # Direct lookup if the rect is just a name
        if rect.isalnum():
            return CrossSection.ALL_NAMED_RECTS[self.name][rect]

        try:
            # Find the opening and closing brace of the slice, and slice out the slice
            open_idx = rect.index("[")
            close_idx = rect.index("]")
            content = rect[open_idx + 1:close_idx]
            rect_name = rect[:open_idx]
            # Regular named rect syntax
            if rect_name.endswith(":"):
                rect_name = rect_name[:-1]
                rect = [float(i) for i in content.split(",")]
                CrossSection.ALL_NAMED_RECTS[self.name][rect_name] = rect
                return rect
            # Slice
            else:
                # Look up the rect name and make a copy to modify
                if rect_name.isalnum():
                    new_rect = CrossSection.ALL_NAMED_RECTS[self.name][rect_name].copy()
                else:
                    cs, r = rect_name.split(":")
                    new_rect = CrossSection.ALL_NAMED_RECTS[cs][r].copy()
                for s in content.split(","):
                    s = s.strip()
                    if not s:
                        continue
                    slice_type, slice_range = s.split("=")
                    start, stop = slice_range.split(":")
                    start_starred = start.startswith("*")
                    if start_starred:
                        start = start[1:]
                    stop_starred = stop.startswith("*")
                    if stop_starred:
                        stop = stop[1:]

                    # Convert start to a number, with default starting at 0
                    start = float(start) if start else 0
                    # Convert stop to a number, with default stopping at the full width/height
                    # since this number will vary, it's assigned later
                    stop = float(stop) if stop else None
                    # Width slice
                    if slice_type == "w":
                        # Default for stop is the end of the rect
                        if stop is None:
                            stop = new_rect[2]
                        # If start and stop are starred, make them relative to the end
                        if start_starred:
                            start += new_rect[2]
                        if stop_starred:
                            stop += new_rect[2]
                        # Shrink from the right first
                        new_rect[2] = stop
                        # Shrink from the left, shifting x to the right and reducing width
                        new_rect[0] += start
                        new_rect[2] -= start
                    elif slice_type == "h":
                        # Same logic as above
                        if stop is None:
                            stop = new_rect[3]
                        if start_starred:
                            start += new_rect[3]
                        if stop_starred:
                            stop += new_rect[3]
                        # Shrink from the top
                        new_rect[3] = stop
                        # Shrink from the bottom
                        new_rect[1] += start
                        new_rect[3] -= start
                    else:
                        raise ValueError(f"Unknown slice type: {slice_type}")
                # Find the tail where a new name is assigned to the new rect
                tail = rect[close_idx + 2:]
                if tail:
                    if not tail.isalnum():
                        raise ValueError(f"Invalid name: {tail}")
                    CrossSection.ALL_NAMED_RECTS[self.name][tail] = new_rect
                return new_rect
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid syntax: {rect}") from e
        except KeyError as e:
            raise ValueError(f"Unknown name: {rect[:open_idx]}") from e

    def calculate_b(self, y0: float, above: Optional[bool] = None) -> float:
        """
        Calculate the total cross-sectional width b at depth y0.

        above specifies boundary behaviour. If it's true, then when y0 is at a boundary, the piece's width
        is only counted if it's above y0; if it's false, then the piece's width is only counted if it's below y0.
        If it's None, then the lesser of the two will be returned.
        """
        # Default calculates both above and below y0 and takes the minimum
        if above is None:
            return min(self.calculate_b(y0, True), self.calculate_b(y0, False))
        # Consider the piece if y0 is in the middle of it
        # That is, y + h should be above y0, and y should be below y0
        # Note float comparisions...
        return sum(w for _, y, w, h in self.geometry
                if floatgeq(y + h, y0) and floatleq(y, y0)
                # And then consider whether the rest of the piece is above or below y0
                and ((above and y + h > y0) or (not above and y < y0)))

    def calculate_q(self, y0: float) -> float:
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
        # If minimum b depth is specified, also try this depth
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
        # The b of each piece is provided in the cross section specifications as the glue area
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
        # Upper and lower bound, for when bottom is in compression and when top is in compression respectively
        mfailu = math.inf
        mfaill = math.inf
        for _, y, w, h in self.local_buckling.two_edge:
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
        the piece is under constant flexural stress, or linearly varying flexural stress. The critical stress for
        linearly varying flexural stress is calculated using the same formula as if the stress was constant with
        maximum magnitude to be conservative (k = 0.425).

        Returns an upper bound and a lower bound. If the bending moment is higher than the upper bound or more
        negative than the lower bound then the structure will fail. Note that if the upper/lower bound does not
        exist for this mode then math.inf will be returned.
        """
        mfailu = math.inf
        mfaill = math.inf
        for _, y, w, h in self.local_buckling.one_edge:
            # sigma = (0.425pi^2*E)/(12(1-nu^2))(t/b)^2 = My/I => M = sigma*I/y
            # Handle both constant and linearly varying flexural stresses
            # The piece could be either vertical or horizontal
            t = min(w, h)
            b = max(w, h)
            sigma = (0.425 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * (t / b) ** 2
            # Handle cases where the piece is above the centroid and the piece is below the centroid
            if y + h > self.ybar:
                # If the piece is entirely above centroid, then it's only under compression for positive bending moment
                # Use the top edge as y for max stress
                mfail = self.i * sigma / (y + h - self.ybar)
                mfailu = min(mfailu, mfail)
            if y < self.ybar:
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
        for _, y, w, h in self.local_buckling.linear_stress:
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
        for (_, y, w, h), min_b_height in self.local_buckling.shear:
            # tau = (5pi^2E)/(12(1 - nu^2))((t/h)^2 + (t/a)^2)
            # For a vertical plate, thickness t is just the width
            tau = (5 * math.pi ** 2 * e) / (12 * (1 - nu ** 2)) * ((w / h) ** 2 + (w / self.diaphragm_distance) ** 2)

            # tau = VQ/Ib => V = tau*Ib/Q
            # Consider 2 depths: Where Q is maximized, and where b is minimized (minimize b/Q)
            # To maximize Q, depth should be chosen to be as close to ybar as possible
            # First take the max of ybar and y to get either ybar if it's above the bottom or y if ybar is below the bottom
            # and then take the min of the last result and the top y value to clamp it again
            depth = min(max(self.ybar, y), y + h)
            # Calculate the fraction b/Q at both the depth closest to the centroidal axis and the supplied min b depth
            bq = self.calculate_b(depth) / self.calculate_q(depth)
            if min_b_height is not None:
                bq = min(bq, self.calculate_b(min_b_height) / self.calculate_q(min_b_height))
            v = tau * self.i * bq
            vfail = min(vfail, v)
        return vfail

    def visualize(self, ax: Axes, show_glued_components: bool = False, show_local_buckling: bool = False) -> None:
        """
        Draw the cross section onto a matplotlib plot to visualize it.
        """
        # Draw all the base rectangles
        for x, y, w, h in self.geometry:
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="k", facecolor="none"))
        # Show the different glued components in different colours
        if show_glued_components:
            for (geom, _), color in zip(self.glued_components, itertools.cycle("bgcmy")):
                for x, y, w, h in geom:
                    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none"))
        if show_local_buckling:
            # Show the components that can undergo each local buckling mode
            # Only one of the rectangles should be labelled so there won't be duplicate legend entires
            label = "Two-edge Buckling"
            for x, y, w, h in self.local_buckling.two_edge:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:orange", edgecolor="none", label=label))
                label = None
            label = "One-edge Buckling"
            for x, y, w, h in self.local_buckling.one_edge:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:olive", edgecolor="none", label=label))
                label = None
            label = "Linear-stress buckling"
            for x, y, w, h in self.local_buckling.linear_stress:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:brown", edgecolor="none", label=label))
                label = None
            label = "Shear buckling"
            for (x, y, w, h), _ in self.local_buckling.shear:
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, facecolor="tab:pink", edgecolor="none", label=label))
                label = None
        # Find the minimum and maximum x and y value and adjust the plot's x and y range accordingly
        xmin = min(x for x, _, _, _ in self.geometry)
        xmax = max(x + w for x, _, w, _ in self.geometry)
        ymin = min(y for _, y, _, _ in self.geometry)
        ymax = max(y + h for _, y, _, h in self.geometry)
        ax.set_xlim(xmin - 10, xmax + 10)
        ax.set_ylim(ymin - 10, ymax + 10)
        # Show centroid
        ax.axhline(self.ybar, c="r", label="Centroid")
        # Set aspect ratio between x and y to be equal so the shape is preserved
        ax.set_aspect("equal")
        ax.legend(loc="best")


class Bridge:
    """
    Main class for holding bridge geometry and material constants and calculating loads.
    """

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

        self.thickness = values["bridge"]["material"]["thickness"] # Thickness of matboard
        self.max_area = values["bridge"]["material"]["maxArea"] # Total area of matboard we have

        # Construct cross sections
        # Cross sections are tuples of (start, stop, cs) where cs is a CrossSection object
        self.cross_sections = [] # type: List[Tuple[int, int, CrossSection]]
        named_cross_sections = {}
        # Handle named cross section lookup
        for d in values["bridge"]["crossSections"]:
            if "geometry" in d:
                cs = CrossSection(d)
                named_cross_sections[d["name"]] = cs
            else:
                if "name" in d:
                    cs = named_cross_sections[d["name"]]
                else:
                    raise ValueError("Cross section must have name or geometry")
            self.cross_sections.append((d["start"], d["stop"], cs))
    
    @classmethod
    def from_yaml(cls, file) -> "Bridge":
        """
        Construct bridge from the input YAML file.
        """
        return Bridge(yaml.load(file, Loader))
    
    def elevation_view(self, ax: Axes) -> None:
        """
        Draw the elevation view onto a matplotlib axis.
        """
        ccycle = plt.get_cmap("tab10")
        miny = 0
        maxy = 0
        for i, (start, stop, cs) in enumerate(self.cross_sections):
            c = ccycle(i)
            label = cs.name
            for _, y, _, h in cs.geometry:
                ax.add_patch(patches.Rectangle((start, y), stop - start, h, edgecolor=c, facecolor="none", label=label))
                label= None
                miny = min(miny, y)
                maxy = max(maxy, y + h)
        ax.set_xlim(self.cross_sections[0][0] - 50, self.cross_sections[-1][1] + 50)
        ax.set_ylim(miny - 50, maxy + 50)
        ax.set_aspect("equal")
        ax.legend(loc="best")

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
        # Accumulate sfd to find bmd
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
        # Keep track of the right cross section
        i = 0
        cs = self.cross_sections[i]
        phi = []
        for x in range(len(bmd)):
            # Move to the next cross section if necessary
            while x >= cs[1]:
                i += 1
                cs = self.cross_sections[i]
            # Curvature phi = M / EI
            phi.append(bmd[x] / (self.e * cs[2].i))
        return np.array(phi)
    
    def max_sfd_train(self, step: Optional[int] = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the maximum shear force at every point from all possible train locations.
        """
        # Upper and lower bound of shear force
        upper = np.zeros((self.length + 1,))
        lower = np.zeros((self.length + 1,))

        # Case 1: Full train is loaded exactly in between the two supports
        train_len = self.train_wheel_edge_distance * 6 + self.train_wheel_distance * 3 + self.train_car_distance * 2
        case1 = self.supports[1] - (self.supports[1] - self.supports[0] - train_len) / 2
        # Case 2: The far right side of the train is against the far right side of the bridge
        case2 = self.length
        # Create and loop through locations
        locations = np.arange(case1, case2 + 1, step)
        for location in locations:
            # Make the sfd for this loading location
            sfd = self.make_sfd(self.reaction_forces(self.load_train(location)))
            # Update upper and lower limits
            upper = np.maximum(upper, sfd)
            lower = np.minimum(lower, sfd)
        return upper, lower
    
    def max_bmd_train(self, step: Optional[int] = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the maximum bending moment at every point from all possible train locations.
        """
        # Same logic as above
        upper = np.zeros((self.length + 1,))
        lower = np.zeros((self.length + 1,))
        # Case 1: Full train is loaded exactly in between the two supports
        train_len = self.train_wheel_edge_distance * 6 + self.train_wheel_distance * 3 + self.train_car_distance * 2
        case1 = self.supports[1] - (self.supports[1] - self.supports[0] - train_len) / 2
        # Case 2: The far right side of the train is against the far right side of the bridge
        case2 = self.length
        locations = np.arange(case1, case2 + 1, step)
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
            m1 = self.sigmat * cs.i / (cs.ybar - cs.ytop)
            # When the bottom is in tension, i.e. positive moment
            m2 = self.sigmat * cs.i / (cs.ybar - cs.ybot)
            # Same for the entire cross section
            upper.extend([m2] * (stop - start))
            lower.extend([m1] * (stop - start))
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
            m1 = self.sigmac * cs.i / (cs.ytop - cs.ybar)
            # When the bottom is in compression, i.e. negative curvature
            # ybot - ybar is negative so this will come out negative
            m2 = self.sigmac * cs.i / (cs.ybot - cs.ybar)
            upper.extend([m1] * (stop - start))
            lower.extend([m2] * (stop - start))
        return np.array(upper), np.array(lower)

    def calculate_matboard_vfail(self) -> np.ndarray:
        """
        Calculate the shear force Vfail that would result in matboard shear failure.
        """
        vfail = []
        # For each cross section, compute a single Vfail
        for start, stop, cs in self.cross_sections:
            vfail.extend([cs.calculate_matboard_vfail(self.tau)] * (stop - start))
        return np.array(vfail)
    
    def calculate_glue_vfail(self) -> np.ndarray:
        """
        Calculate the shear force Vfail that causes shear failure of the glue.
        """
        vfail = []
        for start, stop, cs in self.cross_sections:
            vfail.extend([cs.calculate_glue_vfail(self.glue_tau)] * (stop - start))
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
            # Same Mfail for the entire cross section just like Vfail
            upper.extend([u] * (stop - start))
            lower.extend([l] * (stop - start))
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
            upper.extend([u] * (stop - start))
            lower.extend([l] * (stop - start))
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
            upper.extend([u] * (stop - start))
            lower.extend([l] * (stop - start))
        return np.array(upper), np.array(lower)
    
    def calculate_buckling_vfail(self) -> np.ndarray:
        """
        Calculate the shear force Vfail that causes shear buckling.
        """
        vfail = []
        for start, stop, cs in self.cross_sections:
            vfail.extend([cs.calculate_buckling_vfail(self.e, self.nu)] * (stop - start))
        return np.array(vfail)
    
    def calculate_shear_fos(self, sfd: np.ndarray, fail_shear: List[np.ndarray]) -> float:
        """
        Compute the Factor of Safety between the internal shear force in sfd and the failure shear forces
        in fail_shear.
        """
        # Find the minimum failure shear force by taking the minimum of all the failure shear forces
        min_fail_shear = [min(v) for v in zip(*fail_shear)]
        # FoS is the minimum of the limit divided by the load it's carrying at all points
        return min((cap / abs(dem) for cap, dem in zip(min_fail_shear, sfd) if dem != 0), default=math.inf)
    
    def calculate_moment_fos(self, bmd: np.ndarray, fail_moment_upper: List[np.ndarray], fail_moment_lower: List[np.ndarray]) -> float:
        """
        Compute the Factor of Safety between the internal bending moment in bmd and the failure bending moments
        in fail_moment_upper and fail_moment_lower.
        """
        # Find the max and min failure bending moment
        upper = [min(m) for m in zip(*fail_moment_upper)]
        lower = [max(m) for m in zip(*fail_moment_lower)]
        # FoS needs to consider both when bending moment is positive and when it's negative
        return min(min((cap / dem for cap, dem in zip(upper, bmd) if dem > 0), default=math.inf),
            min((cap / dem for cap, dem in zip(lower, bmd) if dem < 0), default=math.inf))
    
    def calculate_tangential_deviation(self, phi: np.ndarray, a: int, b: int) -> float:
        """
        Calculate the tangential deviation of b from a tangent drawn at a, delta_BA.
        """
        # Compute area and xbar by numerical integration
        # Cut curvature diagram from a to b
        phi_sub = phi[a:b]
        # Calculate values for x * phi from a to b
        xphi = np.multiply(np.arange(a, b), phi_sub)
        # Use trapezoidal sums
        area = np.trapz(phi_sub)
        xbar = np.trapz(xphi) / area
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

    def matboard_area(self) -> float:
        """
        Get an estimate of the total area of matboard used for this bridge design.

        Note that numbers do not account for folding and assumes that matboard can be perfectly utilized so this
        number is likely not very accurate, but may serve as a rough guide.
        """
        # Get an estimate by considering the area of each cross section and the length of that cross section
        # and then dividing the volume by the thickness
        return sum(cs.area * (stop - start) for start, stop, cs in self.cross_sections) / self.thickness
