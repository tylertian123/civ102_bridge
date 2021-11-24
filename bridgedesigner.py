import click
import calculate
import numpy as np
from typing import Iterable, List, TextIO
from matplotlib import pyplot as plt

@click.group()
@click.argument("bridge_yaml", type=click.File("r", encoding="utf-8"))
@click.pass_context
def main_cli(ctx, bridge_yaml: TextIO):
    ctx.obj = calculate.Bridge.from_yaml(bridge_yaml)

@main_cli.command()
@click.option("--visualize/--no-visualize", "-v", default=False, help="Show the cross-section in a GUI window.")
@click.option("--glue/--no-glue", "-g", default=False, help="Show glued components.")
@click.option("--buckling/--no-buckling", "-b", default=False, help="Show buckling types.")
@click.pass_context
def geometry(ctx, visualize: bool, glue: bool, buckling: bool):
    """
    Visualize the bridge geometry and calculate cross-sectional properties.
    """
    bridge = ctx.obj # type: calculate.Bridge
    for i, (start, stop, cs) in enumerate(bridge.cross_sections):
        label = f"Cross section #{i + 1} (start: {start}, stop: {stop})"
        print(label)
        print(f"\tytop:\t{cs.ytop}mm\n\tybot:\t{cs.ybot}mm\n\tybar:\t{cs.ybar}mm\n\tA:\t{cs.area}mm^2\n\tI:\t{cs.i}mm^4")
        if visualize:
            cs.visualize(plt.gca(), show_glued_components=glue, show_buckling_modes=buckling)
            plt.gcf().canvas.set_window_title(label)
            plt.gca().set_title(label)
            plt.show()


@main_cli.command()
@click.argument("load_type", type=click.Choice(["train", "point"], case_sensitive=False))
@click.argument("load_amount", type=str)
@click.option("--mfail", "-m", multiple=True, type=click.Choice(["all", "t", "tensile", "c", "compressive", "buckling", "oe", "oneedge", "te", "twoedge", "ls", "linearstress"]))
@click.option("--vfail", "-v", multiple=True, type=click.Choice(["all", "m", "matboard", "g", "glue", "b", "buckling"]))
@click.pass_context
def load(ctx, load_type: str, load_amount: str, mfail: List[str], vfail: List[str]):
    """
    Load the bridge and assess loading diagrams and failure loads.

    LOAD_TYPE is the type of loading (train or point.)
    If using point loading, LOAD_AMOUNT is the magnitude of each point load; if using train loading, it is the
    location of the front (right side) of the train relative to the left side of the bridge. Use 'max' to get
    the maximum possible shear and bending moment across all possible train positions.
    """
    bridge = ctx.obj # type: calculate.Bridge

    if load_type == "train" and load_amount == "max":
        sfd = bridge.max_sfd_train()
        bmd = bridge.max_bmd_train()
    else:
        if load_type == "point":
            forces = bridge.load_points(float(load_amount))
        else:
            forces = bridge.load_train(float(load_amount))
        forces = bridge.reaction_forces(forces)
        sfd = bridge.make_sfd(forces)
        bmd = bridge.make_bmd(sfd)
    
    x = np.arange(0, bridge.length + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.canvas.set_window_title("SFD and BMD")
    ax1.set_title("SFD")
    ax1.set_xlabel("Location (mm)")
    ax1.set_ylabel("Shear Force (N)")
    ax1.axhline(0, c="k")
    ax2.set_title("BMD")
    ax2.set_xlabel("Location (mm)")
    ax2.set_ylabel("Bending Moment (Nmm)")
    ax2.axhline(0, c="k")

    if load_amount == "max":
        ax1.plot(x, sfd[0], label="Max Shear")
        ax1.plot(x, sfd[1], label="Min Shear")
        ax2.plot(x, bmd[0], label="Max Bending Moment")
        ax2.plot(x, bmd[1], label="Min Bending Moment")
    else:
        ax1.plot(x, sfd, label="Shear")
        ax2.plot(x, bmd, label="Bending Moment")
    
    def plot_fail(keys: List[str], match: Iterable[str], label: str, fn):
        if "all" in keys or any(m in keys for m in match):
            fail_vals = fn()
            if len(fail_vals) == 2:
                upper, lower = fail_vals
            else:
                upper = fail_vals
                lower = -fail_vals
            
            print("Failure values for ", label, ":", sep="")
            for i, (start, stop, _) in enumerate(bridge.cross_sections):
                print(f"\tCross section #{i + 1} (start: {start}, stop: {stop}):\t({lower[(start - stop) // 2]}, {upper[(start - stop) // 2]})")

            ax = ax1 if keys is vfail else ax2
            p = ax.plot(x, upper, label=label)
            ax.plot(x, lower, c=p[0].get_c())
    
    plot_fail(mfail, ("t", "tensile"), "Tensile Failure", bridge.calculate_tensile_mfail)
    plot_fail(mfail, ("c", "compressive"), "Compressive Failure", bridge.calculate_compressive_mfail)
    plot_fail(mfail, ("buckling", "oe", "oneedge"), "One-edge Buckling Failure", bridge.calculate_one_edge_mfail)
    plot_fail(mfail, ("buckling", "te", "twoedge"), "Two-edge Buckling Failure", bridge.calculate_two_edge_mfail)
    plot_fail(mfail, ("buckling", "ls", "linearstress"), "Linear-stress Buckling Failure", bridge.calculate_linear_stress_mfail)

    plot_fail(vfail, ("m", "matboard"), "Matboard Failure", bridge.calculate_matboard_vfail)
    plot_fail(vfail, ("g", "glue"), "Glue Failure", bridge.calculate_glue_vfail)
    plot_fail(vfail, ("b", "buckling"), "Shear Buckling Failure", bridge.calculate_buckling_vfail)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main_cli(obj=None)
