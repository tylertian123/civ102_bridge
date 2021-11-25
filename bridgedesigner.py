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
@click.option("--buckling/--no-buckling", "-b", default=False, help="Show local buckling types.")
@click.pass_context
def geometry(ctx, visualize: bool, glue: bool, buckling: bool):
    """
    Visualize the bridge geometry and calculate cross-sectional properties.
    """
    bridge = ctx.obj # type: calculate.Bridge
    # Analyze each cross section
    for i, (start, stop, cs) in enumerate(bridge.cross_sections):
        label = f"Cross section #{i + 1} (start: {start}, stop: {stop})"
        print(label)
        print(f"\tytop:\t{cs.ytop}mm\n\tybot:\t{cs.ybot}mm\n\tybar:\t{cs.ybar}mm\n\tA:\t{cs.area}mm^2\n\tI:\t{cs.i}mm^4")
        if visualize:
            # Visualize the cross section on the global current axis
            cs.visualize(plt.gca(), show_glued_components=glue, show_local_buckling=buckling)
            # Set window and chart title
            plt.gcf().canvas.set_window_title(label)
            plt.gca().set_title(label)
            plt.show()
    # Estimate matboard area used
    area = bridge.matboard_area()
    print(f"Estimate of total matboard area used: {round(area)}mm^2 out of {bridge.max_area}mm^2 ({area / bridge.max_area * 100:.2f}%)")
    if area > bridge.max_area:
        print("Max matboard area exceeded!")


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

    # Special case for max train loading
    if load_type == "train" and load_amount == "max":
        sfd = bridge.max_sfd_train()
        bmd = bridge.max_bmd_train()
    else:
        # Compute load forces
        if load_type == "point":
            forces = bridge.load_points(float(load_amount))
        else:
            forces = bridge.load_train(float(load_amount))
        # Compute reaction forces, sfd, bmd
        forces = bridge.reaction_forces(forces)
        print("Forces:", forces)
        sfd = bridge.make_sfd(forces)
        bmd = bridge.make_bmd(sfd)
    
    # Set up and label plots
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
        # Additional plots needed since there is both a max and a min for max train loading
        ax1.plot(x, sfd[0], label="Max Shear")
        ax1.plot(x, sfd[1], label="Min Shear")
        ax2.plot(x, bmd[0], label="Max Bending Moment")
        ax2.plot(x, bmd[1], label="Min Bending Moment")
    else:
        ax1.plot(x, sfd, label="Shear")
        ax2.plot(x, bmd, label="Bending Moment")
    
    # Helper function; if match is in keys, then plot the upper and lower bound and print out its value
    def plot_fail(keys: List[str], match: Iterable[str], label: str, upper: np.ndarray, lower: np.ndarray):
        if "all" in keys or any(m in keys for m in match):            
            print("Failure values for ", label, ":", sep="")
            for i, (start, stop, _) in enumerate(bridge.cross_sections):
                # Find the value of the bounds somewhat arbitrarily by taking the value at the middle
                print(f"\tCross section #{i + 1} (start: {start}, stop: {stop}):\t({lower[(start - stop) // 2]}, {upper[(start - stop) // 2]})")

            # Choose the right axis, ax1 for shear and ax2 for bending moment
            ax = ax1 if keys is vfail else ax2
            # Plot both with the same colour but only label 1
            p = ax.plot(x, upper, label=label)
            ax.plot(x, lower, c=p[0].get_c())
    
    tmu, tml = bridge.calculate_tensile_mfail()
    cmu, cml = bridge.calculate_compressive_mfail()
    oemu, oeml = bridge.calculate_one_edge_mfail()
    temu, teml = bridge.calculate_two_edge_mfail()
    lsmu, lsml = bridge.calculate_linear_stress_mfail()
    plot_fail(mfail, ("t", "tensile"), "Tensile Failure", tmu, tml)
    plot_fail(mfail, ("c", "compressive"), "Compressive Failure", cmu, cml)
    plot_fail(mfail, ("buckling", "oe", "oneedge"), "One-edge Buckling Failure", oemu, oeml)
    plot_fail(mfail, ("buckling", "te", "twoedge"), "Two-edge Buckling Failure", temu, teml)
    plot_fail(mfail, ("buckling", "ls", "linearstress"), "Linear-stress Buckling Failure", lsmu, lsml)

    mv = bridge.calculate_matboard_vfail()
    gv = bridge.calculate_glue_vfail()
    bv = bridge.calculate_buckling_vfail()
    plot_fail(vfail, ("m", "matboard"), "Matboard Failure", mv, -mv)
    plot_fail(vfail, ("g", "glue"), "Glue Failure", gv, -gv)
    plot_fail(vfail, ("b", "buckling"), "Shear Buckling Failure", bv, -bv)

    # Calculate FoS
    fail_shear = [mv, gv, bv]
    fail_moment_upper = [tmu, cmu, oemu, temu, lsmu]
    fail_moment_lower = [tml, cml, oeml, teml, lsml]
    if load_amount != "max":
        fos_shear = bridge.calculate_shear_fos(sfd, fail_shear)
        fos_moment = bridge.calculate_moment_fos(bmd, fail_moment_upper, fail_moment_lower)
    else:
        # For max train loading, consider both the max and min loading
        fos_shear = min(bridge.calculate_shear_fos(sfd[0], fail_shear), bridge.calculate_shear_fos(sfd[1], fail_shear))
        fos_moment = min(bridge.calculate_moment_fos(bmd[0], fail_moment_upper, fail_moment_lower), bridge.calculate_moment_fos(bmd[1], fail_moment_upper, fail_moment_lower))
    print("Factors of Safety:")
    print("\tShear:\t", fos_shear, sep="")
    print("\tBending Moment:\t", fos_moment, sep="")
    if fos_shear < 1:
        print("Bridge fails by shear!")
    if fos_moment < 1:
        print("Bridge fails by bending!")

    # Calculate failure P value by simply multiplying the current load by the FoS
    if load_type == "point":
        print("Failure P:", float(load_amount) * fos_moment)
    if load_amount != "max":
        print("Midspan deflection:", bridge.calculate_deflection(bridge.make_curvature_diagram(bmd), bridge.length // 2))

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main_cli(obj=None)
