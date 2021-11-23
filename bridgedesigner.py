import click
import calculate
from typing import TextIO
from matplotlib import pyplot as plt

@click.group()
@click.argument("bridge_yaml", type=click.File("r", encoding="utf-8"))
@click.pass_context
def main_cli(ctx, bridge_yaml: TextIO):
    print("Welcome to CIV102 BridgeDesigner Pro Edition!")

    ctx.obj = calculate.Bridge.from_yaml(bridge_yaml)

@main_cli.command()
@click.option("--visualize/--no-visualize", "-v", default=False, help="Show the cross-section in a GUI window.")
@click.option("--glue/--no-glue", "-g", default=False, help="Show glued components.")
@click.option("--buckling/--no-buckling", "-b", default=False, help="Show buckling types.")
@click.pass_context
def geometry(ctx, visualize: bool, glue: bool, buckling: bool):
    bridge = ctx.obj # type: calculate.Bridge
    for i, (start, stop, cs) in enumerate(bridge.cross_sections):
        label = f"Cross section #{i + 1} (start: {start}, stop: {stop})"
        print(label)
        print(f"\tytop:\t{cs.ytop}\n\tybot:\t{cs.ybot}\n\tA:\t{cs.area}\n\tybar:\t{cs.ybar}\n\tI:\t{cs.i}")
        if visualize:
            cs.visualize(plt.gca(), show_glued_components=glue, show_buckling_modes=buckling)
            plt.gcf().canvas.set_window_title(label)
            plt.gca().set_title(label)
            plt.show()

if __name__ == "__main__":
    main_cli(obj=None)
