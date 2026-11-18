"""Maak een scatterplot van de volledige gemeten pendulumbeweging."""

from pathlib import Path

import click
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from animate_pendulum import DATA_DIR, create_axes, load_measurement


def plot_color(_ctx, _param, value):
    """Accepteer Matplotlib-kleurnamen en hexkleuren."""
    try:
        return mcolors.to_rgb(value)
    except ValueError as exc:
        raise click.BadParameter("Gebruik een kleurnaam of een hexkleur zoals '#2be7ff'.") from exc


def create_scatter(times, positions, origin, invert_y, background, point_colors, glow):
    fig, ax = create_axes(positions, invert_y, "Gemeten beweging van het dubbelpendulum", origin)
    fig.set_facecolor(background)
    ax.set_facecolor(background)
    luminance = np.dot(background, (0.2126, 0.7152, 0.0722))
    foreground = "#151b25" if luminance > 0.5 else "#edf2fa"

    ax.set_axisbelow(True)
    ax.grid(color=foreground, alpha=0.14)
    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    ax.title.set_color(foreground)
    for spine in ax.spines.values():
        spine.set_color(foreground)
        spine.set_alpha(0.3)

    # Latere metingen krijgen een sterkere kleur. Overlap maakt dichte delen helderder.
    strength = 0.38 + 0.62 * times / times[-1] if times[-1] > 0 else np.ones_like(times)
    for point, color in enumerate(point_colors, start=1):
        is_origin = origin == f"p{point}"
        x, y = ([0], [0]) if is_origin else positions[:, point - 1].T
        base_size = 70 if is_origin else 6
        if glow:
            for scale, opacity in ((5, 0.045), (2.5, 0.10)):
                ax.scatter(x, y, s=base_size * scale, color=color,
                           alpha=opacity * glow, linewidths=0, zorder=2)

        label = f"P{point} (oorsprong)" if is_origin else f"P{point}"
        if is_origin:
            ax.scatter(x, y, s=base_size, color=color, linewidths=0, label=label, zorder=4)
        else:
            rgba = np.empty((len(times), 4))
            rgba[:, :3] = color
            rgba[:, 3] = strength * 0.68
            ax.scatter(x, y, s=base_size, color=rgba, linewidths=0, label=label, zorder=3)

    ax.legend(facecolor=background, edgecolor=foreground, labelcolor=foreground, markerscale=1.5)
    return fig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("measurement", metavar="MEETBESTAND", type=click.Path(path_type=Path))
@click.option("--origin", type=click.Choice(["p1", "p2"], case_sensitive=False),
              default="p1", show_default=True, help="Kies het punt op de oorsprong.")
@click.option("--invert-y", is_flag=True, help="Laat de Y-as naar beneden toenemen.")
@click.option("--background", default="#101725", show_default=True, callback=plot_color,
              help="Achtergrondkleur.")
@click.option("--p1-color", default="#f2f5ff", show_default=True, callback=plot_color,
              help="Kleur van P1.")
@click.option("--p2-color", default="#46dfff", show_default=True, callback=plot_color,
              help="Kleur van P2.")
@click.option("--p3-color", default="#ff9a5b", show_default=True, callback=plot_color,
              help="Kleur van P3.")
@click.option("--glow", type=click.FloatRange(0, 1), default=0.55, show_default=True,
              help="Sterkte van de gloed. Kies 0 om de gloed uit te zetten.")
@click.option("--save", "output", type=click.Path(dir_okay=False, path_type=Path),
              help="Sla de scatterplot op als .png.")
@click.option("--show/--no-show", default=None,
              help="Open het plotvenster. Standaard alleen zonder --save.")
def main(measurement, origin, invert_y, background, p1_color, p2_color, p3_color, glow, output, show):
    """Toon alle geldige posities uit MEETBESTAND (.mqa of CSV)."""
    if not measurement.is_absolute() and not measurement.exists():
        measurement = DATA_DIR / measurement
    if not measurement.is_file():
        raise click.ClickException(f"Meetbestand niet gevonden: {measurement}")
    if show is None:
        show = output is None
    if output is None and not show:
        raise click.UsageError("Gebruik --save bij --no-show.")
    if output is not None and output.suffix.lower() != ".png":
        raise click.UsageError("Geef bij --save een bestandsnaam met extensie .png op.")

    times, positions = load_measurement(measurement, origin)
    fig = create_scatter(times, positions, origin, invert_y, background,
                         (p1_color, p2_color, p3_color), glow)
    try:
        if output is not None:
            try:
                fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
            except (OSError, ValueError) as exc:
                raise click.ClickException(f"Kan de scatterplot niet opslaan: {exc}") from exc
            click.echo(f"Opgeslagen: {output}")
        if show:
            plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
