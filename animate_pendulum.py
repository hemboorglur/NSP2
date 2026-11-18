"""Bekijk, animeer of exporteer de gemeten beweging van een dubbelpendulum."""

from pathlib import Path
from subprocess import CalledProcessError

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers

DATA_DIR = Path(__file__).resolve().parent / "datafiles"
COORDINATES = [f"Point #{point}.{axis}" for point in (1, 2, 3) for axis in "XY"]
COLUMNS = ["Time", *COORDINATES]


def load_measurement(path: Path, origin: str = "p1") -> tuple[np.ndarray, np.ndarray]:
    """Lees tijdstempels en zet het gekozen referentiepunt in elk meetframe op de oorsprong."""
    try:
        with path.open(encoding="utf-8-sig") as stream:
            header = next((line for line in stream if line.strip()), "")
            delimiter = max((",", "\t", ";"), key=header.count)
        data = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise click.ClickException(f"Kan {path} niet lezen: {exc}") from exc

    data.columns = data.columns.str.strip()
    missing = [column for column in COLUMNS if column not in data]
    if missing:
        raise click.ClickException(f"Ontbrekende kolommen: {', '.join(missing)}")

    values = data[COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        raise click.ClickException("Het bestand bevat geen volledige numerieke meetrijen.")
    if not valid.all():
        click.echo(f"Onvolledige of ongeldige meetrijen overgeslagen: {(~valid).sum()}.", err=True)

    values = values[valid]
    times = values[:, 0]
    if np.any(np.diff(times) < 0):
        raise click.ClickException("De waarden in Time moeten in oplopende volgorde staan.")

    positions = values[:, 1:].reshape(-1, 3, 2)
    reference = positions[:, int(origin[1]) - 1, None, :]
    return times - times[0], positions - reference


def create_axes(positions: np.ndarray, invert_y: bool, title: str, origin: str):
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    lower = positions.min(axis=(0, 1))
    upper = positions.max(axis=(0, 1))
    padding = max(float(np.max(upper - lower)) * 0.05, 1.0)
    ax.set_xlim(lower[0] - padding, upper[0] + padding)
    ax.set_ylim(lower[1] - padding, upper[1] + padding)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel=f"X relatief aan {origin.upper()}",
           ylabel=f"Y relatief aan {origin.upper()}", title=title)
    ax.grid(alpha=0.25)

    return fig, ax


def create_animation(
    times: np.ndarray, positions: np.ndarray, fps: int, invert_y: bool, origin: str = "p1"
):
    """Selecteer gemeten frames op een regelmatig afspeelrooster."""
    playback = np.minimum(np.arange(int(np.ceil(times[-1] * fps)) + 1) / fps, times[-1])
    right = np.searchsorted(times, playback).clip(max=len(times) - 1)
    left = np.maximum(right - 1, 0)
    frames = np.where(playback - times[left] <= times[right] - playback, left, right)

    fig, ax = create_axes(positions, invert_y, "Dubbelpendulum", origin)

    first_arm, = ax.plot([], [], "-o", color="tab:blue", linewidth=2)
    second_arm, = ax.plot([], [], "-o", color="tab:orange", linewidth=2)
    clock = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})

    def update(index):
        x, y = positions[index].T
        first_arm.set_data(x[:2], y[:2])
        second_arm.set_data(x[1:], y[1:])
        clock.set_text(f"t = {times[index]:.3f} s")
        return first_arm, second_arm, clock

    animation = FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, repeat=False, cache_frame_data=False
    )
    return fig, animation


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("measurement", metavar="MEETBESTAND", type=click.Path(path_type=Path))
@click.option("--fps", type=click.IntRange(1, 240), default=30, show_default=True,
              help="Aantal animatiebeelden per seconde.")
@click.option("--save", "output", type=click.Path(dir_okay=False, path_type=Path),
              help="Sla de animatie op als .mp4 (vereist FFmpeg).")
@click.option("--show/--no-show", default=None,
              help="Open het afspeelvenster. Standaard alleen zonder --save.")
@click.option("--origin", type=click.Choice(["p1", "p2"], case_sensitive=False),
              default="p1", show_default=True, help="Kies het punt op de oorsprong.")
@click.option("--invert-y", is_flag=True, help="Laat de Y-as naar beneden toenemen.")
def main(
    measurement: Path, fps: int, output: Path | None, show: bool | None,
    invert_y: bool, origin: str,
):
    """Bekijk MEETBESTAND (.mqa of CSV), zo nodig uit de map datafiles."""
    if not measurement.is_absolute() and not measurement.exists():
        measurement = DATA_DIR / measurement
    if not measurement.is_file():
        raise click.ClickException(f"Meetbestand niet gevonden: {measurement}")

    if show is None:
        show = output is None
    if output is None and not show:
        raise click.UsageError("Gebruik --save bij --no-show.")
    if output is not None:
        if output.suffix.lower() != ".mp4":
            raise click.UsageError("Geef bij --save een bestandsnaam met extensie .mp4 op.")
        if not writers.is_available("ffmpeg"):
            raise click.ClickException("FFmpeg is nodig voor MP4-export en is niet gevonden.")

    times, positions = load_measurement(measurement, origin)
    fig, animation = create_animation(times, positions, fps, invert_y, origin)
    try:
        if output is not None:
            try:
                animation.save(output, writer=FFMpegWriter(fps=fps, codec="h264"))
            except (OSError, RuntimeError, ValueError, CalledProcessError) as exc:
                raise click.ClickException(f"Kan de animatie niet opslaan: {exc}") from exc
            click.echo(f"Opgeslagen: {output}")
        if show:
            plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
