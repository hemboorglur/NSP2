import matplotlib
matplotlib.use("WebAgg")
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# instellen van een webserver waar matplotlib de grafieken kan weergeven
plt.rcParams["webagg.address"] = "0.0.0.0"
plt.rcParams["webagg.port"] = 8988
plt.rcParams["webagg.open_in_browser"] = False



# CSV-bestand inlezen
df = pd.read_csv("NSP2/bombbba.mqa")

# Posities van de drie punten
x1, y1 = df["Point #1.X"], df["Point #1.Y"]
x2, y2 = df["Point #2.X"], df["Point #2.Y"]
x3, y3 = df["Point #3.X"], df["Point #3.Y"]

time = df["Time"]

# Figuur aanmaken
fig, ax = plt.subplots()

ax.set_xlim(df[["Point #1.X", "Point #2.X", "Point #3.X"]].min().min(),
            df[["Point #1.X", "Point #2.X", "Point #3.X"]].max().max())

ax.set_ylim(df[["Point #1.Y", "Point #2.Y", "Point #3.Y"]].min().min(),
            df[["Point #1.Y", "Point #2.Y", "Point #3.Y"]].max().max())

ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Drie bewegende punten
point1, = ax.plot([], [], "ro", label="Punt 1")
point2, = ax.plot([], [], "bo", label="Punt 2")
point3, = ax.plot([], [], "go", label="Punt 3")

ax.legend()

# Animatiefunctie
def update(frame):
    point1.set_data([x1.iloc[frame]], [y1.iloc[frame]])
    point2.set_data([x2.iloc[frame]], [y2.iloc[frame]])
    point3.set_data([x3.iloc[frame]], [y3.iloc[frame]])

    ax.set_title(f"Tijd: {time.iloc[frame]:.2f} s")

    return point1, point2, point3

# Animatie maken
ani = FuncAnimation(
    fig,
    update,
    frames=len(df),
    interval=50,
    blit=False
)

plt.show()