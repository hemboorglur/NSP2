import matplotlib
matplotlib.use("WebAgg")

from copy import copy

import click
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import math as m
import numpy as np
import pandas as pd

# instellen van een webserver waar matplotlib 
# de grafieken kan weergeven
plt.rcParams["webagg.address"] = "0.0.0.0"
plt.rcParams["webagg.port"] = 8988
plt.rcParams["webagg.open_in_browser"] = False



def data_folder():
    data_dir= Path(__file__).resolve().parents[2] / "datafiles"
    # Ook bruikbaar als deze werkkopie buiten de NSP2-repository staat.
    if not data_dir.is_dir():
        data_dir= Path.home() / "Documents/NSP2/NSP2/datafiles"
    return data_dir


def df_load(measurement):
    return pd.read_csv(data_folder() / measurement)


def p1_tracking(df):
    """Geef de geldige P1-posities en hun covariantie over de hele opname."""
    coords= df[[f"Point #{i}.{axis}" for i in (1, 2, 3) for axis in "XY"]]
    herhaald= coords.eq(coords.shift()).all(axis=1)
    xy= coords[["Point #1.X", "Point #1.Y"]]
    geldig= np.isfinite(xy).all(axis=1) & ~herhaald
    punten= xy.loc[geldig]
    return punten


def df_points_lab(dataframe: pd.DataFrame) -> dict[tuple[str: list[float], str: list[float]]]:
    """ Load all coordinates from DataFrame, converts to labframe and appends to dictonary

    Load coordinates from DataFrame and converts to labframe by subtracting the coordinate-
    value of P1, setting this point as the origin. Finally, appends the new coordinates to 
    a dictonary in the following format: {x:[..], y:[..]}

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        points: Dictonary with coordinates in the labframe
    """
    df, points= dataframe, {}
    for i in [1,2,3]: points[i]= (
        (df[f"Point #{i}.X"]-df["Point #1.X"]).to_numpy(dtype=float),
        (df[f"Point #{i}.Y"]-df["Point #1.Y"]).to_numpy(dtype=float))
    return points

def df_points_hng(dataframe: pd.DataFrame) -> dict[tuple[str: list[float], str: list[float]]]:
    """ Load all coordinates from DataFrame, converts to hinge-frame and appends to dictonary

    Load coordinates from DataFrame and converts to hinhe-frame by subtracting the coordinate-
    value of P2, setting this point as the origin. Finally, appends the new coordinates to 
    a dictonary in the following format: {x:[..], y:[..]}

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        points: Dictonary with coordinates in the labframe
    """
    df, points_hinge= dataframe, {}
    for i in [1,2,3]: 
        points_hinge[i]= (
            (df[f"Point #{i}.X"]-df["Point #2.X"]).to_numpy(dtype=float), 
            (df[f"Point #{i}.Y"]-df["Point #2.Y"]).to_numpy(dtype=float))
    return points_hinge

def df_points_pot(dataframe: pd.DataFrame) -> dict[str: list[float]]:
    """ Load all coordinates from DataFrame, converts to potential-frame and appends to dictonary

    Load coordinates from DataFrame and converts to hinhe-frame by subtracting the coordinate-
    value of P2, setting this point as the origin. Finally, appends the new coordinates to 
    a dictonary in the following format: {y:[..]}

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        points: Dictonary with coordinates in the labframe
    """
    df, points_potential= dataframe, {}
    for i in [1,2,3]: 
        points_potential[i]= (
            df[f"Point #{i}.Y"]).to_numpy(dtype=float)
    return points_potential

def load_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Load time of frames from DataFrame

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        DataFrame of all times
    """
    df= dataframe
    return df["Time"].to_numpy(dtype=float)


def unwrap_angle(angle):
    # Na een ontbrekende marker begint een nieuw stuk van de meting.
    result= np.full_like(angle, np.nan)
    valid= np.flatnonzero(np.isfinite(angle))
    for frames in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
        result[frames]= np.unwrap(angle[frames])
    return result


class calculations():
    
    def __init__(self, measurement) -> None:
        """Initialize measurement data and physical properties of a double pendulum.
        
        Loads the measurement data, extracts the positions and time values, and
        defines the lengths and masses of both pendulum arms.
        
        Args:
            measurement: Number identifying the measurement to load.
        """
        self.measurement= Path(measurement).stem
        self.df= df_load(measurement)
        self.p1_punten= p1_tracking(self.df)
        self.afwijkingen= self.p1_punten - self.p1_punten.mean()
        self.afwijkingen.columns= ["dx_px", "dy_px"]
        self.sigma_x, self.sigma_y= self.afwijkingen.std()
        # Dezelfde geschatte plaatsfout voor elk van de drie markers in deze meting.
        self.puntfout= {punt: (self.sigma_x, self.sigma_y) for punt in (1, 2, 3)}
        self.points_lab= df_points_lab(self.df)
        self.points_hng= df_points_hng(self.df)
        self.points_pot= df_points_pot(self.df)
        self.time= load_time(self.df)
        
        self.L1, self.L2= 225/1000, 220/1000
        self.m1, self.m2= 338/1000, 418/1000
        self.g= 9.81 # m/s**2
        self.set_cm(0.5, 0.5)

    def set_cm(self, fraction_A, fraction_B):
        """Set each centre of mass as a fraction of arm length from its own pivot."""
        if not (0 <= fraction_A <= 1 and 0 <= fraction_B <= 1):
            raise ValueError("Kies voor beide armen een fractie tussen 0 en 1.")
        self.r_cmA= fraction_A * self.L1
        self.r_cmB= fraction_B * self.L2
        
        
    def angle_A(self) -> np.ndarray:
        """ Calculate angle between arm A and the origin using the np.arctan2 function
        
        Returns:
            An array of angles in radians from -π to π 
        """
        return unwrap_angle(np.arctan2(-self.points_hng[1][0], self.points_hng[1][1]))

    def angle_B(self) -> np.ndarray:
        """ Calculate angle between arm B and the origin using the np.arctan2 function
        
        Returns:
            An array of angles in radians from -π to π 
        """
        return unwrap_angle(np.arctan2(self.points_hng[3][0], -self.points_hng[3][1]))
        
    def mid_time(self) -> np.ndarray:
        return (self.time[:-1] + self.time[1:]) / 2
    
    def mid_angle(self, angle: str) -> np.ndarray:
        theta = getattr(self, f"angle_{angle}")()
        return (theta[:-1] + theta[1:]) / 2

    def dt(self) -> np.ndarray:
        """ Gives the difference in time between every two frames using the np.diff function

        Returns:
            An array of time differential (dt) values
        """
        return np.diff(self.time)

    def dpsi(self, angle: np.ndarray) -> np.ndarray:
        """ Gives the difference in angle between every two frames using the np.diff function

        Args:
            angle: Array containing the angle in radians of the arm at every frame
        Returns:
            An array of angle differential (dpsi) values
        """
        return np.diff(getattr(self, f"angle_{angle}")())

    def angular_v(self, angle: np.ndarray) -> np.ndarray:
        """ Calculates the angular velocity using the formula omega = dt / dpsi

        Args:
            angle: Array containing the angle in radians of the arm at every frame
        Returns:
            An array of angular velocities of every frame
        """
        return self.dpsi(angle) / self.dt()
    
    def linear_v_A(self):
        return abs(self.r_cmA * self.angular_v("A"))
    
    def linear_v_B(self):
        omega_A, omega_B= self.angular_v("A"), self.angular_v("B")
        psi_A, psi_B= self.mid_angle("A"), self.mid_angle("B")
        return np.sqrt(self.L1**2 * omega_A**2 + self.r_cmB**2 * omega_B**2 + 2*self.L1*self.r_cmB*omega_A*omega_B*np.cos(psi_A-psi_B))
        
    def E_kin(self):
        return 1/2 * (self.m1 * self.linear_v_A()**2 + self.m2 * self.linear_v_B()**2)
    
    def E_pot(self):
        # Hoogte boven de stand waarin beide armen recht omlaag hangen.
        rise_A= 1 - np.cos(self.angle_A())
        h_A= self.r_cmA * rise_A
        h_B= self.L1 * rise_A + self.r_cmB * (1 - np.cos(self.angle_B()))
        return self.g * (self.m1 * h_A + self.m2 * h_B)
    
    def mid_E_pot(self):
        return (self.E_pot()[:-1] + self.E_pot()[1:]) / 2

    def E_pos(self):
        return self.E_kin() + self.mid_E_pot()
    
    def gemiddelde_energie(self, E, venster):
        """Tijdgewogen gemiddelde binnen elk aaneengesloten stuk."""
        if not np.isfinite(venster) or venster <= 0:
            raise ValueError("Kies een positief tijdvenster in seconden.")
        time= self.mid_time()
        gemiddeld= np.full_like(E, np.nan)
        geldig= np.flatnonzero(np.isfinite(E))
        for stuk in np.split(geldig, np.flatnonzero(np.diff(geldig) > 1) + 1):
            if stuk.size == 0:
                continue
            # Elke energiewaarde hoort bij het interval tussen twee cameraframes.
            # Integreer E * dt en deel het verschil over een volledig venster door de duur.
            grenzen= self.time[stuk[0]:stuk[-1]+2]
            integraal= np.r_[0, np.cumsum(E[stuk] * np.diff(grenzen))]
            links= time[stuk] - venster / 2
            rechts= time[stuk] + venster / 2
            volledig= (links >= grenzen[0]) & (rechts <= grenzen[-1])
            begin= np.interp(links[volledig], grenzen, integraal)
            eind= np.interp(rechts[volledig], grenzen, integraal)
            gemiddeld[stuk[volledig]]= (eind - begin) / venster

        return gemiddeld

    def E_puntfout(self, venster):
        """Tel de bijdragen van de drie onafhankelijke markerfouten kwadratisch op."""
        variant= copy(self)
        som_kwadraten= np.zeros_like(self.mid_time())
        for punt in (1, 2, 3):
            for as_nr, sigma in enumerate(self.puntfout[punt]):
                energie= []
                for teken in (-1, 1):
                    # P2 is de oorsprong in points_hng. Verplaats je P2, dan
                    # verschuiven zowel P1−P2 als P3−P2 in tegengestelde zin.
                    variant.points_hng= self.points_hng.copy()
                    for eindpunt in (1, 3):
                        richting= int(punt == eindpunt) - int(punt == 2)
                        xy= list(self.points_hng[eindpunt])
                        xy[as_nr]= xy[as_nr] + richting * teken * sigma
                        variant.points_hng[eindpunt]= tuple(xy)
                    energie.append(variant.gemiddelde_energie(variant.E_pos(), venster))
                som_kwadraten += ((energie[1] - energie[0]) / 2)**2
        return np.sqrt(som_kwadraten)

    def E_cm_variations(self, stap=1, venster=None, bereken=None):
        """Geef per tijdstip de laagste, middelste stand (50%, 50%) en hoogste energie."""
        # stap in procentpunten. Neem ook 75% mee als de laatste stap korter is.
        if not np.isfinite(stap) or not 0 < stap <= 50:
            raise ValueError("Kies een stap groter dan 0 en maximaal 50 procentpunten.")
        if bereken is None:
            bereken= self.E_pos

        def energie():
            E= bereken()
            if venster is None:
                return E
            return self.gemiddelde_energie(E, venster)

        fractions= np.append(np.arange(25, 75, stap), 75) / 100
        old_A, old_B= self.r_cmA, self.r_cmB
        try:
            self.set_cm(0.5, 0.5)
            E_middle= energie()
            E_min= np.full_like(E_middle, np.inf)
            E_max= np.full_like(E_middle, -np.inf)

            # In E_pos hangen de bijdragen van r_cmA en r_cmB afzonderlijk
            # van hun eigen afstand af. Bereken de verandering door B één keer.
            changes_B= []
            for b in fractions:
                self.set_cm(0.5, b)
                changes_B.append(energie() - E_middle)

            for a in fractions:
                self.set_cm(a, 0.5)
                E_A= energie()
                for change_B in changes_B:
                    energy= E_A + change_B
                    E_min= np.minimum(E_min, energy)
                    E_max= np.maximum(E_max, energy)
        finally:
            self.r_cmA, self.r_cmB= old_A, old_B
        return E_min, E_middle, E_max

    def E_drag(self, Cd, rho=1.2):
        """Energieverlies aan luchtweerstand per tijdstap (J).
        b_A en b_B zijn de effectieve dieptes loodrecht op het bewegingsvlak (m).
        Cd is de aangenomen weerstandscoëfficiënt voor beide armen.
        """
        b_A= 0.005  # dikte arm A in m
        b_B= 0.014  # dikte arm B in m
        omega_A, omega_B= self.angular_v("A"), self.angular_v("B")

        # Arm A draait om een vaste as. Op afstand r is de snelheid r * omega_A.
        P_A= rho * Cd * b_A * self.L1**4 * abs(omega_A)**3 / 8

        # Arm B beweegt mee met het scharnier. Neem de snelheid dwars op arm B.
        v_hng= self.L1 * omega_A * np.cos(self.mid_angle("A") - self.mid_angle("B"))
        dr= self.L2 / 20
        P_B= np.zeros_like(omega_B)
        for i in range(20):
            r= (i + 0.5) * dr
            v= v_hng + r * omega_B
            P_B += 0.5 * rho * Cd * b_B * dr * abs(v)**3

        return (P_A + P_B) * self.dt()
        
    
    # def E_pot_alt(self):
    #     psi_A = self.mid_angle("A")
    #     psi_B = self.mid_angle("B")

    #     h_A = -self.r_cmA * np.cos(psi_A)
    #     h_B = -self.L1 * np.cos(psi_A) - self.r_cmB * np.cos(psi_B)

    #     return self.g * (self.m1 * h_A + self.m2 * h_B)

class plot():
    
    def __init__(self, measurement) -> None:
        self.data= measurement

        self.PLOT_DIR = Path("plots")
        self.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def plot_afwijkingen(metingen):
        """Bewaar en plot per meting de afwijkingen van P1 rond zijn eigen gemiddelde."""
        mapje= Path("plots")
        mapje.mkdir(exist_ok=True)
        reeksen= []
        fig, axes= plt.subplots(2, 1, sharex=True, figsize=(11, 7), layout="constrained")
        for meting in metingen:
            reeks= meting.afwijkingen.copy()
            reeks["tijd_s"]= meting.df.loc[reeks.index, "Time"]
            reeks["meting"]= meting.measurement
            reeksen.append(reeks)
            for ax, kolom in zip(axes, ("dx_px", "dy_px")):
                ax.plot(reeks.tijd_s, reeks[kolom], ".", ms=1.5, alpha=0.6,
                        label=meting.measurement.split("_")[-1])

        alle= pd.concat(reeksen, ignore_index=True)
        alle.to_csv(mapje / "P1_afwijkingen.csv", index=False)
        sigma= alle[["dx_px", "dy_px"]].std()
        for ax, kolom in zip(axes, sigma.index):
            ax.set_ylabel(f"{kolom[:2]} (px)")
            ax.set_title(f"Standaarddeviatie alle metingen: {sigma[kolom]:.3f} px")
        axes[0].legend(ncol=2, fontsize=8, markerscale=3, loc="upper left", bbox_to_anchor=(1, 1))
        axes[1].set_xlabel("Tijd sinds start opname (s)")
        fig.savefig(mapje / "P1_alle_afwijkingen.png", dpi=160)
        plt.show()
        plt.close(fig)
        return sigma

    def plot_P1(self):
        """Toon de X- en Y-positie van P1 over de hele opname."""
        d= self.data
        fig, axes= plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
        time= d.df.loc[d.p1_punten.index, "Time"]
        for ax, asnaam, sigma in zip(axes, "XY", (d.sigma_x, d.sigma_y)):
            positie= d.p1_punten[f"Point #1.{asnaam}"]
            ax.plot(time, positie, ".", markersize=1.5)
            ax.axhline(positie.mean(), color="grey", linestyle="--")
            ax.set_title(f"σ{asnaam.lower()} = {sigma:.3f} px")
            ax.set_ylabel(f"{asnaam} (px)")
        axes[-1].set_xlim(d.time[0], d.time[-1])
        axes[-1].set_xlabel("Opnametijd (s)")
        fig.suptitle(f"P1 · {d.measurement}")
        fig.savefig(self.PLOT_DIR / f"P1_{d.measurement}.png", dpi=160)
        plt.show()
        plt.close(fig)

    def plot_E_pot(self):
        plt.figure()
        plt.scatter(self.data.time, self.data.E_pot(), s=1)
    
        plt.xlabel("Time (s)")
        plt.ylabel("Potential energy (J)")
        plt.title("Potential energy")
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_pot_meting_test.png", dpi=300, bbox_inches="tight")
        plt.show()
           
    def plot_E_kin(self):
        plt.figure()
        plt.scatter(self.data.mid_time(), self.data.E_kin(), s= 1)
        plt.yscale("log")

        plt.xlabel("Time (s)")
        plt.ylabel("Kinetic energy (J)")
        plt.title("Kinetic energy")
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_kin_meting_test.png", dpi=300, bbox_inches="tight")
        plt.show()
        
    def _draw_cm(self, ax, stap):
        time= self.data.mid_time()
        E_min, E_middle, E_max= self.data.E_cm_variations(stap)
        ax.fill_between(time, E_min, E_max, color="#438dcc", alpha=0.70, linewidth=0,
                        label=f"Massamiddelpunten 25–75% (stap {stap:g} procentpunt)")
        ax.plot(time, E_middle, color="black", linewidth=0.35, alpha=0.70, label="Beide op 50%")
        ax.set_ylabel("Kinetische + potentiële energie (J)")
        ax.legend()
        ax.grid(alpha=0.25)
        return E_middle

    def plot_E_cm_variations(self, stap=1):
        fig, ax= plt.subplots(figsize=(10, 5))
        self._draw_cm(ax, stap)
        ax.set_xlabel("Opnametijd (s)")
        ax.set_title(self.data.measurement)
        fig.tight_layout()
        fig.savefig(self.PLOT_DIR / f"E_cm_{self.data.measurement}.png", dpi=180)
        plt.show()

    def plot_E_cm_met_drag(self, Cd=1.0, stap=1, rho=1.2, venster=1.0):
        """Plot energie, cumulatieve verliezen en het tijdgemiddelde bij de ingestelde cm-stand.
        venster is in seconden, stap in procentpunten voor de blauwe variatieband.
        """
        if not np.isfinite(venster) or venster <= 0:
            raise ValueError("Het gemiddelde heeft een positief tijdvenster in seconden nodig.")
        time= self.data.mid_time()
        E_kin= self.data.E_kin()
        E_pot= self.data.mid_E_pot()
        E= E_kin + E_pot
        # Luchtverlies is recht evenredig met Cd. Integreer één keer bij Cd=1.
        Cd_min, Cd_max= 0.8, 2.5
        drag= self.data.E_drag(1.0, rho=rho)
        totaal= np.full_like(E, np.nan)
        lucht_per_Cd= np.full_like(E, np.nan)

        geldig= np.flatnonzero(np.isfinite(E) & np.isfinite(drag))
        for stuk in np.split(geldig, np.flatnonzero(np.diff(geldig) > 1) + 1):
            if stuk.size == 0:
                continue
            totaal[stuk]= E[stuk[0]] - E[stuk]
            lucht_per_Cd[stuk]= np.r_[0, np.cumsum((drag[stuk[:-1]] + drag[stuk[1:]]) / 2)]

        E_min, _, E_max= self.data.E_cm_variations(stap, venster)
        gemiddeld= self.data.gemiddelde_energie(E, venster)

        fig, (boven, midden, fout_ax, onder)= plt.subplots(
            4, 1, figsize=(11, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 1, 3]})
        K_min, _, K_max= self.data.E_cm_variations(stap, bereken=self.data.E_kin)
        U_min, _, U_max= self.data.E_cm_variations(stap, bereken=self.data.mid_E_pot)
        boven.fill_between(time, U_min, U_max, color="#68b2de", alpha=0.55, linewidth=0,
                           label="Potentieel · CM 25–75%")
        boven.fill_between(time, K_min, K_max, color="#b288d1", alpha=0.48, linewidth=0,
                           label="Kinetisch · CM 25–75%")
        boven.plot(time, E_pot, color="#236aa6", linewidth=0.8, label="Potentiële energie")
        boven.plot(time, E_kin, color="#734aa2", linewidth=0.65, label="Kinetische energie")
        boven.set_ylabel("Energie (J)")
        boven.set_title(f"{self.data.measurement} · Cd={Cd:g}")

        lucht= Cd * lucht_per_Cd
        lucht_min= Cd_min * lucht_per_Cd
        lucht_max= Cd_max * lucht_per_Cd
        midden.fill_between(time, totaal - lucht_max, totaal - lucht_min,
                            color="#e4b72e", alpha=0.45, linewidth=0, label="Rest bij Cd 0,8–2,5")
        midden.fill_between(time, lucht_min, lucht_max, color="#df5048",
                            alpha=0.40, linewidth=0, label="Lucht bij Cd 0,8–2,5")
        midden.axhline(0, color="grey", linewidth=0.6)
        midden.plot(time, totaal - lucht, color="#b88700", linewidth=0.7, label=f"Rest bij Cd={Cd:g}")
        midden.plot(time, lucht, color="#cb302c", linewidth=1.5, label=f"Lucht bij Cd={Cd:g}")
        midden.set_ylabel("Cumulatief verlies (J)")
        y_min, y_max= midden.get_ylim()
        boven.set_ylim(0, y_max - y_min)

        onder.fill_between(time, E_min, E_max, color="#438dcc", alpha=0.55, linewidth=0,
                           label=f"Massamiddelpunten 25–75% (stap {stap:g} procentpunt)")
        onder.plot(time, gemiddeld, color="black", linewidth=1.3,
                   label=f"Gemiddelde totale energie over {venster:g} s")
        fout_mJ= 1000 * self.data.E_puntfout(venster)
        fout_ax.fill_between(time, 0, fout_mJ, color="#a96546", alpha=0.60, linewidth=0)
        fout_ax.set_ylim(bottom=0)
        fout_ax.set_ylabel("Fout (mJ)")
        fout_ax.set_title("Bijdrage van de markerposities aan de gemiddelde energie", fontsize=9, loc="left")
        fout_ax.grid(axis="y", alpha=0.25)
        onder.set(xlabel="Opnametijd (s)", ylabel="Energie (J)")
        for ax in (boven, midden, onder):
            ax.legend()
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.PLOT_DIR / f"E_cm_drag_{self.data.measurement}.png", dpi=180)
        plt.show()

    def plot_E_pos(self):
        plt.figure()
        plt.scatter(self.data.mid_time(), self.data.E_pos(), s=1)
    
        plt.xlabel("Time (s)")
        plt.ylabel("Kinetic + Potential energy (J)")
        plt.title("Kinetic + Potential energy")
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_kin+pot_meting_test.png", dpi=300, bbox_inches="tight")
        plt.show()
            
        
    
# if __name__ == "__main__":
#     metingen= []
#     for bestand in sorted(data_folder().glob("Basler*_nov25-meting1.mqa")):
#         if bestand.stem.endswith("_TSV"):  # tweede export van dezelfde opname
#             continue
#         meting= calculations(bestand)
#         metingen.append(meting)
#         graphs= plot(meting)
#         #graphs.plot_P1()
#         graphs.plot_E_cm_met_drag()

#     sigma= plot.plot_afwijkingen(metingen)
#     print(sigma)


if __name__ == "__main__":
    bestand = "Basler_a2A2464_25nov-meting2.mqa"
    meting = calculations(bestand)
    graphs = plot(meting)
    graphs.plot_E_kin()