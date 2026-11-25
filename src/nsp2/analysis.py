# import matplotlib
# matplotlib.use("WebAgg")

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



def df_load(measurement):
    data_dir= Path(__file__).resolve().parents[2] / "datafiles"
    # Ook bruikbaar als deze werkkopie buiten de NSP2-repository staat.
    if not data_dir.is_dir():
        data_dir= Path.home() / "Documents/NSP2/NSP2/datafiles"
    return pd.read_csv(data_dir / measurement)


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
    
    def E_cm_variations(self, stap=1):
        """Geef per tijdstip de laagste, middelste stand (50%, 50%) en hoogste energie."""
        # stap in procentpunten. Neem ook 75% mee als de laatste stap korter is.
        if not np.isfinite(stap) or not 0 < stap <= 50:
            raise ValueError("Kies een stap groter dan 0 en maximaal 50 procentpunten.")
        fractions= np.append(np.arange(25, 75, stap), 75) / 100
        old_A, old_B= self.r_cmA, self.r_cmB
        try:
            self.set_cm(0.5, 0.5)
            E_middle= self.E_pos()
            E_min= np.full_like(E_middle, np.inf)
            E_max= np.full_like(E_middle, -np.inf)

            # In E_pos hangen de bijdragen van r_cmA en r_cmB afzonderlijk
            # van hun eigen afstand af. Bereken de verandering door B één keer.
            changes_B= []
            for b in fractions:
                self.set_cm(0.5, b)
                changes_B.append(self.E_pos() - E_middle)

            for a in fractions:
                self.set_cm(a, 0.5)
                E_A= self.E_pos()
                for change_B in changes_B:
                    energy= E_A + change_B
                    E_min= np.minimum(E_min, energy)
                    E_max= np.maximum(E_max, energy)
        finally:
            self.r_cmA, self.r_cmB= old_A, old_B
        return E_min, E_middle, E_max

    def E_drag(self, Cd, b_A, b_B, rho=1.2):
        """Energieverlies aan luchtweerstand per tijdstap (J).
        b_A en b_B zijn de effectieve dieptes loodrecht op het bewegingsvlak (m).
        Cd is de aangenomen weerstandscoëfficiënt voor beide armen.
        """
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
        plt.scatter(self.data.mid_time(), self.data.E_kin(), s=1)

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

    def plot_E_cm_met_drag(self, Cd, b_A, b_B, stap=1, rho=1.2):
        """Boven de energievariaties, onder het verlies voor de 50%/50%-stand."""
        fig, (boven, onder)= plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        E= self._draw_cm(boven, stap)
        time= self.data.mid_time()
        drag= self.data.E_drag(Cd, b_A, b_B, rho)
        totaal= np.full_like(E, np.nan)
        lucht= np.full_like(E, np.nan)

        # Start na een trackinggat opnieuw: het verlies binnen het gat is onbekend.
        geldig= np.flatnonzero(np.isfinite(E) & np.isfinite(drag))
        for stuk in np.split(geldig, np.flatnonzero(np.diff(geldig) > 1) + 1):
            if stuk.size == 0:
                continue
            totaal[stuk]= E[stuk[0]] - E[stuk]
            # E ligt midden in elke tijdstap: tel halve stappen aan de randen.
            lucht[stuk]= np.r_[0, np.cumsum((drag[stuk[:-1]] + drag[stuk[1:]]) / 2)]

        onder.axhline(0, color="grey", linewidth=0.6)
        onder.plot(time, totaal, color="black", linewidth=0.55, alpha=0.65, label="Totaalverlies (50%/50%)")
        onder.plot(time, totaal - lucht, color="#b88700", linewidth=0.8, label="Resterend verlies")
        onder.plot(time, lucht, color="#cb302c", linewidth=1.4, label="Luchtverlies (model)")
        onder.set(xlabel="Opnametijd (s)", ylabel="Verlies sinds segmentbegin (J)")
        onder.legend()
        onder.grid(alpha=0.25)
        boven.set_title(f"{self.data.measurement} · Cd={Cd:g}, bA={b_A:g} m, bB={b_B:g} m")
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
            
        
if __name__ == "__main__":
    for i in [1,2,3,4,5]:
        test= calculations(f"Basler_a2A2464_11nov-meting{i}.mqa")
        graphs= plot(test)
        graphs.plot_E_cm_met_drag(Cd=1.0, b_A=0.005, b_B=0.014, stap=1)

    for i in [1,2]:
        test= calculations(f"Basler_a2A2464_18nov-meting{i}.mqa")
        graphs= plot(test)
        graphs.plot_E_cm_met_drag(Cd=1.0, b_A=0.005, b_B=0.014, stap=1)
    
    for i in [1,2,3]:
        test= calculations(f"Basler_a2A2464_23nov-meting{i}.mqa")
        graphs= plot(test)
        graphs.plot_E_cm_met_drag(Cd=1.0, b_A=0.005, b_B=0.014, stap=1)
