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
    try:
        dir= Path("datafiles")
        return pd.read_csv(dir / measurement)
    except:
        print("could not find file")
        
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


class calculations():
    
    def __init__(self, measurement) -> None:
        """Initialize measurement data and physical properties of a double pendulum.
        
        Loads the measurement data, extracts the positions and time values, and
        defines the lengths and masses of both pendulum arms.
        
        Args:
            measurement: Number identifying the measurement to load.
        """
        self.df= df_load(measurement)
        self.points_lab= df_points_lab(self.df)
        self.points_hng= df_points_hng(self.df)
        self.points_pot= df_points_pot(self.df)
        self.time= load_time(self.df)
        
        self.L1, self.L2= 225/1000, 220/1000
        self.m1, self.m2= 338/1000, 418/1000
        self.g= 9.81 # m/s**2
        self.r_cmA, self.r_cmB= 1/2 * self.L1, 1/2 * self.L2 
        
        
    def angle_A(self) -> np.ndarray:
        """ Calculate angle between arm A and the origin using the np.arctan2 function
        
        Returns:
            An array of angles in radians from -π to π 
        """
        return np.unwrap(np.arctan2(-self.points_hng[1][0], self.points_hng[1][1]))

    def angle_B(self) -> np.ndarray:
        """ Calculate angle between arm B and the origin using the np.arctan2 function
        
        Returns:
            An array of angles in radians from -π to π 
        """
        return np.unwrap(np.arctan2(self.points_hng[3][0], -self.points_hng[3][1]))
        
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
        ## E_kin = 1/2 * m * v**2
        return 1/2 * (self.m1 * self.linear_v_A()**2 + self.m2 * self.linear_v_B()**2)
    
    def E_pot(self):
        zero, ceil= np.nanmin(self.points_pot[3]), np.nanmax(self.points_pot[1])
        Ltot= self.L1 + self.L2
        scale= Ltot / (ceil - zero)
        return self.g * ((self.m1) * self.points_pot[2] * scale + (self.m2) * self.points_pot[3] * scale)
    
    def mid_E_pot(self):
        return (self.E_pot()[:-1] + self.E_pot()[1:]) / 2

    def E_pos(self):
        return self.E_kin() + self.mid_E_pot()
    
    def E_drag(self):
        
    
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
        
    def plot_E_pos(self):
        plt.figure()
        plt.scatter(self.data.mid_time(), self.data.E_pos(), s=1)
    
        plt.xlabel("Time (s)")
        plt.ylabel("Kinetic + Potential energy (J)")
        plt.title("Kinetic + Potential energy")
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_kin+pot_meting_test.png", dpi=300, bbox_inches="tight")
        plt.show()
            
        
    
            
for i in [1,2,3,4,5]:
    test= calculations(f"Basler_a2A2464_11nov-meting{i}.mqa")
    graphs= plot(test)
    graphs.plot_E_pos()


for i in [1,2]:
    test= calculations(f"Basler_a2A2464_18nov-meting{i}.mqa")
    graphs= plot(test)
    graphs.plot_E_pos()
