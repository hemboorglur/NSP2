# Generated from: draft.ipynb
# Converted at: 2026-11-17T14:30:54.067Z

## NOAH ZEEMAN - 15494365 - University of Amsterdam
## NSP2 - script for the data analysis of the 'Motion Analysis'-project
## 17 nov. '26 - 17 nov. '26 (latest change)


import pandas as pd
import numpy as np
import math as m
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


""" Script for data-analysis for the NSP2 course 

Summary and description is subject to change
"""

def df_load(meting: int) -> pd.DataFrame:
    """ Load dataframe from specific measurement 
    
    Args:
        meting: Number identifying the measurement to load.
    Returns:
        Dataframe of data in .csv-file"""

    return pd.read_csv(f"datafiles/Basler_a2A2464-meting{meting}_11nov.mqa")

def load_points(dataframe: pd.DataFrame) -> dict[tuple[str: list[float], str: list[float]]]:
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
    for i in [1,2,3]: points[i]= (df[f"Point #{i}.X"]-df["Point #1.X"], df[f"Point #{i}.Y"]-df["Point #1.Y"])
    return points
    
def load_points_relative(dataframe: pd.DataFrame) -> dict[tuple[str: list[float], str: list[float]]]:
    """ Load all coordinates from DataFrame, converts to translated frame and appends to dictonary

    Load coordinates from DataFrame and converts to translated frame by subtracting the coordinate-
    value of P2, setting this point as the origin. Finally, appends the new coordinates to 
    a dictonary in the following format: {x:[..], y:[..]}

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        points: Dictonary with coordinates in the labframe
    """

    df, points_relative= dataframe, {}
    for i in [1,2,3]: points_relative[i]= (df[f"Point #{i}.X"]-df["Point #2.X"], df[f"Point #{i}.Y"]-df["Point #2.Y"])
    return points_relative

def load_sizes(dataframe: pd.DataFrame) -> dict[str: list[float], str: list[float]]:
    """ Load in all sizes of tracked points (later usefull for errorbars)

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        sizes: Dictonary with sizes of the tracked points
    """

    df, sizes= dataframe, {}
    for i in [1,2,3]:
        sizes[i]= (df[f"Point #{i}.SizeX"], df[f"Point #{i}.SizeY"])
    return sizes


def load_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Load time of frames from DataFrame

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        DataFrame of all times
    """

    df= dataframe
    return df["Time"]

def load_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Load index number of frames from DataFrame

    Args:
        dataframe: DataFrame from .csv data-file
    Returns:
        DataFrame of all frame-numbers
    """

    df= dataframe
    return df["Frame"]


## load ~meting1~.mqa as df 
#df= df_load(1)

## load points as a library of tuples: 
# visually, points= {(x1:[..],y1:[..]),(x2:[..],y2:[..]),(x3:[..],y3:[..])}
# call on them using: points[1, 2 or 3][0 for x, 1 for y][element position]
#points= load_points()
#points_relative= load_points_relative()
#sizes= load_sizes()

## create a class to for all calculations
class frame:
    
    def __init__(self, meting) -> None:
        """Initialize measurement data and physical properties of a double pendulum.

        Loads the measurement data, extracts the positions and time values, and
        defines the lengths and masses of both pendulum arms.

        Args:
            meting: Number identifying the measurement to load.
        """
        self.meting= meting 
        self.df= df_load(meting)
        self.points= load_points(self.df)
        self.points_relative= load_points_relative(self.df)

        self.time= load_time(self.df)
        self.frame= load_frame(self.df)

        ## for easier coding 
        self.x1, self.y1= self.points_relative[1]
        self.x2, self.y2= self.points_relative[2]
        self.x3, self.y3= self.points_relative[3]

        ## pixel to mm scale
        ##self.scale= 

        ## lenght in mm and mass in gram
        #errors will be added later
        self.L1, self.L2= 225, 220
        #self.L1_err, self.L2_err= 
        self.m1, self.m2= 338, 418
        #self.m1_err, self.m2_err= 
        self.g = 9.81
        
        
        ## Because I am using the np.diff() function, I know the 
        ## difference in angles between 2 frames, I use that as dpsi.
        ## However, I cannout keep using the same angles and points, because 
        ## then I would miss 1 element, meaning: the angles and difference in 
        ## angles are at different times.
        ## The simpelest solution is to use the angle mid-points. This causes
        ## one element to be lost, and can be used in combination with the 
        ## difference in angle 
        
    def angle_A(self) -> np.ndarray:
        """ Calculate angle between arm A and the origin using the np.arctan2 and np.unwrap functions
        
        Returns:
            An array of angles in radians from -π to π 
        """
        return np.unwrap(np.arctan2(-self.x1, self.y1))

    def angle_B(self) -> np.ndarray:
        """ Calculate angle between arm B and the origin using the np.arctan2 and np.unwrap functions
        
        Returns:
            An array of angles in radians from -π to π 
        """
        return np.unwrap(np.arctan2(self.x3, -self.y3))
        
    def angle_mid(self, angle: str) -> np.ndarray:
        theta = getattr(self, f"angle_{angle}")()a
        return (theta[:-1] + theta[1:]) / 2

    def time_mid(self) -> np.ndarray:
        return (self.time[:-1] + self.time[1:]) / 2

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
        omega_A= self.angular_v("A")
        angle_A= self.angle_A()[1:]
        return (self.L1/1000 * omega_A * np.cos(angle_A), self.L1/1000 * omega_A * np.sin(angle_A))

    def linear_v_B(self):
        omega_A, omega_B= self.angular_v("A"), self.angular_v("B") 
        angle_A, angle_B= self.angle_A()[1:], self.angle_B()[1:]
        return (self.L1/1000 * omega_A * np.cos(angle_A) + self.L2/1000 * omega_B * np.cos(angle_B), 
                self.L1/1000 * omega_A * np.sin(angle_A) + self.L2/1000 * omega_B * np.sin(angle_B))
    
    def velocity_C1(self):
        v_A_x, v_A_y = self.linear_v_A()
        return v_A_x / 2, v_A_y / 2

    def velocity_C2(self):
        v_A_x, v_A_y = self.linear_v_A()
        v_B_x, v_B_y = self.linear_v_B()
        return (v_A_x + v_B_x) / 2, (v_A_y + v_B_y) / 2

    def E_kin(self) -> np.ndarray:
        vx1, vy1 = self.velocity_C1()
        vx2, vy2 = self.velocity_C2()
        omega_A = self.angular_v("A")
        omega_B = self.angular_v("B")

        # Translational kinetic energy
        E_trans_A = 0.5 * self.m1/1000 * (vx1**2 + vy1**2)
        E_trans_B = 0.5 * self.m2/1000 * (vx2**2 + vy2**2)

        # Moment of inertia of homogeneous rods
        I_A = (1 / 12) * self.m1/1000 * (self.L1/1000)**2
        I_B = (1 / 12) * self.m2/1000 * (self.L2/1000)**2

        # Rotational kinetic energy
        E_rot_A = 0.5 * I_A * omega_A**2
        E_rot_B = 0.5 * I_B * omega_B**2

        return E_trans_A + E_rot_A + E_trans_B + E_rot_B

    def E_pot(self) -> np.ndarray:

        theta_A = self.angle_mid("A")
        theta_B = self.angle_mid("B")

        # Heights measured relative to fixed point P1
        y_C1 = -((self.L1/1000) / 2) * np.cos(theta_A)
        y_C2 = (-(self.L1/1000) * np.cos(theta_A) - ((self.L2/1000) / 2) * np.cos(theta_B) )

        return self.g * (self.m1/1000 * y_C1 + self.m2/1000 * y_C2)

    def E_tot(self) -> np.ndarray:
        return self.E_kin() + self.E_pot()


class plot:

    def __init__(self, measurement: frame) -> None:
        self.data= measurement
        self.meas= self.data.meting

        self.PLOT_DIR = Path("plots")
        self.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def plot_angular_v(self):
        t = self.data.time_mid()

        plt.figure()
        plt.plot(t[1:], self.data.angular_v("A"), label="Arm A")
        plt.plot(t[1:], self.data.angular_v("B"), label="Arm B")

        plt.xlabel("Time (s)")
        plt.ylabel("Angular velocity (rad/s)")
        plt.title("Angular velocity")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"angular_v_meting{self.meas}-copy.png", dpi=300, bbox_inches="tight")
        
    def plot_linear_v(self):
        t = self.data.time_mid()

        vx1, vy1 = self.data.velocity_C1()
        vx2, vy2 = self.data.velocity_C2()

        v1 = np.hypot(vx1, vy1)
        v2 = np.hypot(vx2, vy2)

        plt.figure()
        plt.plot(t[1:], v1, label="Center of mass A")
        plt.plot(t[1:], v2, label="Center of mass B")

        plt.xlabel("Time (s)")
        plt.ylabel("Linear speed (m/s)")
        plt.title("Center-of-mass speeds")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"linear_v_meting{self.meas}-copy.png", dpi=300, bbox_inches="tight")
       
    def plot_E_kin(self):
        plt.figure()
        plt.plot(self.data.time_mid()[1:], self.data.E_kin())

        plt.xlabel("Time (s)")
        plt.ylabel("Kinetic energy (J)")
        plt.title("Kinetic energy")
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_kin_meting{self.meas}-copy.png", dpi=300, bbox_inches="tight")
       
    def plot_E_pot(self):
        plt.figure()
        plt.plot(self.data.time_mid(), self.data.E_pot())

        plt.xlabel("Time (s)")
        plt.ylabel("Potential energy (J)")
        plt.title("Potential energy")
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_pot_meting{self.meas}-copy.png", dpi=300, bbox_inches="tight")
       
    def plot_E_tot(self):
        plt.figure()
        plt.plot(self.data.time_mid()[1:], self.data.E_kin(), label="Kinetic")
        plt.plot(self.data.time_mid(), self.data.E_pot(), label="Potential")
        plt.plot(self.data.time_mid(), self.data.E_tot(), label="Total", linewidth=2)

        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.title("Mechanical energy")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.PLOT_DIR / f"E_tot_meting{self.meas}-copy.png", dpi=300, bbox_inches="tight")
       

for i in range(1,6):
    plt.close("all")

    test = frame(i)
    graphs = plot(test)
    graphs.plot_angular_v()
    graphs.plot_linear_v()
    graphs.plot_E_kin()
    graphs.plot_E_pot()
    graphs.plot_E_tot()