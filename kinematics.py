from typing import Literal
import numpy as np
from numpy import cos, sin, atan2, sqrt

class TMatrix:

    def __init__(self, array):
        """
        Parameters
        ----------
        array: np.ndarray
            The 4D array representing the transformation matrix
        """
        self.matrix = array
        # Position vector part of the transformation matrix
        self.position = self.matrix[:3, 3]
        # Rotation matrix part of the transformation matrix
        self.rotation = self.matrix[:3, :3]


class Kinematics:

    def __init__(self, th, *dh_parameters):
        """
        Parameters
        ----------
        dh_parameters: 3-tuple of lists
            A three-tuple containing lists of a parameters, alpha parameters, and d parameters, in that order
        """

        # TODO: Down the line, make this a 4 tuple and pass theta in too (I don't actually think that would be a good idea here)
        self.a, self.alpha, self.d = dh_parameters
        self.th = th
        # NOTE: This is always initialised by the subclass - I wonder if there's any way to make that more explicit...
        self.max_robot_len = 1
        
    def solve_IK(self, o, R=None, elbow_up=True) -> tuple[Literal[True], list[float]] | tuple[Literal[False], str]:
        """ Solve the inverse kinematics problem """
        raise NotImplemented

    def full_FK_pos(self, thetas) -> list[list]:
        """ Provides the coordinates of every frame origin - used for plotting the robot arm """
        raise NotImplemented

    def end_effector_tmatrix(self, thetas) -> TMatrix:
        """ Alias for the final TMatrix  """
        raise NotImplemented


class ElbowKinematics(Kinematics):

    def __init__(self, th, *dh_parameters):
        super().__init__(th, *dh_parameters)
        # NOTE: For 5DOF, this is: self.max_robot_len = self.a[1]+self.a[2]+self.d[4]+20
        self.max_robot_len = self.d[0]+self.a[1]+self.a[2]+20

    def T01(self, th1: float) -> TMatrix:
        """ Calculate the T_0^1 transformation matrix """

        return TMatrix(np.array([
            [cos(th1), 0, sin(th1),  0 ],
            [sin(th1), 0, -cos(th1), 0 ],
            [0,        1, 0,         self.d[0]],
            [0,        0, 0,         1 ]
        ]))

    def T02(self, th1, th2) -> TMatrix:
        """ Calculate the T_0^2 transformation matrix """

        return TMatrix(np.array([
            [cos(th1)*cos(th2), -cos(th1)*sin(th2), sin(th1),  self.a[1]*cos(th1)*cos(th2)  ],
            [sin(th1)*cos(th2), -sin(th1)*sin(th2), -cos(th1), self.a[1]*sin(th1)*cos(th2)  ],
            [sin(th2),          cos(th2),           0,         self.a[1]*sin(th2)+self.d[0] ],
            [0,                 0,                  0,         1                            ]
        ]))

    def T03(self, th1, th2, th3) -> TMatrix:
        """ Calculate the T_0^3 transformation matrix """

        return TMatrix(np.array([
            [cos(th1)*cos(th2+th3), -cos(th1)*sin(th2+th3), sin(th1),  self.a[2]*cos(th1)*cos(th2+th3)+self.a[1]*cos(th1)*cos(th2) ],
            [sin(th1)*cos(th2+th3), -sin(th1)*sin(th2+th3), -cos(th1), self.a[2]*sin(th1)*cos(th2+th3)+self.a[1]*sin(th1)*cos(th2) ],
            [sin(th2+th3),          cos(th2+th3),           0,         self.a[2]*sin(th2+th3)+self.a[1]*sin(th2)+self.d[0]         ],
            [0,                     0,                      0,         1                                                           ]
        ]))


    def full_FK_pos(self, thetas):
        return [
            self.T01(thetas[0]).position,
            self.T02(thetas[0], thetas[1]).position,
            self.T03(thetas[0], thetas[1], thetas[2]).position
        ]

    def end_effector_tmatrix(self, thetas):
        return self.T03(*thetas)

    def solve_IK(self, o, R=None, elbow_up=True):
        # TODO: Add a parameter to toggle range checks
        """
        Updates theta parameters via the inverse kinematic equations. Also checks if the thetas are
        valid given our angle constraints (theta in [-pi/2, pi/2])

        Parameters
        ----------
        o: list[float]
            The desired x, y, z coordinates of the end-effector.
        R: 2D numpy array
            The desired orientation of the end-effector, given as a rotation matrix 
            which gives the orientation of the end-effector frame wrt the base frame.
        elbow_up: bool
            If true, use the elbow up configuration, otherwise use the elbow down configuration.
            If <unadded_parameter_to_check_theta_ranges> is also true, and the specified configuration
            is unachievable, the other configuration will be used instead, despite this parameter's value.

        Returns
        -------
        thetas: list[float]
            The theta parameters necessary to achieve the desired position and orientation.
            Given in radians.
        """

        # TODO: Update this when changin to the full solution
        oc = o #- d5*R[:,2] 
        # Singularity check: if xc and yc are both 0, then th1 is undefined, so just manually set th1 = 0 (as in a singular configuration, th1's value is irrelevant)
        if oc[0] == 0 and oc[1] == 0:
            th1 = 0
        else:
            th1 = atan2(oc[0], oc[1])
        # th2 is a function of th3, so have to declare them out of order
        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
        # NOTE: Temp debug print
        print(f"D: {D}")
        if elbow_up:
            th3 = atan2(D, sqrt(1 - D**2))
        else:
            th3 = atan2(D, -sqrt(1 - D**2))
        # NOTE: This check is kinda stupid - isn't atan2 bound between -pi and pi anyway?
        if th3 < -np.pi or th3 > np.pi:
            return (False, "Theta 3 out of bounds")
        th2 = atan2(sqrt(oc[0]**2 + oc[1]**2), oc[2] - self.d[0]) - atan2(self.a[1]+self.a[2]*cos(th3), self.a[2]*sin(th3))
        if th2 < -np.pi or th2 > np.pi:
            return (False, "Theta 2 out of bounds")

        print(f"IK: th1={th1}, th2={th2}, th3={th3}")
        return (True, [th1, th2, th3])












