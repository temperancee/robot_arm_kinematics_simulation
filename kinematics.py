from abc import ABC, abstractmethod
from typing import Any, Literal
import numpy as np
from numpy import cos, ndarray, sin, atan2, sqrt

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


class Kinematics(ABC):
    def __init__(self, *dh_parameters):
        """
        Parameters
        ----------
        dh_parameters: 3-tuple of lists
            A three-tuple containing lists of a parameters, alpha parameters, and d parameters, in that order
        """

        self.a, self.alpha, self.d = dh_parameters


    @property
    @abstractmethod
    def max_robot_len(self) -> float:
        """ Used to define maximum length of each axis on the plot """

        
    @abstractmethod
    def solve_IK(self, o, R=None) -> Any:
        """ Solve the inverse kinematics problem """
        pass

    @abstractmethod
    def full_FK(self, thetas) -> ndarray:
        """ Provides the transformation matrices for every frame - used for plotting the robot arm and frame coordinate axes """
        pass

    @abstractmethod
    def end_effector_tmatrix(self, thetas) -> TMatrix:
        """ Alias for the final TMatrix  """
        pass


class ElbowKinematics(Kinematics):

    def __init__(self, *dh_parameters):
        super().__init__(*dh_parameters)

    def T00(self) -> TMatrix:
        """ Provide the T_0^0 transformation matrix (identity matrix) """
        return TMatrix(np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))

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

    @property
    def max_robot_len(self) -> float:
        # NOTE: For 5DOF, this is: self.max_robot_len = self.a[1]+self.a[2]+self.d[4]+20
        return self.d[0]+self.a[1]+self.a[2]+20


    def full_FK(self, thetas):
        return np.array([
            self.T00(),
            self.T01(thetas[0]),
            self.T02(thetas[0], thetas[1]),
            self.T03(thetas[0], thetas[1], thetas[2])
        ])

    def end_effector_tmatrix(self, thetas):
        return self.T03(*thetas)

    def solve_IK(self, o, R=None):
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

        #NOTE: The atan2 function is of the form atan2(y, x), not atan2(x, y) 
        # In my opinion, that is backwards, but whatever
        oc = o #- d5*R[:,2] 
        # Singularity check: if xc and yc are both 0, then th1 is undefined, so just manually set th1 = 0 (as in a singular configuration, th1's value is irrelevant)
        if oc[0] == 0 and oc[1] == 0:
            th1 = 0
        else:
            th1 = atan2(oc[1], oc[0])
        # th2 is a function of th3, so have to declare them out of order
        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
        # NOTE: Temp debug print
        print(f"D: {D}")
        th3_up = atan2(-sqrt(1 - D**2), D)
        th3_down = atan2(sqrt(1 - D**2), D)
        # NOTE: This check is kinda stupid - isn't atan2 bound between -pi and pi anyway?
        # NOTE: We need only check one th3 - the other will necesarrily be valid if the first is (I think)
        if th3_up < -np.pi or th3_up > np.pi or np.isnan(th3_up):
            return (False, "Theta 3 out of bounds")
        th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_up), self.a[1]+self.a[2]*cos(th3_up))
        th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_down), self.a[1]+self.a[2]*cos(th3_up))
        if th2_up < -np.pi or th2_up > np.pi or np.isnan(th2_up):
            return (False, "Theta 2 out of bounds")

        return (True, [[th1, th2_up, th3_up], [th1, th2_down, th3_down]])



class SixDOFKinematics(ElbowKinematics):
    def __init__(self, *dh_parameters):
        super().__init__(*dh_parameters)


    def T04(self, th1, th2, th3, th4) -> TMatrix:
        """ Calculate the T_0^4 transformation matrix """

        return TMatrix(np.array([
            [cos(th1)*cos(th2+th3+th4),    -sin(th1), -cos(th1)*sin(th2+th3+th4), self.a[2]*cos(th1)*cos(th2+th3)+self.a[1]*cos(th1)*cos(th2) ],
            [sin(th1)*cos(th2+th3+th4),     cos(th1), -sin(th1)*sin(th2+th3+th4), self.a[2]*sin(th1)*cos(th2+th3)+self.a[1]*sin(th1)*cos(th2) ],
            [         sin(th2+th3+th4),            0,           cos(th2+th3+th4),         self.a[2]*sin(th2+th3)+self.a[1]*sin(th2)+self.d[0] ],
            [                        0,            0,                          0,                                                           1 ]
        ]))

    def T05(self, th1, th2, th3, th4, th5) -> TMatrix:
        """ Calculate the T_0^5 transformation matrix """

        return TMatrix(np.array([
            [cos(th1)*cos(th2+th3+th4)*cos(th5) - sin(th1)*sin(th5), -cos(th1)*sin(th2+th3+th4),    cos(th1)*cos(th2+th3+th4)*sin(th5) + sin(th1)*cos(th5), self.a[2]*cos(th1)*cos(th2+th3)+self.a[1]*cos(th1)*cos(th2) ],
            [sin(th1)*cos(th2+th3+th4)*cos(th5) + cos(th1)*sin(th5), -sin(th1)*sin(th2+th3+th4),    sin(th1)*cos(th2+th3+th4)*sin(th5) - cos(th1)*cos(th5), self.a[2]*sin(th1)*cos(th2+th3)+self.a[1]*sin(th1)*cos(th2) ],
            [                             sin(th2+th3+th4)*cos(th5),           cos(th2+th3+th4),                                                         0,         self.a[2]*sin(th2+th3)+self.a[1]*sin(th2)+self.d[0] ],
            [                                                     0,                          0,                                                         0,                                                           1 ]
        ]))

    def T06(self, th1, th2, th3, th4, th5, th6) -> TMatrix:
        """ Calculate the T_0^6 transformation matrix """

        return TMatrix(np.array([
            [cos(th6)*(cos(th1)*cos(th2+th3+th4)*cos(th5) - sin(th1)*sin(th5)) - cos(th1)*sin(th2+th3+th4)*sin(th6), -sin(th6)*(cos(th1)*cos(th2+th3+th4)*cos(th5) - sin(th1)*sin(th5)) - cos(th1)*sin(th2+th3+th4)*cos(th6),    cos(th1)*cos(th2+th3+th4)*sin(th5) + sin(th1)*cos(th5), self.a[2]*cos(th1)*cos(th2+th3)+self.a[1]*cos(th1)*cos(th2) + self.d[5]*(cos(th1)*cos(th2+th3+th4)*sin(th5) + sin(th1)*cos(th5)) ],
            [cos(th6)*(sin(th1)*cos(th2+th3+th4)*cos(th5) + cos(th1)*sin(th5)) - sin(th1)*sin(th2+th3+th4)*sin(th6), -sin(th6)*(sin(th1)*cos(th2+th3+th4)*cos(th5) + cos(th1)*sin(th5)) - sin(th1)*sin(th2+th3+th4)*cos(th6),    sin(th1)*cos(th2+th3+th4)*sin(th5) - cos(th1)*cos(th5), self.a[2]*sin(th1)*cos(th2+th3)+self.a[1]*sin(th1)*cos(th2) + self.d[5]*(sin(th1)*cos(th2+th3+th4)*sin(th5) - cos(th1)*cos(th5)) ],
            [                                        sin(th2+th3+th4)*cos(th5)*cos(th6) + cos(th2+th3+th4)*sin(th6),                                         -sin(th2+th3+th4)*cos(th5)*sin(th6) + cos(th2+th3+th4)*cos(th6),                                 sin(th2+th3+th4)*sin(th5),                                        self.a[2]*sin(th2+th3)+self.a[1]*sin(th2)+self.d[0] + self.d[5]*sin(th2+th3+th4)*sin(th5) ],
            [                                                                                                     0,                                                                                                       0,                                                         0,                                                                                                                                1 ]
        ]))

    @property
    def max_robot_len(self) -> float:
        # NOTE: For 5DOF, this is: self.max_robot_len = self.a[1]+self.a[2]+self.d[4]+20
        return self.d[0]+self.a[1]+self.a[2]+self.d[5]+20


    def full_FK(self, thetas):
        return np.array([
            self.T00(),
            self.T01(thetas[0]),
            self.T02(thetas[0], thetas[1]),
            self.T03(thetas[0], thetas[1], thetas[2]),
            self.T04(thetas[0], thetas[1], thetas[2], thetas[3]),
            self.T05(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]),
            self.T06(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5])
        ])

    def end_effector_tmatrix(self, thetas):
        return self.T06(*thetas)

    def solve_IK(self, o, R=None):
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

        if R is None:
            R = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        #NOTE: The atan2 function is of the form atan2(y, x), not atan2(x, y) 
        # In my opinion, that is backwards, but whatever
        oc = o - self.d[5]*R[:,2] 
        # Singularity check: if xc and yc are both 0, then th1 is undefined, so just manually set th1 = 0 (as in a singular configuration, th1's value is irrelevant)
        if oc[0] == 0 and oc[1] == 0:
            th1 = 0
        else:
            th1 = atan2(oc[1], oc[0])
        # th2 is a function of th3, so have to declare them out of order
        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
        # NOTE: Temp debug print
        print(f"D: {D}")
        th3_up = atan2(-sqrt(1 - D**2), D)
        th3_down = atan2(sqrt(1 - D**2), D)
        # NOTE: This check is kinda stupid - isn't atan2 bound between -pi and pi anyway?
        # NOTE: We need only check one th3 - the other will necesarrily be valid if the first is (I think)
        if th3_up < -np.pi or th3_up > np.pi or np.isnan(th3_up):
            return (False, "Theta 3 out of bounds")
        th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_up), self.a[1]+self.a[2]*cos(th3_up))
        th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_down), self.a[1]+self.a[2]*cos(th3_up))
        if th2_up < -np.pi or th2_up > np.pi or np.isnan(th2_up):
            return (False, "Theta 2 out of bounds")

        return (True, [[th1, th2_up, th3_up], [th1, th2_down, th3_down]])
