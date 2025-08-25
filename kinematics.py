from abc import ABC, abstractmethod
from typing import Any, Literal, override
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
        
        self.T00 = TMatrix(np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))


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

    def A(self, i: int, theta: float) -> TMatrix:
        """
        Returns the DH A_i matrix, that is, the TMatrix T^{i-1}_i.
        Parameters
        ----------
        i: int
            The subscript of the *matrix*, not the list that stores the DH parameters. That is, this is a 1-index, not a 0-index
        theta: float
            The theta for this matrix - should be theta_i (again, 1-indexed i, as in the maths)
        """

        return TMatrix(np.array([
            [cos(theta),  -sin(theta)*cos(self.alpha[i-1]),  sin(theta)*sin(self.alpha[i-1]), self.a[i-1]*cos(theta)],
            [sin(theta),   cos(theta)*cos(self.alpha[i-1]), -cos(theta)*sin(self.alpha[i-1]), self.a[i-1]*sin(theta)],
            [         0,              sin(self.alpha[i-1]),             cos(self.alpha[i-1]),            self.d[i-1]],
            [         0,                                 0,                                0,                      1]
        ]))

    def T01(self, th: int) -> TMatrix:
        """ Calculate the T^0_1 matrix """
        return self.A(1, th)

    def T0i(self, i: int, th: float, prev_T: TMatrix) -> TMatrix:
        """ Calculate the T^0_i transformation matrix for i>1 """

        return TMatrix(np.matmul(prev_T.matrix, self.A(i, th).matrix))

class ElbowKinematics(Kinematics):

    def __init__(self, *dh_parameters):
        super().__init__(*dh_parameters)


    @property
    def max_robot_len(self) -> float:
        # NOTE: For 5DOF, this is: self.max_robot_len = self.a[1]+self.a[2]+self.d[4]+20
        return self.d[0]+self.a[1]+self.a[2]


    def full_FK(self, thetas) -> ndarray:
        T01 = self.T01(thetas[0])
        T02 = self.T0i(2, thetas[1], T01)
        T03 = self.T0i(3, thetas[2], T02)
        return np.array([self.T00, T01, T02, T03])


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
        oc = o
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

    @property
    def max_robot_len(self) -> float:
        return self.d[0]+self.a[1]+self.d[3]+self.d[5]


    def full_FK(self, thetas) -> ndarray:
        T01 = self.T01(thetas[0])
        T02 = self.T0i(2, thetas[1], T01)
        T03 = self.T0i(3, thetas[2], T02)
        T04 = self.T0i(4, thetas[3], T03)
        T05 = self.T0i(5, thetas[4], T04)
        T06 = self.T0i(6, thetas[5], T05)
        return np.array([self.T00, T01, T02, T03, T04, T05, T06])


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
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])

        # NOTE: The atan2 function is of the form atan2(y, x), not atan2(x, y) 
        # In my opinion, that is backwards, but whatever
        oc = o - self.d[5]*R[:,2] 
        print(f"o:{o}, oc:{oc}")
        # Singularity check: if xc and yc are both 0, then th1 is undefined, so just manually set th1 = 0 (as in a singular configuration, th1's value is irrelevant)
        if oc[0] == 0 and oc[1] == 0:
            th1 = 0
        else:
            th1 = atan2(oc[1], oc[0])
        # th2 is a function of th3, so have to declare them out of order
        #D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
        print(f"d4 = {self.d[3]}")
        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.d[3]**2)/(2*self.a[1]*self.d[3])
        # NOTE: Temp debug print
        print(f"D: {D}")
        th3_up = atan2(-sqrt(1 - D**2), D)
        th3_down = atan2(sqrt(1 - D**2), D)
        # NOTE: This check is kinda stupid - isn't atan2 bound between -pi and pi anyway?
        # NOTE: We need only check one th3 - the other will necesarrily be valid if the first is (I think)
        if th3_up < -np.pi or th3_up > np.pi or np.isnan(th3_up):
            return (False, f"Theta 3 out of bounds: th3_up={th3_up}, th3_down={th3_down}")
        # th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_up), self.a[1]+self.a[2]*cos(th3_up))
        # th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_down), self.a[1]+self.a[2]*cos(th3_up))
        th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.d[3]*sin(th3_up), self.a[1]+self.d[3]*cos(th3_up))
        th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.d[3]*sin(th3_down), self.a[1]+self.d[3]*cos(th3_up))
        if th2_up < -np.pi or th2_up > np.pi or np.isnan(th2_up):
            return (False, "Theta 2 out of bounds")
        # Once the first three angles have been calculated, make adjustments to th3 since the x axes are actually 90 degrees apart in the "default configuration" (as draw in the DH frame, which differs 
        # from how they are expressed the in the geometric half of the 6DOF IK derivation)
        th3_up = np.pi/2 - th3_down
        th3_down += np.pi/2
        # Inverse orientation
        # NOTE: While currently unused, this alt might be useful when we limit ourselves to 180 deg rotation
        th5_alt = atan2(-sqrt(1 - (sin(th1)*R[0,2] - cos(th1)*R[1,2])**2), sin(th1)*R[0,2]-cos(th1)*R[1,2])
        th5 = atan2(sqrt(1 - (sin(th1)*R[0,2] - cos(th1)*R[1,2])**2), sin(th1)*R[0,2]-cos(th1)*R[1,2])
        th4_up = atan2(-cos(th1)*sin(th2_up+th3_up)*R[0,2]-sin(th1)*sin(th2_up+th3_up)*R[1,2]+cos(th2_up+th3_up)*R[2,2], cos(th1)*cos(th2_up+th3_up)*R[0,2]+sin(th1)*cos(th2_up+th3_up)*R[1,2]+sin(th2_up+th3_up)*R[2,2])
        th4_down = atan2(-cos(th1)*sin(th2_down+th3_down)*R[0,2]-sin(th1)*sin(th2_down+th3_down)*R[1,2]+cos(th2_down+th3_down)*R[2,2], cos(th1)*cos(th2_down+th3_down)*R[0,2]+sin(th1)*cos(th2_down+th3_down)*R[1,2]+sin(th2_down+th3_down)*R[2,2])
        th6 = atan2(sin(th1)*R[0,1]-cos(th1)*R[1,1], -sin(th1)*R[0,0]+cos(th1)*R[1,0])
        if sin(th5) == 0:
            print("WARNING: Singular config s_5 = 0 - currently unhandled")

        return (True, [[th1, th2_up, th3_up, th4_up, th5, th6], [th1, th2_down, th3_down, th4_down, th5, th6]])
