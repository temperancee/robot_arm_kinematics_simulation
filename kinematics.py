from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional
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

@dataclass
class IKResult:
    """ Return type of the inverse kinematics functions """
    success: bool
    solutions: dict[str, list[float]] = field(default_factory=dict)
    error: Optional[str] = None 


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
    def solve_IK(self, o, rpy=None) -> IKResult:
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


    def solve_IK(self, o, rpy=None) -> IKResult:
        # TODO: Add a parameter to toggle range checks
        """
        Updates theta parameters via the inverse kinematic equations. Also checks if the thetas are
        valid given our angle constraints (theta in [-pi/2, pi/2])

        Parameters
        ----------
        o: list[float]
            The desired x, y, z coordinates of the end-effector.
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

        # NOTE: The atan2 function is of the form atan2(y, x), not atan2(x, y) 
        oc = o
        # Singularity check: if xc and yc are both 0, then th1 is undefined, so just manually set th1 = 0 (as in a singular configuration, th1's value is irrelevant)
        # Also flag this as a singularity so it can be manually changed outside of this function - if we had th1=45 for example at x=1 y=1, then moved back to x=0 y=0, we would want th1 to remain 45 for
        # continuity of movement
        if round(oc[0]) == 0 and round(oc[1]) == 0:
            th1 = 0
        else:
            th1 = atan2(oc[1], oc[0])

        # th2 is a function of th3, so have to declare them out of order
        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
        th3_up = atan2(-sqrt(1 - D**2), D)
        th3_down = atan2(sqrt(1 - D**2), D)
        # NOTE: This check is kinda stupid - isn't atan2 bound between -pi and pi anyway? We leave this here since it will be useful when bounding the angles
        # NOTE: We need only check one th3 - the other will necesarrily be valid if the first is (I think)
        if th3_up < -np.pi or th3_up > np.pi or np.isnan(th3_up):
            return IKResult(False, error=f"Theta 3 out of bounds: th3_up={th3_up}, th3_down={th3_down}")
        th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_up), self.a[1]+self.a[2]*cos(th3_up))
        th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_down), self.a[1]+self.a[2]*cos(th3_up))
        if th2_up < -np.pi or th2_up > np.pi or np.isnan(th2_up):
            return IKResult(False, error="Theta 2 out of bounds")

        return IKResult(True, solutions={"up": [th1, th2_up, th3_up], "down": [th1, th2_down, th3_down]})



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


    def solve_IK(self, o, rpy=None) -> IKResult:
        # TODO: Add a parameter to toggle range checks
        """
        Updates theta parameters via the inverse kinematic equations. Also checks if the thetas are
        valid given our angle constraints (theta in [-pi/2, pi/2] (to be added later))

        Parameters
        ----------
        o: list[float]
            The desired x, y, z coordinates of the end-effector.
        R: 2D numpy array
            The desired orientation of the end-effector. Given as a rotation matrix 
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

        if rpy is None:
            R = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
        else:
            (r, p, y) = np.deg2rad(rpy)
            print(f"roll={r}, pitch={p}, yaw={y}")
            R = np.array([
                [cos(y)*cos(p), -sin(y)*cos(r)+cos(y)*sin(p)*sin(r),  sin(y)*sin(r)+cos(y)*sin(p)*cos(r)],
                [sin(y)*cos(p),  cos(y)*cos(r)+sin(y)*sin(p)*sin(r), -cos(y)*sin(r)+sin(y)*sin(p)*cos(r)],
                [      -sin(p),                       cos(p)*sin(r),                       cos(p)*cos(r)]
            ])
            print(f"R = {R}")

        # NOTE: The atan2 function is of the form atan2(y, x), not atan2(x, y) 
        oc = o - self.d[5]*R[:,2] 
        print(f"o:{o}, oc:{oc}")
        # We don't need to check if ox and oy = 0, np.atan2(0,0) is defined to be 0
        # Also flag this as a singularity so it can be manually changed outside of this function - if we had th1=45 for example at x=1 y=1, then moved back to x=0 y=0, we would want th1 to remain 45 for
        # continuity of movement
        th1 = atan2(oc[1], oc[0])
        # HACK: If xc and yc are both really close to 0, but y is something like -2.28x10^-18, th1 will be 90, even though it should be atan2(0,0)=0. We manually intervene and set theta to 0
        if round(oc[0]) == 0 and round(oc[1]) == 0:
            th1 = 0

        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.d[3]**2)/(2*self.a[1]*self.d[3])
        # NOTE: |D|>1 implies th3 cannot be computed, and hence th2 cannot be computed. This only occurs when a o_c is unreachable by the elbow manipulator.
        if abs(D) > 1:
            return IKResult(False, error=f"Configuration impossible: D={D}")
        th3_up = atan2(-sqrt(1 - D**2), D)
        th3_down = atan2(sqrt(1 - D**2), D)
        th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.d[3]*sin(th3_up), self.a[1]+self.d[3]*cos(th3_up))
        th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.d[3]*sin(th3_down), self.a[1]+self.d[3]*cos(th3_up))
        # Once the first three angles have been calculated, make adjustments to th3 since the x axes are actually 90 degrees apart in the "default configuration" as draw in the DH frame, which differs 
        # from how they are expressed the in the geometric solution of the inverse position problem
        th3_up = np.pi/2 - th3_down
        th3_down += np.pi/2

        def inverse_orientation(th1, th2, th3):

            r11 = cos(th1)*cos(th2+th3)*R[0,0]+sin(th1)*cos(th2+th3)*R[1,0]+sin(th2+th3)*R[2,0]
            r21 = sin(th1)*R[0,0]-cos(th1)*R[1,0]
            r13 = cos(th1)*cos(th2+th3)*R[0,2]+sin(th1)*cos(th2+th3)*R[1,2]+sin(th2+th3)*R[2,2]
            r23 = sin(th1)*R[0,2]-cos(th1)*R[1,2]
            r31 = cos(th1)*sin(th2+th3)*R[0,0]+sin(th1)*sin(th2+th3)*R[1,0]-cos(th2+th3)*R[2,0]
            r32 = cos(th1)*sin(th2+th3)*R[0,1]+sin(th1)*sin(th2+th3)*R[1,1]-cos(th2+th3)*R[2,1]
            r33 = cos(th1)*sin(th2+th3)*R[0,2]+sin(th1)*sin(th2+th3)*R[1,2]-cos(th2+th3)*R[2,2]

            th5 = atan2(sqrt(1 - r33**2), r33)

            if sin(th5) == 0 and cos(th5) == 1:
                th5 = 0
                th4 = 0
                th6 = atan2(r21, r11)
            elif sin(th5) == 0 and cos(th5) == -1:
                th5 = 0
                th4 = 0
                th6 = atan2(-r21, -r11)
            else:
                th4 = atan2(r23, r13)
                th6 = atan2(r32, -r31)
            return (th4, th5, th6)
        
        th4_up, th5_up, th6_up = inverse_orientation(th1, th2_up, th3_up)
        th4_down, th5_down, th6_down = inverse_orientation(th1, th2_down, th3_down)

        return IKResult(True, solutions={"up": [th1, th2_up, th3_up, th4_up, th5_up, th6_up], "down": [th1, th2_down, th3_down, th4_down, th5_down, th6_down]})



class FiveDOFKinematics(ElbowKinematics):
    def __init__(self, *dh_parameters):
        super().__init__(*dh_parameters)

    @property
    def max_robot_len(self) -> float:
        return self.d[0]+self.a[1]+self.a[2]+self.d[4]


    def full_FK(self, thetas) -> ndarray:
        T01 = self.T01(thetas[0])
        T02 = self.T0i(2, thetas[1], T01)
        T03 = self.T0i(3, thetas[2], T02)
        T04 = self.T0i(4, thetas[3], T03)
        T05 = self.T0i(5, thetas[4], T04)
        return np.array([self.T00, T01, T02, T03, T04, T05])


    def solve_IK(self, o, rpy=None) -> IKResult:
        # TODO: Add a parameter to toggle range checks
        """
        Updates theta parameters via the inverse kinematic equations. Also checks if the thetas are
        valid given our angle constraints (theta in [-pi/2, pi/2] (to be added later))

        Parameters
        ----------
        o: list[float]
            The desired x, y, z coordinates of the end-effector.
        R: 2D numpy array
            The desired orientation of the end-effector. Given as a rotation matrix 
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

        # NOTE: The atan2 function is of the form atan2(y, x), not atan2(x, y) 

        # For the 5DOF arm, we do not have control over the yaw independently of theta1, so we must derive theta1 first in order to derive R.
        # We can derive theta1 based on x and y rather than xc and yc because we cannot yaw the end-effector with respect to the wrist centre
        # There is a caveat to this, dealt with below
        th1 = atan2(o[1], o[0])

        if rpy is None:
            R = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
        else:
            (r, p, y) = np.deg2rad(rpy)
            # For this 5DOF arm, we do not have control over the yaw independently of theta1 like in the 6DOF arm, that is, the yaw is now dependent on theta 1
            # A yaw of 90 corresponds to theta1 being 0, due to the way the robot arm is structured
            y = np.pi/2 + th1
            print(f"roll={np.rad2deg(r)}, pitch={np.rad2deg(p)}, yaw={np.rad2deg(y)}")
            R = np.array([
                [cos(y)*cos(p), -sin(y)*cos(r)+cos(y)*sin(p)*sin(r),  sin(y)*sin(r)+cos(y)*sin(p)*cos(r)],
                [sin(y)*cos(p),  cos(y)*cos(r)+sin(y)*sin(p)*sin(r), -cos(y)*sin(r)+sin(y)*sin(p)*cos(r)],
                [      -sin(p),                       cos(p)*sin(r),                       cos(p)*cos(r)]
            ])
            print(f"R = {R}")

        # When the x or y coordinate is less than d5 we run into an issue, as atan2(y, x) != atan2(oy, ox) anymore
        # Since this only occurs when theta should actually be flipped 180 degrees, we just flip it
        if -self.d[4] < abs(o[1])+abs(o[0]) < self.d[4]:
            if th1 > 0:
                th1 -= np.pi
            else:
                th1 += np.pi

        oc = o - self.d[4]*R[:,2] 
        print(f"o:{o}, oc:{oc}")

        D = (oc[0]**2 + oc[1]**2 + (oc[2] - self.d[0])**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
        # NOTE: |D|>1 implies th3 cannot be computed, and hence th2 cannot be computed. This only occurs when a o_c is unreachable by the elbow manipulator.
        if abs(D) > 1:
            return IKResult(False, error=f"Configuration impossible: D={D}")
        th3_up = atan2(-sqrt(1 - D**2), D)
        th3_down = atan2(sqrt(1 - D**2), D)
        th2_up = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_up), self.a[1]+self.a[2]*cos(th3_up))
        th2_down = atan2(oc[2] - self.d[0], sqrt(oc[0]**2 + oc[1]**2)) - atan2(self.a[2]*sin(th3_down), self.a[1]+self.a[2]*cos(th3_up))
        # Once the first three angles have been calculated, make adjustments to th3 since the x axes are actually 90 degrees apart in the "default configuration" as draw in the DH frame, which differs 
        # from how they are expressed the in the geometric solution of the inverse position problem
        # th3_up = np.pi/2 - th3_down
        # th3_down += np.pi/2

        def inverse_orientation(th1, th2, th3):

            s4 = cos(th1)*cos(th2+th3)*R[0,2]+sin(th1)*cos(th2+th3)*R[1,2]+sin(th2+th3)*R[2,2]
            c4 = cos(th1)*sin(th2+th3)*R[0,2]+sin(th1)*sin(th2+th3)*R[1,2]-cos(th2+th3)*R[2,2]
            s5 = sin(th1)*R[0,0]-cos(th1)*R[1,0]
            c5 = sin(th1)*R[0,1]-cos(th1)*R[1,1]

            th5 = atan2(s5, c5)
            th4 = atan2(s4, c4)
            return (th4, th5)
        
        th4_up, th5_up = inverse_orientation(th1, th2_up, th3_up)
        th4_down, th5_down = inverse_orientation(th1, th2_down, th3_down)

        return IKResult(True, solutions={"up": [th1, th2_up, th3_up, th4_up, th5_up], "down": [th1, th2_down, th3_down, th4_down, th5_down]})
