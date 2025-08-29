import numpy as np
import kinematics

class RobotArm:
    
    def __init__(self, kinematics: kinematics.Kinematics, th: list[float]):
        """
        Parameters
        ----------
        kinematics: kinematics.Kinematics
            The kinematic configuration of the robot, one of type: ElbowKinematics, 5DOFKinematics, 6DOFKinematics
        th: list[float]
            The initial joint angles of the robot
        """

        # Save parameters
        self.kinematics = kinematics
        self.th = th

        # Initialise link positions and coordinate axes by running forward kinematics
        self.update_link_positions()
        self.end_effector_pos = self.link_points[-1]
        # NOTE: Roll is around X, Pitch is around Y, Yaw is around Z
        self.end_effector_orient = (90, 0, 90)

        # Initialise constraints 
        self.joint_limits = [
            (-np.pi, np.pi) * len(self.th)
        ]

    def update_link_positions(self):
        # Specify the origins of the coordinate frames, given in terms of frame 0, when the robot is 
        # in the position drawn in the DH diagrams.
        fk = self.kinematics.full_FK(self.th)
        frame_axes = []
        link_points = []
        for tm in fk:
            frame_axes.append(tm.rotation)
            link_points.append(tm.position)
        self.frame_axes = np.array(frame_axes)
        self.link_points = np.array(link_points)


    def set_end_effector_pos(self, x, y, z, rpy=None):
        """
        Updates the robot link configuration via inverse kinematics 

        Compares the Euclidean distance between old and new thetas for  both sets of thetas returned from the IK function
        (representing the elbow-up and elbow-down configurations) and sets theta to the closest configuration.
        """
        # TODO: The Euclidean distance didn't fix the issue - th1 still flips to 180 from 0 sometimes. We might want to
        #       fix this by passing thetas to solve_IK() and then making a direct comparison 


        # Check that the end effector position given is not in the floor before running IK
        if z > 0:
            # Save old pos and orient in case IK fails
            old_end_effector_pos = self.end_effector_pos
            old_end_effector_orient = self.end_effector_orient
            self.end_effector_pos = [x, y, z]
            if rpy is not None:
                self.end_effector_orient = rpy
            result = self.kinematics.solve_IK(self.end_effector_pos, rpy)
            if not result.success:
                print(f"ERROR: IK failed with {result.error}")
                self.end_effector_pos = old_end_effector_pos
                self.end_effector_orient = old_end_effector_orient
            else:
                # Generally, we choose the configuration that is closest to the previous one. When there is a large flip in th1, however, we choose the opposite, as this will be a more natural position
                up_closer = abs(self.th[1] - result.solutions["down"][1]) > abs(self.th[1] - result.solutions["up"][1]) - 1
                th_180_change = abs(self.th[0] - result.solutions["up"][0]) > 3 # it's in radians, remember
                print(f"old_theta: {self.th[0]} new_theta: {result.solutions["up"][0]}")
                if up_closer and th_180_change:
                    self.th = result.solutions["down"]
                    print("DOWN PICKED")
                elif th_180_change:
                    self.th = result.solutions["up"]
                    print("TH180 and UP PICKED")
                elif up_closer:
                    self.th = result.solutions["up"]
                    print("UP PICKED")
                else:
                    self.th = result.solutions["down"]
                    print("DOWN PICKED")

                # Update the link positions via forward kinematics
                print(f"IK Thetas: {np.rad2deg(self.th)}")
                self.update_link_positions()
        else:
            print("ERROR: z <= 0 - invalid position given")


    # def choose_solution(self, thetas: )

    def validate_solution(self, thetas: list[float], x: float, y: float, z: float) -> list[str]:
        """Check robot-specific constraints on a candidate solution."""
        errors = []

        # Joint limits
        for i, (theta, (lo, hi)) in enumerate(zip(thetas, self.joint_limits)):
            if not (lo < theta <= hi):
                errors.append(f"Joint {i+1} out of range: {theta:.2f} rad")


        return errors
