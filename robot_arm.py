import numpy as np
import kinematics

class RobotArm:
    
    def __init__(self, kinematics: kinematics.Kinematics, th: list[float], joint_limits):
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
        self.joint_limits = joint_limits

        # Initialise link positions and coordinate axes by running forward kinematics
        self.update_link_positions()
        self.end_effector_pos = self.link_points[-1]
        # NOTE: Roll is around X, Pitch is around Y, Yaw is around Z
        self.end_effector_orient = (90, 0, 90)

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

    def validate_solution(self, thetas: list[float]) -> list[str]:
        """Check robot-specific constraints on a candidate solution."""
        errors = []

        # Joint limits
        for i, (theta, (lo, hi)) in enumerate(zip(thetas, self.joint_limits)):
            if not (np.deg2rad(lo) <= theta <= np.deg2rad(hi)):
                errors.append(f"Joint {i+1} out of range ({lo}, {hi}): {np.rad2deg(theta):.2f} deg")

        return errors

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
                up_errors = self.validate_solution(result.solutions["up"])
                down_errors = self.validate_solution(result.solutions["down"])
                if len(up_errors) > 0 and len(down_errors) > 0:
                    print("Both configurations invalid: ")
                    up_errors.extend(down_errors)
                    for err in up_errors:
                        print(err)
                    self.end_effector_pos = old_end_effector_pos
                    self.end_effector_orient = old_end_effector_orient
                elif len(down_errors) > 0:
                    print("UP configuration chosen")
                    self.th = result.solutions["up"]
                    # Update the link positions via forward kinematics
                    print(f"IK Thetas: {np.rad2deg(self.th)}\n")
                    self.update_link_positions()
                elif len(up_errors) > 0:
                    print("DOWN configuration chosen")
                    self.th = result.solutions["down"]
                    # Update the link positions via forward kinematics
                    print(f"IK Thetas: {np.rad2deg(self.th)}\n")
                    self.update_link_positions()
                else:
                    print("Both configs valid, UP chosen")
                    self.th = result.solutions["up"]
                    # Update the link positions via forward kinematics
                    print(f"IK Thetas: {np.rad2deg(self.th)}\n")
                    self.update_link_positions()

                print(f"EE is at: {self.end_effector_pos}, orientation: {self.end_effector_orient}")
        else:
            print("ERROR: z <= 0 - invalid position given")

