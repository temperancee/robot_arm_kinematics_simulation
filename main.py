from robot_arm import RobotArm
import kinematics
import numpy as np
from plotter import Plotter
        

if __name__ == '__main__':
    # d1 = 30
    # a2 = 40
    # a3 = 40
    d4 = 40
    # d5 = 15
    d6 = 15
    d1 = 114.655 
    a2 = 106.040
    a3 = 106.040
    d5 = 103.740
    # TODO: Probably more readable as a dict

    # Elbow manipulator
    joint_limits = []
    elbow_dh_parameters = [
        [0, a2, a3], # a
        [np.pi/2, 0, 0], # alpha
        [d1, 0, 0]  # d
    ]
    # thetas: list[float] = [0, 0, 0]
    elbow_kinematics = kinematics.ElbowKinematics(*elbow_dh_parameters)
    # robot = RobotArm(elbow_kinematics, thetas, joint_limits)

    # 6DOF manipulator
    joint_limits = []
    six_dof_dh_parameters = [
        [0, a2, 0, 0, 0, 0], # a
        [np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2, 0], # alpha
        [d1, 0, 0, d4, 0, d6]  # d
    ]
    thetas: list[float] = [0, 0, np.pi/2, 0, 0, -np.pi/2]
    # six_dof_kinematics = kinematics.SixDOFKinematics(*six_dof_dh_parameters)
    # robot = RobotArm(six_dof_kinematics, thetas, joint_limits)

    # 5DOF manipulator
    joint_limits = [
        (-90, 90),
        (0, 180),
        (-135, 45),
        (0, 180),
        (-91, 91)
    ]
    five_dof_dh_parameters = [
        [0, a2, a3, 0, 0], # a
        [np.pi/2,  0,  0, np.pi/2, 0], # alpha
        [d1, 0, 0, 0, d5]  # d
    ]
    thetas: list[float] = [0, 0, 0, np.pi/2, 0]
    five_dof_kinematics = kinematics.FiveDOFKinematics(*five_dof_dh_parameters)
    robot = RobotArm(five_dof_kinematics, thetas, joint_limits)


    # Plot configuration
    plotter = Plotter(robot)
    # plotter.flair_animation("accident", 570)
    plotter.show()
