import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import kinematics

class RobotArm:
    
    def __init__(self, kinematics: kinematics.Kinematics):
        """
        Parameters
        ----------
        kinematics: kinematics.Kinematics
            The kinematic configuration of the robot, one of: ElbowKinematics, 5DOFKinematics, 6DOFKinematics
        """

        # Save parameters
        self.kinematics = kinematics

        self.th: list[float] = [0, np.pi/6, -np.pi/6]
        # Initialise end effector position by running forward kinematics
        self.end_effector_pos = self.kinematics.end_effector_tmatrix(self.th).position
        self.update_link_positions()


    def update_link_positions(self):
        # Specify the origins of the coordinate frames, given in terms of frame 0, when the robot is 
        # in the position drawn in the DH diagrams.
        self.link_points = self.kinematics.full_FK_pos(self.th)


    def set_end_effector_pos(self, x, y, z, elbow_up=True):
        """ Updates the robot link configuration via inverse kinematics """

        old_end_effector_pos = self.end_effector_pos
        self.end_effector_pos = [x, y, z]
        # TODO: Clean up this error checking; it seems a bit grim
        success, ret = self.kinematics.solve_IK(self.end_effector_pos, elbow_up=elbow_up)
        if not success:
            print(f"ERROR: IK failed with {ret}")
            self.end_effector_pos = old_end_effector_pos
        else:
            self.th = ret # type: ignore # Annoyingly, the type checker can't see that ret must be of type list[float] in this section
            # Update the link positions via forward kinematics
            self.link_points = self.kinematics.full_FK_pos(self.th)



class Plotter:

    def __init__(self, robot: RobotArm):
        # Save paramters
        self.robot = robot

        # Axes setup
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        max_robot_len = self.robot.kinematics.max_robot_len
        self.ax.set(xlim=[-max_robot_len, max_robot_len], ylim=[-max_robot_len, max_robot_len], zlim=[0, max_robot_len], xlabel='x', ylabel='y', zlabel='z')
        self.ax.view_init(elev=30, azim=-60)

        # Store widgets to prevent garbage collection
        self.buttons = []
        self.sliders = []
        # Store coordinate_values to update later
        self.coordinate_values = []

        # Create widgets
        self._create_widgets()

        # Initial plot
        self.arm_plot = self.ax.plot3D(robot.link_points[:, 0], robot.link_points[:, 1], robot.link_points[:, 2])[0]


    def _create_widgets(self):
        """ 
        Creates buttons to control the end effector coordinates and update the plot via IK,
        and sliders to control the joint angles directly and update the plot via FK
        """

        # Buttons
        # Specifies the name, position and dimensions, and commands of the buttons 
        button_width, button_height = 0.02, 0.04
        button_spec = [
            ("^", (0.01 + 0.03,      0.775,        button_width, button_height), Move(self.robot, self, dx=1).execute),
            ("^", (0.01 + 0.03+0.05, 0.775,        button_width, button_height), Move(self.robot, self, dy=1).execute),
            ("^", (0.01 + 0.03+0.10, 0.775,        button_width, button_height), Move(self.robot, self, dz=1).execute),
            ("v", (0.01 + 0.03,      0.775 - 0.15, button_width, button_height), Move(self.robot, self, dx=-1).execute),
            ("v", (0.01 + 0.03+0.05, 0.775 - 0.15, button_width, button_height), Move(self.robot, self, dy=-1).execute),
            ("v", (0.01 + 0.03+0.10, 0.775 - 0.15, button_width, button_height), Move(self.robot, self, dz=-1).execute)
        ]
        # Add button text
        for i, label in enumerate(["x", "y", "z"]):
            self.fig.text(0.045 + 0.05*i, 0.775 - 0.05, label)
        # Add text showing what x y and z equal
        for i, value in enumerate(self.robot.end_effector_pos):
            self.coordinate_values.append(self.fig.text(0.035 + 0.05*i, 0.775 - 0.10, f"{value:.2f}"))
        for (label, rect, command) in button_spec:
            ax_button = self.fig.add_axes(rect)
            button = Button(ax_button, label)
            # Button.on_clicked requires an event parameter, but we don't use it
            button.on_clicked(lambda event, cmd=command: cmd())
            self.buttons.append(button)

        # Sliders
        ax_angle_sliders = [self.fig.add_axes((0.175, 0.01 + i*0.03, 0.65, 0.02)) for i in range(len(self.robot.th))]
        for i in range(len(self.robot.th)):
            b = Slider(ax_angle_sliders[i], f"Theta {i}", -180, 180, valinit=0)
            # Slider.on_changes requires an event parameter, but we don't use it
            b.on_changed(lambda event, cmd=MoveAngle(self.robot, self, i).execute: cmd())
            self.sliders.append(b)



    def update_figure(self):
        """
        Update the plot of the arm
        """
        # TODO: The IK and FK controls should update each others slider/text fields - right now they are completely decoupled

        # Update position text 
        for i, value in enumerate(self.robot.end_effector_pos):
            self.coordinate_values[i].set_text(f"{value:.2f}")

        self.arm_plot.set_data_3d(self.robot.link_points[:, 0], self.robot.link_points[:, 1], self.robot.link_points[:, 2]) # type: ignore # This is not an error: upsettingly, due to the way Axes3D.plot/Axes3D.plot3D is written, type interface sees it as returning ArrayOf(Line2D), when actually, it returns ArrayOf(Line3D)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()



""" ---- Plot control ---- """

class Move:
    def __init__(self, robot: RobotArm, plotter: Plotter, dx=0, dy=0, dz=0):
        self.robot = robot
        self.plotter = plotter
        self.dx = dx
        self.dy = dy
        self.dz = dz


    def execute(self):
        """
        Executes the given movement command, updates the robot accordingly, and plots the result on the figure
        """
        x, y, z = self.robot.end_effector_pos
        self.robot.set_end_effector_pos(x + self.dx, y + self.dy, z + self.dz)
        self.plotter.update_figure()

class MoveAngle:
    def __init__(self, robot: RobotArm, plotter: Plotter, th_index: int):
        """
        Parameters
        ----------
        th_index: int
            The index of the list of thetas. This determines which theta this instance edits.
        """
        self.robot = robot
        self.plotter = plotter
        self.th_index = th_index

    def execute(self):
        """
        Executes the given movement command by updating the robot accordingly, and plots the result on the figure
        """
        print(self.th_index)
        self.robot.th[self.th_index] = np.deg2rad(self.plotter.sliders[self.th_index].val)
        print(self.robot.th[self.th_index])
        self.robot.update_link_positions()
        self.plotter.update_figure()



        

def main():
    """ ---- Run ---- """
    d1 = 20
    a2 = 20
    a3 = 20
    d5 = 15
    # Probably more readable as a dict
    dh_parameters = [
        [0, a2, a3], # a
        [0,  0,  0], # alpha
        [d1, 0,  0]  # d
    ]
    elbow_kinematics = kinematics.ElbowKinematics(*dh_parameters)
    robot = RobotArm(elbow_kinematics)
    plotter = Plotter(robot)
    plotter.show()


if __name__ == '__main__':
    main()
