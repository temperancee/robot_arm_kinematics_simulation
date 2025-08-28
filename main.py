from abc import abstractmethod
from typing import Iterable
from matplotlib import animation
from matplotlib.artist import Artist
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
            result = self.kinematics.solve_IK(self.th[0], self.end_effector_pos, rpy)
            if not result.success:
                print(f"ERROR: IK failed with {result.error}")
                self.end_effector_pos = old_end_effector_pos
                self.end_effector_orient = old_end_effector_orient
            else:
                # Just choose up always for now - choose the solution who's value of theta1 is closest to the previous value
                if abs(self.th[0] - result.solutions["up"][0]) > abs(self.th[0] - result.solutions["up_alt"][0]):
                    self.th = result.solutions["up_alt"]
                    print("ALT PICKED")
                    print(f"ALT: {np.rad2deg(result.solutions["up_alt"][0])}, STANDARAD: {np.rad2deg(result.solutions["up"][0])}")
                else:
                    self.th = result.solutions["up"]
                    print("STANDARD PICKED")
                    print(f"ALT: {np.rad2deg(result.solutions["up_alt"][0])}, STANDARAD: {np.rad2deg(result.solutions["up"][0])}")
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
        # Store current end effector coordinates and orientation
        self.coordinate_values = []
        self.orient_values = []

        # Create widgets
        self._draw_widgets()

        # Initial plot
        self.arm_plot = self.ax.plot3D(robot.link_points[:, 0], robot.link_points[:, 1], robot.link_points[:, 2])[0]
        self.axes_plots = []
        self.draw_frame_axes()


    def draw_frame_axes(self, initial=True):
        """
        This loops over all the frames and origins and draws the axes in x=red, y=green, z=blue
        """

        i=0
        for frame, origin in zip(self.robot.frame_axes, self.robot.link_points):
            for j, col in zip(range(3), ['red', 'green', 'blue']):
                axis_line = np.array([
                    origin,
                    origin + 5*frame[:, j]
                ])
                #print(origin + 5*frame[:, j])
                if initial:
                    self.axes_plots.append(self.ax.plot3D(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color=col)[0])
                else:
                    self.axes_plots[i].set_data_3d(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2])
                    i+=1


    def _draw_widgets(self):
        """ 
        Creates buttons to control the end effector coordinates and update the plot via IK,
        and sliders to control the joint angles directly and update the plot via FK
        """

        # Buttons
        # Specifies the name, position and dimensions, and commands of the buttons 
        button_width, button_height = 0.02, 0.04
        button_spec = [
            ("^", (0.01 + 0.03,      0.775,        button_width, button_height), MovePos(self.robot, self, dx=5).execute),
            ("^", (0.01 + 0.03+0.05, 0.775,        button_width, button_height), MovePos(self.robot, self, dy=5).execute),
            ("^", (0.01 + 0.03+0.10, 0.775,        button_width, button_height), MovePos(self.robot, self, dz=5).execute),
            ("v", (0.01 + 0.03,      0.775 - 0.15, button_width, button_height), MovePos(self.robot, self, dx=-5).execute),
            ("v", (0.01 + 0.03+0.05, 0.775 - 0.15, button_width, button_height), MovePos(self.robot, self, dy=-5).execute),
            ("v", (0.01 + 0.03+0.10, 0.775 - 0.15, button_width, button_height), MovePos(self.robot, self, dz=-5).execute),

            ("^", (0.01 + 0.03,      0.55,        button_width, button_height), MovePos(self.robot, self, dr=5).execute),
            ("^", (0.01 + 0.03+0.05, 0.55,        button_width, button_height), MovePos(self.robot, self, dp=5).execute),
            ("^", (0.01 + 0.03+0.10, 0.55,        button_width, button_height), MovePos(self.robot, self, dyaw=5).execute),
            ("v", (0.01 + 0.03,      0.55 - 0.15, button_width, button_height), MovePos(self.robot, self, dr=-5).execute),
            ("v", (0.01 + 0.03+0.05, 0.55 - 0.15, button_width, button_height), MovePos(self.robot, self, dp=-5).execute),
            ("v", (0.01 + 0.03+0.10, 0.55 - 0.15, button_width, button_height), MovePos(self.robot, self, dyaw=-5).execute)
        ]
        # Add button text
        for i, label in enumerate(["x", "y", "z"]):
            self.fig.text(0.045 + 0.05*i, 0.775 - 0.05, label)
        for i, label in enumerate(["r", "p", "y"]):
            self.fig.text(0.045 + 0.05*i, 0.55 - 0.05, label)
        # Add value text
        for i, value in enumerate(self.robot.end_effector_pos):
            self.coordinate_values.append(self.fig.text(0.035 + 0.05*i, 0.775 - 0.10, f"{value:.1f}"))
        for i, value in enumerate(self.robot.end_effector_orient):
            self.orient_values.append(self.fig.text(0.035 + 0.05*i, 0.55 - 0.10, f"{value:.1f}"))
        # Draw buttons
        for (label, rect, command) in button_spec:
            ax_button = self.fig.add_axes(rect)
            button = Button(ax_button, label)
            # Button.on_clicked requires an event parameter, but we don't use it
            button.on_clicked(lambda event, cmd=command: cmd())
            self.buttons.append(button)

        # Sliders
        ax_angle_sliders = [self.fig.add_axes((0.175, 0.01 + i*0.03, 0.65, 0.02)) for i in range(len(self.robot.th))]
        for i in range(len(self.robot.th)):
            b = Slider(ax_angle_sliders[i], f"Theta {i+1}", -180, 180, valinit=np.rad2deg(self.robot.th[i]))
            # Slider.on_changes requires an event parameter, but we don't use it
            b.on_changed(lambda event, cmd=MoveAngle(self.robot, self, i).execute: cmd())
            self.sliders.append(b)



    def update_figure(self):
        """
        Update the plot of the arm
        """
        # TODO: The IK and FK controls should update each others slider/text fields - right now they are completely decoupled

        # Update position and orientation text 
        for i, value in enumerate(self.robot.end_effector_pos):
            self.coordinate_values[i].set_text(f"{value:.1f}")
        for i, value in enumerate(self.robot.end_effector_orient):
            self.orient_values[i].set_text(f"{value:.1f}")


        self.arm_plot.set_data_3d(self.robot.link_points[:, 0], self.robot.link_points[:, 1], self.robot.link_points[:, 2]) # type: ignore # This is not an error: upsettingly, due to the way Axes3D.plot/Axes3D.plot3D is written, type interface sees it as returning ArrayOf(Line2D), when actually, it returns ArrayOf(Line3D)
        self.draw_frame_axes(initial=False)
        self.fig.canvas.draw_idle()


    def _flair_update(self, frame) -> Iterable[Artist]:
        # goto x=40
        if frame<41:
            MovePos(self.robot, self, auto_update=False, dx=-1).execute()
        # goto z=5
        elif frame<66:
            MovePos(self.robot, self, auto_update=False, dz=-1).execute()
        # goto y = 55
        elif frame<121:
            MovePos(self.robot, self, auto_update=False, dy=1).execute()
        # Square
        elif frame<161:
            MovePos(self.robot, self, auto_update=False, dz=1.6).execute()
        elif frame<241:
            MovePos(self.robot, self, auto_update=False, dy=-1).execute()
        elif frame<281:
            MovePos(self.robot, self, auto_update=False, dz=-1.6).execute()
        elif frame<361:
            MovePos(self.robot, self, auto_update=False, dy=1).execute()
        # End of square
        # Return to y=0
        elif frame<416:
            MovePos(self.robot, self, auto_update=False, dy=-1).execute()
        # Final pose
        elif frame<456:
            MovePos(self.robot, self, auto_update=False, dx=-0.8, dz=1.5).execute()
        elif frame<491:
            MovePos(self.robot, self, auto_update=False, dz=0.5).execute()

        self.arm_plot.set_data_3d(self.robot.link_points[:, 0], self.robot.link_points[:, 1], self.robot.link_points[:, 2]) # type: ignore # This is not an error: upsettingly, due to the way Axes3D.plot/Axes3D.plot3D is written, type interface sees it as returning ArrayOf(Line2D), when actually, it returns ArrayOf(Line3D)
        self.draw_frame_axes(initial=False)
        plots = [self.arm_plot]
        plots.extend(self.axes_plots)
        return plots


    def flair_animation(self):
        self.ani = animation.FuncAnimation(fig=self.fig, func=self._flair_update, frames=490, interval=30, repeat=False)
        # self.ani.save(filename="/home/alexander/Videos/saved-videos/elbow_manipulator_sim.gif", writer="ffmpeg", dpi=550)


    def show(self):
        plt.show()



""" ---- Plot control ---- """

class Command:
    def __init__(self, robot: RobotArm, plotter: Plotter) -> None:
        self.robot = robot
        self.plotter = plotter

    @abstractmethod
    def execute(self):
        """
        Executes the given movement command, updates the robot accordingly, and plots the result on the figure
        """
        pass



class MovePos(Command):
    def __init__(self, robot: RobotArm, plotter: Plotter, auto_update=True, dx=0.0, dy=0.0, dz=0.0, dr=0.0, dp=0.0, dyaw=0.0):
        super().__init__(robot, plotter)
        self.auto_update = auto_update
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dr = dr
        self.dp = dp
        self.dyaw = dyaw

    def execute(self):
        x, y, z = self.robot.end_effector_pos
        (r, p, yaw) = self.robot.end_effector_orient
        self.robot.set_end_effector_pos(x + self.dx, y + self.dy, z + self.dz, (r + self.dr, p + self.dp, yaw + self.dyaw))
        if self.auto_update:
            self.plotter.update_figure()



class MoveAngle(Command):
    def __init__(self, robot: RobotArm, plotter: Plotter, th_index: int):
        """
        Parameters
        ----------
        th_index: int
            The index of the list of thetas. This determines which theta this instance edits.
        """
        super().__init__(robot, plotter)
        self.th_index = th_index

    def execute(self):
        """
        Executes the given movement command by updating the robot accordingly, and plots the result on the figure
        """
        self.robot.th[self.th_index] = np.deg2rad(self.plotter.sliders[self.th_index].val) # type: ignore # Thinks robot.th can be a str due to the error handling in the RobotArm class not working
        self.robot.update_link_positions()
        self.plotter.update_figure()



        

def main():
    """ ---- Run ---- """
    d1 = 30
    a2 = 40
    a3 = 40
    d4 = 40
    d5 = 15
    d6 = 15
    # TODO: Probably more readable as a dict
    elbow_dh_parameters = [
        [0, a2, a3], # a
        [np.pi/2,  0,  0], # alpha
        [d1, 0,  0]  # d
    ]
    six_dof_dh_parameters = [
        [0, a2, 0, 0, 0, 0], # a
        [np.pi/2,  0,  np.pi/2, -np.pi/2, np.pi/2, 0], # alpha
        [d1, 0,  0, d4, 0, d6]  # d
    ]
    # thetas: list[float] = [0, 0, 0]
    thetas: list[float] = [0, 0, np.pi/2, 0, 0, -np.pi/2]
    elbow_kinematics = kinematics.ElbowKinematics(*elbow_dh_parameters)
    six_dof_kinematics = kinematics.SixDOFKinematics(*six_dof_dh_parameters)
    # robot = RobotArm(elbow_kinematics, thetas)
    robot = RobotArm(six_dof_kinematics, thetas)
    plotter = Plotter(robot)
    # plotter.flair_animation()
    plotter.show()


if __name__ == '__main__':
    main()
