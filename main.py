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

        self.kinematics = kinematics

        self.th: list[float] = [0, 0 ,0]
        # Initialise end effector position by running forward kinematics
        self.end_effector_pos = self.kinematics.end_effector_tmatrix(self.th).position
        # Specify the origins of the coordinate frames, given in terms of frame 0, when the robot is 
        # in the position drawn in the DH diagrams.
        self.link_points = self.kinematics.full_FK_pos(self.th)


    def set_end_effector_pos(self, x, y, z, elbow_up=True):
        """ Updates the robot link configuration via inverse kinematics """
        self.end_effector_pos = [x, y, z]
        # TODO: Clean up this error checking; it seems a bit grim
        success, ret = self.kinematics.solve_IK(self.end_effector_pos, elbow_up=elbow_up)
        if not success:
            print(f"ERROR: IK failed with {ret}")
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

        # Initial plot
        self.arm_plot = self.ax.plot3D(robot.link_points[0], robot.link_points[1], robot.link_points[2])[0]


    def update_figure(self):
        """
        Update the plot of the arm
        """
        # TODO: This should also update the angle sliders, and the pos sliders
        self.arm_plot.set_data_3d(self.robot.link_points[:, 0], self.robot.link_points[:, 1], self.robot.link_points[:, 2]) # type: ignore # This is not an error: upsettingly, due to the way Axes3D.plot/Axes3D.plot3D is written, type interface sees it as returning ArrayOf(Line2D), when actually, it returns ArrayOf(Line3D)
        self.fig.canvas.draw_idle()



""" ---- Plot control ---- """

class Move:
    def __init__(self, robot: RobotArm, plotter: Plotter, dx=0, dy=0, dz=0):
        self.robot = robot
        self.plotter = plotter
        self.dx = dx
        self.dy = dy
        self.dz = dz


    def execute(self, event=None):
        """
        Executes the given movement command, updates the robot accordingly, and plots the result on the figure

        Parameters
        ----------
        event (default: None):
           Required by matplotlib Button.on_clicked method. Unused. 
        """
        x, y, z = self.robot.end_effector_pos
        self.robot.set_end_effector_pos(x + self.dx, y + self.dy, z + self.dz)
        self.plotter.update_figure()
        

""" -------------- Old --------------- """

def manual_update_thetas(val: float) -> None:
    """
    Updates theta parameters via sliders, causing the arm to move

    Parameters
    ----------
    val : float
        Required by Slider.on_changed. Stores the value of the slider
        that calls this function. Not used here.
    """
    global th 
    for i, s in enumerate(angle_sliders):
        th[i] = np.deg2rad(s.val)
    update_figure()



""" ---- Widgets ---- """
ax_angle_sliders = [fig.add_axes((0.175, 0.01 + i*0.03, 0.65, 0.02)) for i in range(len(th))]
angle_sliders = []
for i in range(len(th)):
    b = Slider(ax_angle_sliders[i], f"Theta {i}", -180, 180, valinit=0)
    b.on_changed(manual_update_thetas)
    angle_sliders.append(b)

# Specifies the position and dimensions of the buttons 
button_width, button_height = 0.02, 0.04
rectangles = [
    (0.01 + 0.03,   0.775,        button_width, button_height),
    (0.01 + 0.03,   0.775 - 0.12, button_width, button_height),
    (0.01 + 0.03*2, 0.775,        button_width, button_height),
    (0.01 + 0.03*2, 0.775 - 0.12, button_width, button_height),
    (0.01 + 0.03*3, 0.775,        button_width, button_height),
    (0.01 + 0.03*3, 0.775 - 0.12, button_width, button_height)
]
# The order of the buttons in the list is "incr x, decr x, incr y, decr y, ..."
ax_pos_buttons = [fig.add_axes(rect) for rect in rectangles]
# Add text axes and text
axes_names = ["x", "y", "z"]
for i in range(3):
    fig.text(0.045 + 0.03*i, 0.775 - 0.05, axes_names[i])
position_buttons = []
button_callbacks = [x_incr_callback, y_incr_callback, z_incr_callback, x_decr_callback, y_decr_callback, z_decr_callback]
# Get the initial x, y, z coordinates when the angles are all 0
initial_pos = kinematics.T03(0, 0, 0)[:3, 3]
for i in range(6):
    b = Button(ax_pos_buttons[i], "^" if i%2==0 else "v")
    b.on_clicked(button_callbacks[i])
    position_buttons.append(b)
    # TODO: Add some text specifying what x y and z are currently equal to on the figure
    # TODO: Each function will add/subtract 1 from its respective axis. The values for x
    # y and z will be stored in a global variable. The text boxes on screen will pull from
    # this global, and be updated in update_figure. Following the addition/subtraction in
    # the button function, IK_update_thetas will be called, with the x y z values (in an 
    # array suitably called o) passed as a parameter.


""" ---- Run ---- """
plt.show()
