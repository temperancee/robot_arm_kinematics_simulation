import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from numpy import atan2 #temporary
import kinematics

""" ---- DH Paramters and position and orientation ---- """
# We record the DH paramters pertaining to joint length, so we know how long the links are

d1 = 20
a2 = 20
a3 = 20
d5 = 15
th: list[float] = [0, 0 ,0]

# Initialise end effector position by running forward kinematics
o = kinematics.T03(*th)[:3, 3]


""" ---- Setup ---- """
# Data setup 

# Specify the origins of the coordinate frames, given in terms of frame 0, when the robot is 
# in the position drawn in the DH diagrams.
# TODO: There has to be a better way to instantiate this...
link_points = np.array([
    [0, 0, 0],         # o0
    [0, 0, d1],        # o1
    [a2, 0, d1],       # o2
    [a2+a3, 0, d1],    # o3
    # [a2+a3, 0, d1],    # o4
    # [a2+a3+d5, 0, d1], # o5
])

# Axes setup
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
max_robot_len = a2+a3+d5+20
ax.set(xlim=[-max_robot_len, max_robot_len], ylim=[-max_robot_len, max_robot_len], zlim=[0, max_robot_len], xlabel='x', ylabel='y', zlabel='z')
ax.view_init(elev=30, azim=-60)

# Initial plot
arm_plot = ax.plot3D(link_points[:, 0], link_points[:, 1], link_points[:, 2])[0]



""" ---- Plot control ---- """

def x_incr_callback(event):
    """
    Callback ran when the "Increase x" button is clicked

    Simply updates the global desired x coordinate and calls IK_update_thetas to update the IK.
    """

    o[0] += 1
    IK_update_thetas()

def y_incr_callback(event):
    """
    Callback ran when the "Increase y" button is clicked

    Simply updates the global desired y coordinate and calls IK_update_thetas to update the IK.
    """

    o[1] += 1
    IK_update_thetas()
def z_incr_callback(event):
    """
    Callback ran when the "Increase z" button is clicked

    Simply updates the global desired z coordinate and calls IK_update_thetas to update the IK.
    """

    o[2] += 1
    IK_update_thetas()
def x_decr_callback(event):
    """
    Callback ran when the "Increase x" button is clicked

    Simply updates the global desired x coordinate and calls IK_update_thetas to update the IK.
    """

    o[0] -= 1
    IK_update_thetas()
def y_decr_callback(event):
    """
    Callback ran when the "Decrease y" button is clicked

    Simply updates the global desired y coordinate and calls IK_update_thetas to update the IK.
    """

    o[1] -= 1
    IK_update_thetas()
def z_decr_callback(event):
    """
    Callback ran when the "Decrease z" button is clicked

    Simply updates the global desired z coordinate and calls IK_update_thetas to update the IK.
    """

    o[2] -= 1
    IK_update_thetas()

def IK_update_thetas() -> None:
    """
    Updates theta parameters via the inverse kinematic equations, using slider values for o

    R is currently implemented in an... odd way. Since this is a 5DOF arm, there are many
    constraints on the value of R. To avoid this for the moment, we simply have the end 
    effector face "straight ahead" in the direction specified by theta1.
    This involves first rotating the base frame into the "standard" (as specified in the diagram)
    end effector frame, where Z is the approach axis, Y is the sliding axis, and X is normal to
    them. 
    We then rotate by theta1 around the base Z axis (pre-multiply) (Note, this is different
    from R^0_1, as that also rotates to have Z aligned with the 2nd joint (motor)).

    Ignore all the above for now - we don't need R until we add the final two joints.
    While we just work on the elbow manipulator, R is irrelevant, apart from determining
    where the wrist centre lies. We simply denote oc=o for now.
    """

    # R_EE = np.array([
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [0, 1, 0]
    # ])


    global th
    # Error check the IK solution - if it fails, print where it failed, and do not update arm
    # TODO: Clean up the error checking, this is all a bit janky
    status, ret = kinematics.inverse_kinematics(o, elbow_up=True)
    if not status:
        print(f"ERROR: ik_failed with {ret}")
    else:
        for i, theta in enumerate(ret):
            th[i] = theta # type: ignore # Annoyingly thinks th[i] is an int, and theta could be a string...
        update_figure()

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

def update_figure():
    """
    Update the plot of the arm via forward kinematics
    """
    link_points = np.array([
        [0, 0, 0],         # o0
        kinematics.T01(th[0])[:3,3],        # o1
        kinematics.T02(th[0], th[1])[:3,3],       # o2
        kinematics.T03(th[0], th[1], th[2])[:3,3],    # o3
        # [a2+a3, 0, d1],    # o4
        # [a2+a3+d5, 0, d1], # o5
    ])
    # TODO: This should update the angle sliders, and the pos sliders
    arm_plot.set_data_3d(link_points[:, 0], link_points[:, 1], link_points[:, 2]) # type: ignore # This is not an error: upsettingly, due to the way Axes3D.plot/Axes3D.plot3D is written, type interface sees it as returning ArrayOf(Line2D), when actually, it returns ArrayOf(Line3D)
    fig.canvas.draw_idle()



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
