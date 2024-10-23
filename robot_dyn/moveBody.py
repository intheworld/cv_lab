import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

from robot_descriptions.loaders.pinocchio import load_robot_description
 
robot = load_robot_description("ur5_description")
model = robot.model
collision_model = robot.collision_model
visual_model = robot.visual_model

viz = GepettoVisualizer(model, collision_model, visual_model)

robot.setVisualizer(viz)
robot.initViewer(loadModel=True)

visualObj = robot.visual_model.geometryObjects[4]  # 3D object representing the robot forarm
visualName = visualObj.name                        # Name associated to this object
visualRef = robot.getViewerNodeName(visualObj, pin.GeometryType.VISUAL)    # Viewer reference (string) representing this object

q1 = (1, 1, 1, 1, 0, 0, 0)  # x, y, z, quaternion
robot.viewer.gui.applyConfiguration(visualRef, q1)
robot.viewer.gui.refresh()  # Refresh the window.


rgbt = [1.0, 0.2, 0.2, 1.0]  # red, green, blue, transparency
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)  # .1 is the radius

robot.viewer.gui.applyConfiguration("world/sphere", (.5, .1, .2, 1.,0.,0.,0. ))
robot.viewer.gui.refresh()  # Refresh the window.