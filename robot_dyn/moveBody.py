import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

from robot_descriptions.loaders.pinocchio import load_robot_description
 
robot = load_robot_description("ur5_description")
model = robot.model
collision_model = robot.collision_model
visual_model = robot.visual_model

viz = GepettoVisualizer(model, collision_model, visual_model)

viz.initViewer()
viz.loadViewerModel()

# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
print("display model end.")