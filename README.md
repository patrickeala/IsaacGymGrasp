# IsaacGymGrasp
IsaacGymGrasp runs a robot grasping physics simulator that can visualize, execute, and evaluate numerous robot grasps in simultaneous environments.

Below is an example of how IsaacGymGrasp evaluates the quality of grasps.

![isaac_demo](https://user-images.githubusercontent.com/47981615/224977728-4952720e-0fb7-470e-836f-8861064573ab.gif)

# Method
## Simulator Setup
The objective is to take a target object and evaluate the success of multiple different grasps on that object. The simulator executes these grasps on the object and labels them based on their grasping success. The simulator generates numerous environments and loads a given target object in a neutral position. Each of these environments will test a different grasp pose. For each environment, a gripper is loaded into a unique pre-grasp position.

## Grasping Sequence
Once the gripper is in the pre-grasp position, the grasping sequence is executed. First, the gripper approaches the object until it reaches the final grasp pose. Then, the gripper fingers are closed with 80N. After this, the gripper performs a preset shaking motion to test the grasps' robustness to movement. Finally, the simulator evaluates whether or not the object was successfully grasped between the gripper fingers.   

# Installation

# Usage



