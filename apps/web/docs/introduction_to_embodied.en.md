# Chapter 1 What Is Embodied Intelligence?

## 1.1 A working definition of embodied intelligence

Embodied intelligence studies how an agent uses its body to perceive, make decisions and act in a particular environment to accomplish a task. Its focus extends beyond an information-processing algorithm to **intelligent behavior arising from continuous interaction among an agent, its body and its environment**.

“Embodied” has two related meanings: the agent has a physical or simulated body that can sense and act, and it operates in a particular environment where space, contact and physical laws affect its behavior. A body enables interaction while constraining what the agent can observe, where it can reach and which actions it can execute.

We therefore cannot assess embodied intelligence solely by the computations inside a model. Task performance also depends on perception, bodily constraints, environmental changes and whether execution results inform subsequent decisions. Research often describes this relationship as a synergy of morphology, perception, action and learning.[1]

For example, recognizing an object with a vision model addresses only part of the question “What is in the environment?” The system must also consider arm reachability, gripper geometry, object pose and contact state when choosing an action.

## 1.2 Agent, body, environment and task

Four elements define an embodied intelligence problem.

An **agent** receives information and produces behavior. In an arm task, it may include perception, decision-making and control modules, rather than only a neural network.

The **body** provides the mechanical structure, sensors and actuators used to perceive and act. Link lengths, joint limits, gripper design and camera placement all influence feasible policies. A suction cup and a two-finger gripper, for example, require different contact locations on the same object.

The **environment** is the external system with which the agent interacts. It may include objects, obstacles, lighting, friction and gravity. Actions change the environment, which may also change independently.

The **task** specifies objectives, constraints and success criteria. Even with the same body and environment, different tasks require different information, action sequences and evaluation criteria.

Together, these elements determine whether behavior is effective. A method may fail on another arm because its action range differs; the same controller can also behave differently under different friction conditions. This coupling is central to embodied intelligence.

> **Figure 1-1. Four elements of an embodied intelligence problem.** The body connects the agent to sensors and actuators; body and environment influence each other; the task supplies goals and constraints. The diagram illustrates relationships, rather than four unrelated modules.

## 1.3 The perception–decision–action–feedback loop

An embodied system typically operates in a loop: sensors acquire information, a decision module chooses an action, actuators affect the environment, and sensors observe the result. The new observation informs the next decision.

Let the environment state at time \(t\) be \(s_t\). State describes the environment completely within the chosen model, including joint positions, object poses and contact relationships. The agent usually receives only an observation \(o_t\), such as images, joint angles or force readings.

A **policy** is a rule for choosing an action from available information. We begin with a simplified deterministic policy:

\[
a_t=\pi(o_t),
\]

where \(\pi\) is the policy, \(o_t\) the current observation and \(a_t\) the action at time \(t\). A policy may be a handcrafted rule, a planning algorithm or a neural network.

Likewise, write the environment as a deterministic state transition:

\[
s_{t+1}=T(s_t,a_t).
\]

Here \(T\) represents transition dynamics supplied by physical laws, a simulator or real equipment. The agent then receives \(o_{t+1}\) and repeats the interaction. Real systems may use stochastic policies and stochastic transitions: identical observations need not yield identical actions, and identical state–action pairs need not yield identical next states. These equations introduce the loop, without developing its probabilistic model. State, observation, action and policy recur throughout reinforcement learning.[5]

Feedback closes the loop. Updated position or contact information can correct subsequent behavior. **Feedback is not the same as reward, and a feedback loop does not imply learning.** Feedback returns information to decision-making; learning changes a model, policy or parameters using data or experience.

> **Figure 1-2. The interaction loop.** Sensors turn environment state \(s_t\) into observation \(o_t\); policy \(\pi\) produces action \(a_t\); actuators apply it and the environment reaches \(s_{t+1}\). An animation can introduce a position disturbance after an action to show how the next observation changes the next action.

## 1.4 Control, planning and learning

An embodied system is not a single learning algorithm. Control, planning and learning address different problems and often work together.

**Control** makes robot motion or force approach a desired target. A controller may repeatedly adjust motor commands using joint-position feedback to track a trajectory. Modern Robotics introduces robot feedback control through the relationship among sensor measurements, controller calculations and actuator outputs.[4]

**Planning** constructs a feasible sequence toward a goal. Motion planning finds a path that reaches a target while avoiding obstacles; task planning may divide “clear the table” into recognizing, grasping, moving and placing objects.

**Learning** improves perception, prediction or decisions using data or interaction. A robot may learn actions from demonstrations or improve a policy through trial and error. Learning need not replace classical methods: robotics courses treat model-based and data-driven motion generation as different approaches that can be combined.[6]

An arm system might estimate object position with a learned model, generate a collision-free path with a planner and track it with a controller. Alternatively, a learned policy may output action targets while a lower-level controller executes them. To understand a system, identify each module's role; the presence of a neural network alone does not explain the architecture.

## 1.5 Related concepts and their differences

Embodied intelligence and traditional AI are not mutually exclusive. Computer vision and language processing often focus on extracting information or generating content. Embodied intelligence additionally asks how information becomes action and how action-induced changes feed back into the system. Traditional AI models can serve as perception or decision-making components.

Robots are important embodiments, but “robot” and “embodied intelligence” are not synonyms. A fixed-program industrial arm has a body and contacts its environment, yet its behavior may work only in a predetermined workstation. Embodied intelligence investigates how behavior is formed, selected or adapted under bodily and environmental constraints. Research can begin with simulated robots, provided observations, actions and environmental interaction are explicitly represented.

A **neural network** is a parameterized computational model used for recognition, state estimation or policy representation. It is a tool, not an entire embodied system. A robot without a neural network can accomplish tasks using rules, planning and feedback control; a static image classifier does not become embodied merely because it uses a neural network.

A **large language model** primarily processes and generates language. In robotics, it may interpret instructions, decompose tasks or invoke existing skills. A plausible linguistic plan does not guarantee feasible physical execution: the system still needs to check robot capabilities, the current environment and execution feedback. SayCan combines the relevance of a skill to a language instruction with the robot's ability to execute that skill.[7]

> **Figure 1-3. Information processing and embodied interaction.** Compare “input data → model → output” with “environment → observation → decision → action → changed environment.” Traditional AI models can be embedded in embodied systems; these are not exclusive categories.

## 1.6 Components of an embodied intelligence system

From an engineering perspective, an embodied system commonly includes:

- **Task and goal representation:** objectives, success and failure criteria, and constraints.
- **Perception and state estimation:** converting images, joint measurements and forces into information suitable for decisions.
- **Decision-making and planning:** selecting policies, skills, paths or next actions from goals and current information.
- **Control and execution:** converting action targets into commands that motors, joints or grippers can execute.
- **Body, sensors and actuators:** the physical means of motion, contact, measurement and interaction.
- **Feedback and evaluation:** assessing outcomes and task progress to inform subsequent decisions or learning.
- **Learning:** updating perception or behavior from data when needed. Online learning is optional, not a requirement of every system.

These are functional relationships, not a requirement for separate software modules. An end-to-end model can combine perception and action generation; a classical system may connect several specialized modules. Either way, it must close the loop from information to action and from action results back to information.

Embodied intelligence arises from the combined effects of algorithms, body and environment. For arms, joint arrangement, tool–object contact and reachability directly affect perception, planning and control. The next chapter introduces the structure, degrees of freedom and motion descriptions needed to understand simulation and learning.

## Further reading

1. Liu, H., Guo, D., & Cangelosi, A. (2025). *Embodied Intelligence: A Synergy of Morphology, Action, Perception and Learning*. ACM Computing Surveys, 57(7), Article 186. [Paper](https://doi.org/10.1145/3717059)
2. Cangelosi, A., Bongard, J., Fischer, M. H., & Nolfi, S. (2015). *Embodied Intelligence*. In *Springer Handbook of Computational Intelligence*. [Publication](https://research.manchester.ac.uk/en/publications/embodied-intelligence/)
3. Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think: A New View of Intelligence*. MIT Press. [Publisher](https://mitpress.mit.edu/9780262162395/how-the-body-shapes-the-way-we-think/)
4. Lynch, K. M., & Park, F. C. *Modern Robotics: Mechanics, Planning, and Control*. [11.1 Control System Overview](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)
5. OpenAI. *Spinning Up in Deep RL: Key Concepts in RL*. [Documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
6. Hugging Face. *Robotics Course: Introduction to Robot Learning; Classical Robotics*. [Course](https://huggingface.co/learn/robotics-course/unit1/1)
7. Ahn, M., et al. (2022). *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*. [Project and paper](https://say-can.github.io/)

---

# Chapter 2 Understanding Robot Arms

A robot arm is a common embodiment. Its geometry, joint ranges, tools and sensor placement determine reachability, available information and feasible operations, not merely how algorithm outputs are executed. Understanding arm structure and motion is a foundation for simulation, policy training and deployment.

## 2.1 Structure and basic components

Industrial and educational arms usually form a serial chain: joints connect successive links from a fixed base to an end effector. The base supplies the mounting reference, links form the body, joints permit relative motion and the end effector interacts directly with objects.

:::learning-visual arm-anatomy

Joints near the base often make larger positional adjustments, while wrist joints mainly change tool orientation. This is an intuition, not an absolute division: rotating an upstream joint moves every downstream link, so end-effector position results from several joints acting together.

The connected components form a kinematic chain. The base is usually link 0, followed by joint variables in chain order. A **tool center point (TCP)** defines the tool reference used in planning and control; “end-effector position” usually means the TCP, not an arbitrary point on the gripper casing. When tool length changes, its TCP offset must be updated to avoid incorrect contact positions.

Actuators drive joints. Motors, gearboxes and transmissions convert electrical commands into rotation or translation; encoders measure position. Some arms also have torque, six-axis force/torque or tactile sensors. Controllers read these measurements and compute new commands. The body therefore combines mechanics, sensing, actuation and control.

The end effector determines how the arm acts on its environment. Two-finger grippers suit objects with defined graspable dimensions; suction cups require suitable surfaces. Dexterous hands offer more complex in-hand manipulation but require more sensing and control. Changing a tool often changes available skills, the action space and training data.

Camera placement is also part of body design. A fixed camera gives a relatively stable global view but can be occluded. A wrist camera provides close-up views while introducing viewpoint changes caused by arm motion. Placement changes observations and consequently the design of perception and policy models.

Payload, repeatability and stiffness matter too. Payload includes tools, cameras and sensors as well as the object: heavier tooling leaves less capacity for the object. Repeatability measures how closely repeated commands return to the same position; it is distinct from absolute positioning accuracy. These properties constrain speed, trajectories and suitable objects.

Longer links increase reach but can increase inertia and endpoint errors. More complex hands add manipulation options and control variables. Morphology both constrains behavior and can simplify tasks. Pfeifer and Bongard emphasize that the body supports and constrains perception and behavior rather than merely enclosing intelligence.[8]

## 2.2 Joint types, degrees of freedom and constraints

A **revolute joint** rotates adjacent links around an axis, with an angular variable in radians or degrees. A **prismatic joint** translates along an axis, usually measured in meters. Each permits one independent motion and contributes one degree of freedom.

| Joint type | Joint variable | Typical units | Relative motion | Typical role |
|---|---|---|---|---|
| Revolute | Angle \(q_i\) | rad or ° | Rotation around the joint axis | Change link direction and tool orientation |
| Prismatic | Displacement \(q_i\) | m or mm | Translation along the joint axis | Change extension or linear position |

**Degrees of freedom (DOF)** count independently variable motion parameters. An arm with six independent revolute joints is commonly called a six-DOF arm. An unconstrained rigid-body pose in 3D has three positional and three rotational degrees of freedom, motivating six-DOF control of both tool position and orientation.

DOF count alone does not guarantee every orientation at every position. Axis arrangement, link lengths, interference and singularities also constrain motion. A robot with more DOF than a task requires may be redundant: it can change elbow position while maintaining the endpoint target, avoiding obstacles or joint limits.

For a position-only task, three coordinates specify the target, and a seven-DOF arm commonly has multiple configurations that reach it. Extra DOF can reduce motion, increase obstacle clearance or prepare for the next operation. Specifying a strict position and orientation leaves less freedom.

Real joints have position, velocity, acceleration and torque limits. Position limits define travel; speed and acceleration limits affect duration and smoothness; torque limits constrain supported and generated loads. Planning and control must satisfy all of them for mathematically feasible goals to be executable safely.

These limits interact. Extending an arm increases upstream torque demand, so a geometrically reachable target near the workspace boundary may be unsuitable for high speeds or heavy objects.

The same constraints affect learning. Joint count sets the dimension of joint-space actions; joint ranges set valid components. Different joint counts, orders or limits can make a transferred policy's outputs physically meaningless without adaptation.

## 2.3 Coordinate frames, position and orientation

Motion descriptions must identify their reference frame. An object can be described in world, base, camera or tool coordinates. Numbers alone are incomplete without that frame.

Common frames include:

- A **world frame**, the shared reference for the scene.
- A **base frame**, attached to the arm's mounting base.
- An **end-effector frame**, attached to the gripper or TCP and moving with it.
- An **object frame**, used to define grasp points and desired object poses.

Position is usually a three-dimensional vector \(\mathbf{p}=[x,y,z]^T\) in meters. Orientation may be represented by a rotation matrix, quaternion or Euler angles. Reaching above a box does not guarantee a graspable orientation: the gripper opening must also align with its contact surfaces.

Euler angles are intuitive but depend on rotation order and can have representation singularities. Rotation matrices express axis directions directly but use nine elements with orthogonality constraints. Quaternions represent rotations compactly with four numbers and are common in robotics software. Initially, remember that general orientation requires more than a single angle and that representations must follow consistent conventions.

A coordinate transform connects representations of the same geometry. A homogeneous transform combines rotation and translation:

\[
{}^{A}T_{B}=
\begin{bmatrix}
{}^{A}R_{B} & {}^{A}\mathbf{p}_{B}\\
\mathbf{0}^{T} & 1
\end{bmatrix}.
\]

The indices denote frame \(B\) relative to frame \(A\). The dimensionless \({}^{A}R_B\) is a \(3\times3\) rotation matrix; \({}^{A}\mathbf{p}_B\) is a translation in meters. A point expressed as \({}^{B}\mathbf p\) becomes

\[
{}^{A}\mathbf p={}^{A}R_B\,{}^{B}\mathbf p+{}^{A}\mathbf p_B.
\]

First rotate the vector into frame A, then add the offset between frame origins. If the frames have the same orientation, the tool origin is \((0.4,0.1,0.3)\,\mathrm m\) in the base frame, and a point is \((0,0,0.1)\,\mathrm m\) in the tool frame, its base coordinates are \((0.4,0.1,0.4)\,\mathrm m\).

A 2D example shows the same order. Let B rotate 90° counterclockwise relative to A, with its origin at \((0.4,0.2)\) m in A. The point is \((0.1,0)\) m in B:

```python
import math

def transform_point_2d(point_b, theta_ab, origin_b_in_a):
    x_b, y_b = point_b
    c, s = math.cos(theta_ab), math.sin(theta_ab)

    # R_ab @ p_b: rotate the vector into frame A first
    x_rot = c * x_b - s * y_b
    y_rot = s * x_b + c * y_b

    # Then add the position of frame B's origin in A
    return (
        x_rot + origin_b_in_a[0],
        y_rot + origin_b_in_a[1],
    )

point_a = transform_point_2d(
    point_b=(0.1, 0.0),
    theta_ab=math.radians(90.0),
    origin_b_in_a=(0.4, 0.2),
)
print(tuple(round(v, 3) for v in point_a))  # (0.4, 0.3) m
```

B's positive X axis points along A's positive Y axis. Rotation gives \((0,0.1)\) m; adding the origin offset yields \((0.4,0.3)\) m. A 3D implementation uses a rotation matrix or quaternion but follows the same relation.

Homogeneous transforms let us compose chains by matrix multiplication. ROS tf2 maintains time-varying frame relationships in a transform tree. ROS [REP-0103](https://reps.openrobotics.org/rep-0103/) specifies conventions including right-handed frames and SI units such as meters and radians.[4][5]

Order matters: combine object-to-camera with camera-to-base transforms to obtain object-to-base. Reversing matrix order changes the frame relationship; transforms do not commute like scalar factors.

Frames connect perception to action. A fixed camera may estimate an object in camera coordinates while the planner expects base coordinates. Calibration errors can turn correct recognition into incorrect motion. Coordinate transforms are therefore essential to linking perception, policy outputs and execution.

## 2.4 Joint space and task space

Arms admit two complementary descriptions. **Joint space** directly represents joint variables. For \(n\) joints:

\[
\mathbf{q}=[q_1,q_2,\ldots,q_n]^T.
\]

Revolute components are usually in radians; mixed chains may contain both radians and meters. A point in joint space describes one configuration, and joint limits define its allowed region.

Motion traces a curve in joint space. Feasibility depends on limits and collisions of the entire arm, not just individual joint values. Configuration space converts complex link collisions into the question of whether a configuration enters a collision region.

**Task space** describes quantities of interest: tool position, orientation, object position or contact force. End-effector pose \(\mathbf{x}\) relates to configuration by

\[
\mathbf{x}=f(\mathbf{q}),
\]

where \(f\) is determined by arm geometry. Joint angles describe internal motion; tool pose describes its external effect.

“Move joint 1 to 30°” is a joint-space target. “Move the gripper 10 cm above the object” is a task-space target. Systems often specify intent in task space and convert it to joint commands.

Policies can likewise output joint-angle increments or tool displacements converted through inverse kinematics. Joint actions are closer to actuators but robot-specific. Tool-space actions express task intent more directly but require dependable kinematics and constraints. Sharing task-space intent does not eliminate differences in workspace, speed, payload or gripper design.

Observations involve similar choices. Joint angles describe internal configuration without explicitly expressing target relations. Tool position and relative target coordinates make these relations explicit but depend on correct kinematics and frame transforms. Observation and action design must therefore account for the body.

## 2.5 Forward and inverse kinematics

**Forward kinematics (FK)** computes endpoint position and orientation from joint configuration. Every upstream joint changes downstream links, so we compose the chain. Modern Robotics describes this as a mapping from joint variables to end-effector frame configuration.[1][2]

For a planar two-link arm, let lengths \(l_1,l_2\) be in meters, \(q_1\) the first angle, \(q_2\) the second angle relative to link 1, and \((x,y)\) the endpoint coordinates in the base frame:

\[
x=l_1\cos q_1+l_2\cos(q_1+q_2),
\]

\[
y=l_1\sin q_1+l_2\sin(q_1+q_2).
\]

:::learning-visual planar-kinematics

Link 1's direction is \(q_1\); link 2's absolute direction is \(q_1+q_2\). The two joints jointly determine both coordinates; each joint does not control just one axis.

For \(l_1=0.40\,\mathrm m\), \(l_2=0.30\,\mathrm m\), \(q_1=30^\circ\) and \(q_2=60^\circ\), link 2 points at 90°:

\[
x=0.40\cos30^\circ+0.30\cos90^\circ\approx0.346\,\mathrm{m},
\]

\[
y=0.40\sin30^\circ+0.30\sin90^\circ=0.500\,\mathrm{m}.
\]

The endpoint is approximately \((0.346,0.500)\,\mathrm m\). FK is usually directly computable from known angles and geometry. The Hugging Face Robotics Course uses a similar planar arm to explain this mapping.[3]

The following Python function implements the same calculation. Trigonometric functions take radians, so convert degree inputs with `math.radians()`. Outputs remain in meters.

```python
import math

def forward_kinematics_2r(l1, l2, q1, q2):
    """Planar two-link FK; lengths in m and angles in rad."""
    x = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    y = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
    return x, y

q1 = math.radians(30.0)
q2 = math.radians(60.0)
x, y = forward_kinematics_2r(0.40, 0.30, q1, q2)
print(f"x={x:.3f} m, y={y:.3f} m")
```

The result, `x=0.346 m, y=0.500 m`, matches the hand calculation. Changing either angle affects both coordinates. This ideal planar calculation omits limits, collisions, orientation and numerical robustness; it is not a replacement for a full kinematics library.

**Inverse kinematics (IK)** finds a joint configuration for a desired endpoint position or pose. It is not simply “reversing” FK: a target may have multiple solutions or no feasible solution.

For an ideal two-link arm, let \(r=\sqrt{x^2+y^2}\). Ignoring joint limits, the geometric reachability condition is

\[
|l_1-l_2|\le r\le l_1+l_2
\]

An interior target commonly has elbow-up and elbow-down solutions. Targets beyond the outer reach or inside the inner hole have none. Joint limits, obstacles and orientation requirements can invalidate an otherwise reachable point.

The cosine rule gives

\[
\cos q_2=\frac{x^2+y^2-l_1^2-l_2^2}{2l_1l_2}.
\]

If the right-hand side has magnitude greater than 1, no real solution exists. Otherwise, we usually take

\[
q_2=\pm\arccos\!\left(\frac{x^2+y^2-l_1^2-l_2^2}{2l_1l_2}\right),
\]

and obtain the first angle from

\[
q_1=\operatorname{atan2}(y,x)-
\operatorname{atan2}\!\left(l_2\sin q_2,\,l_1+l_2\cos q_2\right)
\]

For \((0.346,0.500)\,\mathrm m\), approximate solutions are \((30.0^\circ,60.0^\circ)\) and \((80.6^\circ,-60.0^\circ)\). They reach approximately the same point with different elbow positions. Selection must consider limits, collisions, travel distance and stability.

IK also encounters singular configurations. For the two-link positional Jacobian:

\[
\det J(\mathbf q)=l_1l_2\sin q_2.
\]

When \(q_2=0\) or \(q_2=\pi\), the arm is straight or folded, the determinant vanishes and the Jacobian loses rank. Some endpoint motions may require extremely large joint velocities or be unavailable instantaneously. Modern Robotics describes singularities through this loss of independent motion directions.[1]

Complex arms often use iterative numerical IK rather than a simple closed form. Different initial configurations can converge to different solutions or fail because of limits, singularities or local conditions. Practical solvers therefore specify time limits, tolerances and solution-ranking criteria.

MoveIt provides kinematics plugins and combines solutions with joint limits and collision checks.[6] Reaching a target means satisfying whole-body and environment constraints, not merely solving coordinate equations.

## 2.6 Workspace, trajectories and motion control

The **workspace** is the set of reachable positions or poses. Requiring a particular orientation reduces it: reaching a position does not imply reaching it with any orientation.

A workspace is not a fixed sphere. Tool length shifts the TCP, mounting orientation changes reach relative to the scene, and a grasped object needs collision clearance. Table height and object placement should be designed together with the workspace.

Link lengths set its scale; limits, interference and obstacles reduce usable regions. Even a nearby target may be infeasible because of joint limits, required orientation or a path crossing the table. Workspace is a direct bodily constraint on possible actions.

A **path** is a geometric route. A **trajectory** assigns times along that route and consequently specifies velocity and acceleration. Abrupt target changes can cause shocks, vibration or actuator overload, so practical systems use smooth time parameterization.

A simple joint-space trajectory is

\[
\mathbf{q}(t)=\mathbf{q}_0+\alpha(t)(\mathbf{q}_f-\mathbf{q}_0),
\]

where \(\mathbf q_0\) and \(\mathbf q_f\) are the start and finish, time is in seconds and dimensionless \(\alpha\) moves smoothly from 0 to 1. For duration \(T_f\), set \(\tau=t/T_f\) and use

\[
\alpha(\tau)=3\tau^2-2\tau^3,\qquad 0\le\tau\le1.
\]

Its derivative is zero at both ends, avoiding an instantaneous nonzero starting or stopping velocity. A joint moving from 20° to 60° in two seconds reaches 40° at one second, where \(\tau=0.5\) and \(\alpha=0.5\).

After planning, controllers track the trajectory using feedback. Load, friction, model errors and contact prevent a one-time target command from guaranteeing accuracy. MoveIt time-parameterizes paths under velocity and acceleration limits; its default pipeline checks self-collision and environmental collision.[7]

Control interfaces also have levels. High-level pose or joint targets become lower-level velocity, current or torque commands. Position control suits free-space motion; assembly, wiping and grasping also require force and compliance considerations. Motor-level policy outputs offer more freedom but increase training and safety demands. A mature low-level controller often reduces the policy's burden.

Structure determines possible motion; kinematics maps joints to endpoints; planning chooses feasible paths and trajectories; control makes the body follow them. Simulation turns these relationships into state, observation and action boundaries.

## Research outlook: using the body to solve problems

### Dexterous hands, touch and multimodal manipulation

Dexterous hands must coordinate fingers, changing contact locations and in-hand object rotation, beyond stable grasping. External cameras cannot fully observe occluded contacts. Touch contributes force, slip and local deformation information. A 2024 Nature Communications study combined vision with multimodal tactile sensing for robotic housekeeping, illustrating this research direction.[9]

More finger joints increase DOF, touch changes observations and tooling changes available skills. A richer body expands both capability and the sensing and control problem.

### Co-designing morphology and control

Instead of fixing mechanics before designing control, embodiment research considers joint optimization. Link proportions, axis layout, compliance and sensor placement change control difficulty. Suitable morphology can passively adapt to objects, reducing the precision a policy must compute.

Research treats morphology–controller co-optimization as a key design problem and notes that a changed body often requires an adapted controller.[10][11] Algorithm performance therefore depends on more than network architecture and training; the body is part of the learning problem.

## References

1. Lynch, K. M., & Park, F. C. *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press. [Textbook and resources](https://modernrobotics.northwestern.edu/)
2. Lynch, K. M., & Park, F. C. *Product of Exponentials Formula in the Space Frame*. [Modern Robotics course](https://modernrobotics.northwestern.edu/nu-gm-book-resource/4-1-1-product-of-exponentials-formula-in-the-space-frame/)
3. Hugging Face. *Robotics Course — Understanding Robot Kinematics*. [Course](https://huggingface.co/learn/robotics-course/unit2/3)
4. Open Robotics. *REP-0103: Standard Units of Measure and Coordinate Conventions*. [Specification](https://reps.openrobotics.org/rep-0103/)
5. ROS 2 Documentation. *Introduction to tf2*. [Documentation](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html)
6. MoveIt 2 Documentation. *Kinematics*. [Documentation](https://moveit.picknik.ai/main/doc/concepts/kinematics.html)
7. MoveIt 2 Documentation. *Motion Planning*. [Documentation](https://moveit.picknik.ai/main/doc/concepts/motion_planning.html)
8. Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think: A New View of Intelligence*. MIT Press. [Publisher](https://mitpress.mit.edu/9780262162395/how-the-body-shapes-the-way-we-think/)
9. Mao, Q., et al. (2024). *Multimodal tactile sensing fused with vision for dexterous robotic housekeeping*. Nature Communications, 15, 6871. [Paper](https://www.nature.com/articles/s41467-024-51261-5)
10. Liu, H., Guo, D., & Cangelosi, A. (2025). *Embodied Intelligence: A Synergy of Morphology, Action, Perception and Learning*. ACM Computing Surveys, 57(7), Article 186. [Paper](https://doi.org/10.1145/3717059)
11. Cheney, N., Bongard, J., SunSpiral, V., & Lipson, H. (2018). *Scalable co-optimization of morphology and control in embodied machines*. Journal of the Royal Society Interface, 15(143), 20170937. [Paper](https://doi.org/10.1098/rsif.2017.0937)

---

# Chapter 3 Simulation and the Perception–Decision–Action Loop

An environment is neither a background image nor an animation. It computes what happens next from robot geometry, object properties and commands, then returns information available to sensors. Understanding this process clarifies what simulated success establishes and why it may not transfer directly to hardware.

## 3.1 What robot simulation does—and its limits

Simulation models the robot, task scene and their interaction. An arm environment typically includes links, joints, actuators, sensors, geometry, mass, inertia, gravity, collision relationships, contact and friction, plus episode initialization and termination logic.

The simulator receives an action, computes forces, contact and motion, then updates measurements such as camera images, encoder readings and forces. Rendering visualizes internal state; physics and interaction can continue without a visible window. Computing state evolution is the physics engine's central purpose.

Simulation offers repeatable initial conditions, varied object layouts, large volumes of interaction and testing without damaging equipment. Recorded seeds and parameters help reproduce conditions, diagnose failures and compare methods.

Greater complexity does not automatically mean greater realism. Finite geometry and numerical models approximate deformation, backlash, cable forces, exposure and wear—or omit them. Contact is particularly difficult: slip depends on shape, friction, stiffness, grip force and solver settings. Engines such as MuJoCo model contact constraints and tangential, torsional or rolling friction, but these remain computational approximations.[1][2]

Required fidelity depends on the question. Reachability emphasizes geometry and limits; fast trajectories require accurate mass, inertia and actuation; stable grasping depends strongly on shape, friction and compliance. Model details should serve a defined question.

| Dimension | Simulation | Hardware | Possible effect |
|---|---|---|---|
| Geometry and mass | Model files and chosen parameters | Manufacturing, installation and measurement errors | Different positions, inertia and collision boundaries |
| Contact and friction | Contact models and coefficients | Materials, surfaces, temperature and wear | Different pushing distances and grasp stability |
| Sensors | Ideal measurements or artificial noise | Noise, occlusion, drift and calibration errors | Different observations of the same scene |
| Actuators | Prescribed dynamics and limits | Saturation, hysteresis, backlash and communication delay | Different motion for the same command |
| Time | Numerical steps and scheduling | Hardware cycles and system load | Different control rates and feedback delays |

Simulated success establishes performance only under the current model, parameters and initial conditions. It does not independently establish hardware reliability. These model-induced differences form the **reality gap**. Record them when interpreting results; later chapters discuss system identification, randomization and calibration.[8]

## 3.2 Physics, sensor and actuator models

Kinematics explains where an endpoint is, not why it accelerates or changes motion on contact. Dynamics is needed to advance state. A common open-chain arm equation is

\[
M(\mathbf q)\ddot{\mathbf q}
+\mathbf c(\mathbf q,\dot{\mathbf q})
+\mathbf g(\mathbf q)
+\boldsymbol\tau_f
=\boldsymbol\tau+J^T(\mathbf q)\mathbf F_{\mathrm{ext}}.
\]

It computes acceleration from configuration, velocity, actuation and external forces. Joint positions \(\mathbf q\), velocities \(\dot{\mathbf q}\) and accelerations \(\ddot{\mathbf q}\) usually use rad, rad/s and rad/s² for revolute joints. \(M\) is the mass matrix; \(\mathbf c\) contains velocity-dependent Coriolis and centrifugal terms; \(\mathbf g\) represents gravity; \(\boldsymbol\tau_f\) resistance such as friction; and \(\boldsymbol\tau\) actuator torque in N·m. External contact wrench \(\mathbf F_{\mathrm{ext}}\) maps to generalized joint forces through \(J^T\). Prismatic generalized forces use newtons.

The left side describes inertia, velocity coupling, gravity and resistance; the right describes actuation and contact. Solving for acceleration and integrating updates velocity and position. Modern Robotics calls this **forward dynamics**.[3]

For an illustrative single revolute joint, omit gravity, friction and external loads:

\[
I\ddot q=\tau.
\]

With equivalent rotational inertia \(I=0.20\,\mathrm{kg\cdot m^2}\) and net torque \(0.40\,\mathrm{N\cdot m}\):

\[
\ddot q=\frac{\tau}{I}=2.0\,\mathrm{rad/s^2}.
\]

This shows that greater inertia gives less acceleration for the same torque. Real arms cannot generally ignore gravity, friction, transmission or limits.

Collision detection finds approaching or intersecting geometry. Contact solving computes constraint forces that resist penetration and friction that resists slip. Soft contact may visibly sink; excessively stiff contact with large time steps can be unstable. High friction can prevent sliding in Push; low friction can make a closed gripper lose an object. Behavior results from these parameters together.

Sensor models turn state into measurements: encoders provide angles and velocities, cameras produce color or depth, and force sensors report forces and moments. Real measurements have finite resolution, noise, bias, occlusion and delay. Wrist motion changes the image of even a stationary object.

Actuator models convert position, velocity, torque or gripper commands into bodily effects. Motors, gearing, controller bandwidth, saturation and latency mean commands are not necessarily realized exactly. Changed mass, damping or gripper material can change trajectories and contact outcomes, illustrating dependence on embodiment.

## 3.3 State, observation and action

**State**, **observation** and **action** are distinct.

State \(s_t\) describes the environment internally: joint positions and velocities, object poses and velocities, contacts and actuator state. A simulator usually stores more information for physics than it exposes to the policy.

Observation \(o_t\) is the information supplied to the agent: angles, endpoint position, images, relative goals or force readings. Occlusion, unmeasured contact and velocities inferred from image sequences make it only a partial view of state.

Action \(a_t\) is the input submitted to the environment: joint targets, velocities, torques, endpoint displacements or gripper commands. Tool-space targets delegate some work to kinematics and low-level control. Torque actions expose more dynamics but increase learning difficulty and safety demands.

| Task | Possible internal state | Possible observations | Example actions |
|---|---|---|---|
| Reach | Joint state, target and dynamics variables | Angles and relative goal position | Joint velocity or tool displacement |
| Push | Arm/object poses, velocities, contact and friction | Images, joint angles and endpoint position | Planar endpoint displacement or velocity |
| Grasp | Object pose, fingers, contact locations and forces | Images, gripper opening and limited force information | Tool motion and gripper opening/closure |

These are examples, not mandatory interfaces. State supports physics and task checks; observations define available decision information; actions define how the agent influences the system.

A simplified Reach structure makes the distinction explicit. `ReachState` contains internal information; `make_observation()` selects visible fields; `proportional_policy()` converts them into a normalized 2D action:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ReachState:
    joint_q: tuple          # Six joint angles, rad
    joint_dq: tuple         # Six joint velocities, rad/s
    ee_xy: tuple            # Endpoint XY position, m
    ee_velocity_xy: tuple   # Endpoint XY velocity, m/s
    goal_xy: tuple          # Goal XY position, m
    contact_force: tuple    # Internal contact force, N; not observed by this policy

def make_observation(state, previous_action):
    return {
        "arm_q": state.joint_q,
        "arm_dq": state.joint_dq,
        "ee_to_goal_xy": tuple(
            goal - ee for goal, ee in zip(state.goal_xy, state.ee_xy)
        ),
        "ee_velocity_xy": state.ee_velocity_xy,
        "previous_action_xy": previous_action,
    }

def proportional_policy(observation, gain=20.0):
    # Teaching policy: relative position times gain, bounded to [-1, 1]
    return tuple(
        max(-1.0, min(1.0, gain * error))
        for error in observation["ee_to_goal_xy"]
    )

state = ReachState(
    joint_q=(0.0,) * 6,
    joint_dq=(0.0,) * 6,
    ee_xy=(0.42, 0.01),
    ee_velocity_xy=(0.0, 0.0),
    goal_xy=(0.45, 0.00),
    contact_force=(0.0, 0.0, 0.0),
)
observation = make_observation(state, previous_action=(0.0, 0.0))
action = proportional_policy(observation)

print(tuple(round(v, 3) for v in observation["ee_to_goal_xy"]))
print(tuple(round(v, 3) for v in action))
# Output: (0.03, -0.01) m and normalized action (0.6, -0.2)
```

`contact_force` is not observed, so the policy cannot use it directly. Absolute tool and target positions become a relative observation. The action is a command—not the next state: scaling, control and physics must still produce a new `ReachState`. This proportional example is not the S1 PPO policy and does not implement real dynamics.

Gymnasium CartPole has a four-component observation—cart position/velocity and pole angle/angular velocity—and two discrete force actions. S1 Reach instead uses continuous 2D endpoint actions. Both support MDP interaction, but their different spaces imply different network inputs, outputs and learning problems.[9]

Do not assume the policy can see every simulator variable. Training with exact object velocities, contact forces or noise-free poses unavailable on hardware can create a policy whose inputs cannot be reproduced in deployment.

History can help partial observability: one image may not reveal motion direction, while multiple frames or previous encoder readings reveal change. Reward is another scalar returned by the interface; designing it is a separate subject. State describes the environment, observation is available information, and action is the agent's input.

## 3.4 Open-loop control, feedback and tracking error

**Open-loop control** executes prescribed commands without using measured outcomes to correct later actions. Running a motor at a chosen speed for a fixed duration may work with an exact model and no disturbances, but lacks a correction mechanism for changed load, friction or contact.

**Closed-loop control** feeds measurements back to the controller. For target \(x^\ast\) and current value \(x_t\):

\[
e_t=x^\ast-x_t.
\]

The error answers “How far are we from the target?” Its units match the controlled quantity—radians for angles or meters for position—and its sign indicates correction direction.

A proportional controller uses

\[
a_t=K_p e_t.
\]

If action is angular velocity in rad/s and error is in rad, \(K_p\) has units s⁻¹. For target 1.0 rad, current angle 0.7 rad and gain 2.0 s⁻¹:

\[
e_t=0.3\,\mathrm{rad},\qquad
a_t=0.6\,\mathrm{rad/s}.
\]

Larger errors yield larger commands, which diminish near the target. This is a basic feedback-control example in Modern Robotics.[4]

Assume an ideal actuator realizes a velocity command and discrete integration uses \(\Delta t\):

\[
x_{t+1}=x_t+\Delta t\,a_t.
\]

Substitution yields

\[
e_{t+1}=(1-K_p\Delta t)e_t.
\]

For \(0<K_p\Delta t<1\), error decreases monotonically in this ideal model. Between 1 and 2, its sign alternates while shrinking. Excessive gain can prevent convergence. Real inertia, delay, saturation and flexibility make overshoot, oscillation and large commands more likely; smaller gains tend to be smoother but slower.[4][5]

The following teaching model fixes a 1 m target, zero initial position and 0.1 s time step, changing only the gain:

```python
def proportional_response(kp, dt=0.1, steps=12, target=1.0):
    x = 0.0
    trajectory = [x]
    for _ in range(steps):
        error = target - x
        action = kp * error
        x = x + dt * action
        trajectory.append(x)
    return trajectory

for kp in (0.5, 5.0, 15.0, 25.0):
    xs = proportional_response(kp)
    first_steps = [round(x, 3) for x in xs[:6]]
    print(f"Kp={kp:>4}: first 5 steps={first_steps}, final={xs[-1]:.3f}")
```

Gains 0.5 and 5 give slow and faster monotonic convergence. Gain 15 gives an error multiplier of −0.5, producing diminishing oscillation. Gain 25 gives −1.5 and divergence. These formula-generated teaching curves omit inertia, speed limits, noise and collision; they are not real S1 Run evidence.

Feedforward and feedback can be combined: one provides nominal motion, the other corrects model errors. Feedback is not reinforcement learning. A proportional controller follows a specified rule without learning; RL improves a decision rule from interaction. They may coexist without being the same concept.

## 3.5 Time steps, control frequency, noise and delay

Physical motion is continuous, while simulation and digital control update discretely. For step duration \(\Delta t\):

\[
s_{t+1}=f(s_t,a_t,\Delta t).
\]

Here \(f\) includes physics and numerical integration; \(s_t,s_{t+1}\) are the before/after states and \(a_t\) the action. If actions update once per step, frequency is

\[
f_c=\frac{1}{\Delta t},
\]

measured in hertz. A 0.02 s period gives 50 Hz.

Physics integration and policy action periods can differ. A simulator may use several small substeps for contact before receiving another action. Smaller steps often improve accuracy at higher computational cost. Large steps may miss contacts or cause numerical oscillation: the step size changes discrete dynamics, not just execution speed.

A simplified observation model is

\[
o_t=h(s_t)+\varepsilon_t.
\]

where \(h\) maps state to a measurement and \(\varepsilon_t\) represents measurement error. Units follow the signal: radians for encoders or pixel values for images. This expression does not assume a particular noise distribution. Occlusion, fixed bias and quantization are not always well represented by independent additive noise.

With action delay \(d\) steps:

\[
s_{t+1}=f(s_t,a_{t-d},\Delta t).
\]

Delay duration is approximately \(d\Delta t\). At a 0.02 s period and three-step delay, correction takes 0.06 s. An endpoint moving at 0.25 m/s can travel another

\[
0.25\times0.06=0.015\,\mathrm m,
\]

or 15 mm before the new command acts—significant for precision contact.

Noise makes corrections fluctuate; delay makes them use outdated information; the time step determines the update rhythm. The same controller or policy can have very different stability under different rates, noise and delays. Training only with ideal sensing and zero delay may encourage dependence on unrealistically precise information.

## 3.6 Environment interfaces and episode lifecycle

Physics engines calculate motion; environment interfaces organize repeatable interaction. Gymnasium provides `reset()` and `step(action)`.[6][7]

`reset()` starts an episode, initializes arm and objects, clears prior counters and temporary data, and returns:

```python
observation, info = env.reset(seed=0)
```

`observation` is the initial input; `info` contains auxiliary diagnostics. `seed` controls randomization, but matching a seed alone does not guarantee reproduction without matching implementation, parameters and software versions.

After choosing an action, call:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

`step` advances interaction and returns the new observation, scalar reward and diagnostic `info`.

`terminated` indicates a task-defined terminal condition, such as success or a specified failure. `truncated` indicates an external or imposed boundary, commonly a time or step limit. Gymnasium separates them instead of the old single `done` flag.[6]

A Reach episode may randomize a goal, expose joint angles and relative target position, advance continuous actions, and check error. It ends on success, specified failure or truncation. Call `reset()` before a new episode rather than continuing ordinary steps after termination.

```python
observation, info = env.reset(seed=0)

while True:
    action = policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

This illustrates interaction only: it neither specifies how the policy is obtained nor starts training. Environment interfaces connect robot simulation to subsequent policy learning while keeping initialization, action execution and episode boundaries explicit.

## References

1. MuJoCo Documentation. *Computation*. [Documentation](https://mujoco.readthedocs.io/en/latest/computation/index.html)
2. MuJoCo Documentation. *Modeling*. [Documentation](https://mujoco.readthedocs.io/en/latest/modeling.html)
3. Lynch, K. M., & Park, F. C. *Modern Robotics: Mechanics, Planning, and Control — Chapter 8, Dynamics of Open Chains*. [Course resources](https://modernrobotics.northwestern.edu/chapters/chapter8/)
4. Lynch, K. M., & Park, F. C. *Motion Control with Velocity Inputs*. [Course resources](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-3-motion-control-with-velocity-inputs-part-1-of-3/)
5. Lynch, K. M., & Park, F. C. *Error Response*. [Course resources](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-2-1-error-response/)
6. Farama Foundation. *Gymnasium Env API*. [Documentation](https://gymnasium.farama.org/main/api/env/)
7. Farama Foundation. *Create a Custom Environment*. [Tutorial](https://gymnasium.farama.org/main/introduction/create_custom_env/)
8. Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*. IEEE International Conference on Robotics and Automation. [Paper and Project overview](https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization/)
9. Farama Foundation. *Cart Pole*. [Gymnasium classic control environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

---

# Chapter 4 Neural Network Fundamentals

Neural networks are parameterized functions: numerical inputs pass through layers to produce outputs, while data adjusts the parameters. In embodied systems, they can process images, joint state, touch or language and predict object position, dynamics or actions. This chapter explains function representation, prediction and learning from error.

## 4.1 From an explicit function to a learnable model

A classical program implements a known rule. Two-link forward kinematics provides an example:

\[
x=l_1\cos q_1+l_2\cos(q_1+q_2),
\]

\[
y=l_1\sin q_1+l_2\sin(q_1+q_2).
\]

Known lengths and angles directly determine endpoint position through geometry, without training.

A neural network can instead approximate this mapping from many measured or computed examples:

\[
(q_1,q_2)\longrightarrow(x,y).
\]

Writing the network as \(f_{\boldsymbol\theta}\), with all trainable parameters collected in \(\boldsymbol\theta\):

\[
(\hat x,\hat y)=f_{\boldsymbol\theta}(q_1,q_2).
\]

Hats indicate predictions. Training selects parameters so predicted positions approach their labels.

Each example contains input angles \((q_1,q_2)\), target coordinates \((x,y)\) computed analytically, and the network's current prediction. Learning from known input–label pairs is **supervised learning**.

The designer chooses layer counts, widths and activations—the architecture. Training adjusts weights and biases. It seeks shared parameters that work across examples, rather than a lookup table. Performance on unseen angles depends on representation, optimization and data coverage.

This is a demonstration, not a recommendation to replace accurate analytical FK. Known equations are interpretable and require no training. Learned models become useful when rules are hard to specify, parameters hard to measure, or inputs complex, such as object pose from images.

The network learns numerical relationships supported by data. It does not automatically understand rigid links, limits, collision or gravity merely because its inputs come from a robot.

## 4.2 Neurons, weights, biases and activations

An artificial neuron is a mathematical unit, not a complete model of a biological neuron. For input \(\mathbf x\in\mathbb R^d\), it computes a weighted sum:

\[
z=\mathbf w^T\mathbf x+b,
\]

and applies an activation:

\[
y=\sigma(z).
\]

Weights \(\mathbf w\) determine the direction and strength of input contributions; bias \(b\) shifts the response independently of input. For a one-dimensional neuron, \(wx+b=0\) locates the transition in its response.

Hidden weights and activations generally have no fixed physical interpretation. A large weight does not alone identify an “important joint”: normalization and subsequent layers affect its meaning. Such interpretations need an explicitly interpretable design or additional analysis.

Composing only linear layers still produces one linear transformation. Nonlinear activations allow curved or piecewise relationships such as the sine and cosine dependence in FK. PyTorch's introductory tutorial places nonlinear activations between linear layers for this reason.[1]

| Activation | Expression | Output range | Typical use and behavior |
|---|---|---|---|
| ReLU | \(\max(0,z)\) | \([0,+\infty)\) | Common hidden-layer activation; zero for negative inputs |
| tanh | \(\tanh z\) | \((-1,1)\) | Bounded and centered at zero; gradients diminish at large magnitudes |
| Linear | \(y=z\) | \((-\infty,+\infty)\) | Common regression output; no additional range compression |

The FK example can use ReLU hidden layers and linear outputs in meters. A linear output can predict outside the true workspace; geometric limits are not automatically enforced.

Likewise, `tanh` bounds numbers, not physical safety. Unit conversion, action scaling, joint and velocity limits, and collision checks remain necessary.

## 4.3 Multilayer networks and forward propagation

A multilayer perceptron (MLP) has inputs, hidden layers and outputs. A one-hidden-layer FK model is

\[
\mathbf h=\sigma(W_1\mathbf x+\mathbf b_1),
\]

\[
\hat{\mathbf y}=W_2\mathbf h+\mathbf b_2.
\]

with input

\[
\mathbf x=
\begin{bmatrix}q_1\\q_2\end{bmatrix}
\in\mathbb R^2,
\]

and output

\[
\hat{\mathbf y}=
\begin{bmatrix}\hat x\\\hat y\end{bmatrix}
\in\mathbb R^2.
\]

For hidden width \(m\), \(W_1\) has shape \(m\times2\), \(\mathbf b_1\) length \(m\), and \(\mathbf h\) length \(m\). The output matrix \(W_2\) has shape \(2\times m\), with a two-component bias. Angles become a hidden representation and then a two-coordinate prediction.

**Forward propagation** performs these computations with current parameters; it does not update them. A batch of \(B\) examples has input shape \(B\times2\), hidden shape \(B\times m\), and output shape \(B\times2\).

More layers or width increase representational capacity and computation, without guaranteeing lower training or test error. A small MLP is enough to illustrate this two-input, two-output problem.

Hidden components need not correspond to predefined physical concepts. Their combinations support lower prediction error; the network does not automatically acquire the semantics of rotary joints or fixed link lengths.

Input representation matters. Angles \(-\pi\) and \(\pi\) are numerically far apart but represent the same direction. Feeding sine and cosine of each angle encodes periodicity explicitly, increasing the input dimension from two to four. Raw angles can also work with sufficient coverage.

An accurate position prediction does not establish safe reachability, choose an IK solution or guarantee a collision-free path. Those remain kinematics, planning and control questions.

## 4.4 Loss, gradients and backpropagation

A loss quantifies disagreement between prediction and target. For \(N\) position examples, one choice is mean squared position error:

\[
L(\boldsymbol\theta)=
\frac{1}{N}\sum_{i=1}^{N}
\left\|\hat{\mathbf y}_i-\mathbf y_i\right\|_2^2.
\]

Here \(\boldsymbol\theta\) contains all parameters and \(\|\cdot\|_2\) is the Euclidean norm. Coordinates in meters make this loss have units m². Lower loss means closer predictions on these examples.

For true position \((0.30,0.40)\) m and prediction \((0.28,0.43)\) m:

\[
(-0.02,0.03)\,\mathrm m,
\]

\[
(-0.02)^2+(0.03)^2=0.0013\,\mathrm{m^2}.
\]

The square root is approximately 0.036 m, or 3.6 cm. Prediction loss is not task success rate: grasping also depends on orientation, collision, contact and stability.

The gradient \(\nabla_{\boldsymbol\theta}L\) collects partial derivatives and indicates the direction of greatest local loss increase. Consider the scalar model

\[
\hat y=wx+b,
\qquad
L=(\hat y-y)^2,
\]

The chain rule gives

\[
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial w}
=2(\hat y-y)x.
\]

At \(x=2,y=5,w=1.5,b=0.5\), the prediction is 3.5:

\[
\frac{\partial L}{\partial w}=2(3.5-5)\times2=-6.
\]

A negative gradient means increasing this weight slightly reduces loss locally. With learning rate 0.1:

\[
w_{\mathrm{new}}=w-\alpha\frac{\partial L}{\partial w}
=1.5-0.1\times(-6)=2.1.
\]

Holding other quantities fixed, the prediction becomes 4.7 and loss falls from 2.25 to 0.09. This illustrates direction, not a guarantee that a larger learning rate always improves more.

**Backpropagation** efficiently computes parameter gradients by applying the chain rule backward through the computation graph. The **optimizer** then updates parameters. PyTorch records forward operations and applies this differentiation when `backward()` is called.[2]

## 4.5 Optimization and the training loop

Basic gradient descent is

\[
\boldsymbol\theta_{k+1}
=\boldsymbol\theta_k-alpha
\nabla_{\boldsymbol\theta}L(\boldsymbol\theta_k).
\]

Update index \(k\) counts iterations; positive learning rate \(\alpha\) controls step size. Too small can be slow, suitable values can converge, and too large can oscillate or diverge. Momentum and adaptive optimizers modify updates but still use gradients.[3]

| Term | Meaning |
|---|---|
| Batch | Examples used together for a forward pass, loss and gradient calculation |
| Step | One optimizer parameter update |
| Epoch | Approximately one pass through the training set |

With 2,800 examples and batch size 128, an epoch has about 22 steps. Batches are not permanent dataset partitions; examples are commonly reshuffled each epoch.

Full-batch descent uses the entire set, while one-example updates are highly variable. Mini-batches balance parallel computation with varied samples. Local fluctuations do not alone prove divergence. Learning-rate comparisons should fix the split, initialization and batch settings and inspect validation trends.

Different units and scales affect gradients. Angles in radians, positions in meters and velocities in rad/s may have very different ranges. A common normalization is

\[
x'=\frac{x-\mu}{s},
\]

where \(\mu\) and standard deviation \(s\) come **only from the training set**. The normalized value is dimensionless. Apply those same statistics to validation, test and future inputs; using test statistics leaks information.

The following complete PyTorch example fits analytical two-link FK. Its data are instructional, not platform Run results or measured hardware accuracy.

<details>
<summary>Show the two-link forward-kinematics fitting example</summary>

```python
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# 1. Generate teaching data: angles as inputs, analytical FK positions as labels
n = 4000
l1, l2 = 0.40, 0.30
q = (torch.rand(n, 2) * 2 - 1) * math.pi
q1, q2 = q[:, 0], q[:, 1]
y = torch.stack([
    l1 * torch.cos(q1) + l2 * torch.cos(q1 + q2),
    l1 * torch.sin(q1) + l2 * torch.sin(q1 + q2),
], dim=1)

# 2. Fixed split: 70% training, 15% validation, 15% test
order = torch.randperm(n)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
train_id = order[:n_train]
val_id = order[n_train:n_train + n_val]
test_id = order[n_train + n_val:]

q_train, y_train = q[train_id], y[train_id]
q_val, y_val = q[val_id], y[val_id]
q_test, y_test = q[test_id], y[test_id]

# 3. Standardize inputs using training statistics only
mu = q_train.mean(dim=0, keepdim=True)
std = q_train.std(dim=0, keepdim=True).clamp_min(1e-6)

def normalize(value):
    return (value - mu) / std

loader = DataLoader(
    TensorDataset(normalize(q_train), y_train),
    batch_size=128,
    shuffle=True,
)

# 4. Small two-hidden-layer MLP: 2D angles in, 2D position out
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Train: predict, calculate loss, backpropagate, update
for epoch in range(200):
    model.train()
    for q_batch, y_batch in loader:
        optimizer.zero_grad()
        prediction = model(q_batch)
        loss = ((prediction - y_batch) ** 2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_prediction = model(normalize(q_val))
            val_mse = ((val_prediction - y_val) ** 2).sum(dim=1).mean()
        print(
            f"epoch={epoch + 1:3d}, "
            f"Teaching example validation MSE={val_mse.item():.6e} m^2"
        )

# 6. Use the test set only for final assessment after training and model selection
model.eval()
with torch.no_grad():
    test_prediction = model(normalize(q_test))
    test_mse = ((test_prediction - y_test) ** 2).sum(dim=1).mean()
    test_rmse = torch.sqrt(test_mse)

print(f"Teaching example test RMSE={test_rmse.item():.6f} m")
```

</details>

Data generation defines angles, labels and range; splitting separates parameter updates, model selection and final assessment. The network uses two ReLU hidden layers and linear outputs. Each training iteration clears gradients, predicts, computes squared position error, backpropagates and updates.

`optimizer.zero_grad()` prevents unintended gradient accumulation. `loss.backward()` computes gradients; `optimizer.step()` changes parameters. Evaluation uses `model.eval()` and disables gradient tracking because it needs predictions only.[2][3]

No fixed test error is claimed: results depend on PyTorch, hardware and numerical implementations. Printed validation and test metrics must be labeled instructional, not evidence of platform training or real-arm precision.

## 4.6 Overfitting, generalization and data splits

Training error measures fit to data used for updates. **Generalization** concerns unseen examples; training loss alone cannot establish it.

Training data support gradients, validation data support choices such as architecture and learning rate, and test data assess the final selected model. Repeatedly tuning against test outcomes makes the test set part of model selection.

Suppose training angles lie within

\[
q_1,q_2\in[-90^\circ,90^\circ].
\]

New angle combinations inside that range mainly test interpolation. Testing at 90°–120° outside it tests extrapolation, for which the data provide less support.

Checking coordinate-wise minima and maxima is insufficient. Data concentrated near \(q_1\approx q_2\) leave much of the angle rectangle uncovered. Inspect the joint distribution and its FK-mapped workspace coverage.

Limits, workspace, collision and safety restrict collectable configurations. Central-workspace data may hide large errors near joint limits or singularities. A low overall mean can conceal local failures.

**Overfitting** occurs when training error keeps improving while validation or unseen-data error stops improving or rises. Larger models can fit useful structure and accidental noise. Better coverage, limited capacity, weight decay and validation-based early stopping are common responses.

Analytical FK offers plentiful noise-free labels, making this example simpler than real images, touch or manual annotations. Real deployment may also differ in distribution. Smooth teaching curves do not establish equivalent real-world performance.

In embodied systems, networks may estimate visual scene information, fuse sensors, predict dynamics, choose actions or estimate future value. Always ask what the inputs and outputs are, where data came from and how predictions enter planning, control and interaction.

> **Research outlook: where networks sit in the interaction chain**
>
> - Vision and touch networks convert sensor signals into poses, classes, contact information or learned features.
> - Policy networks choose actions; value networks estimate future outcomes.
> - Learned dynamics predict next states for planning or to complement incomplete physical models.
> - Vision–language–action models connect images and instructions to robot actions. They still depend on action interfaces, data coverage and safety constraints. RT-2 is a representative example.[7]
> - World models learn internal representations of action-conditioned change. Their predictions remain approximate and do not replace environmental feedback.[8]
>
> These names describe roles, not a single architecture. Later chapters examine policy learning, language-model planning and robot data in more detail.

## References

1. PyTorch. *Build the Neural Network*. [Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
2. PyTorch. *Automatic Differentiation with torch.autograd*. [Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
3. PyTorch. *Optimizing Model Parameters*. [Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
4. PyTorch. *Datasets & DataLoaders*. [Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning — Chapter 5: Machine Learning Basics*. MIT Press. [Online textbook](https://www.deeplearningbook.org/contents/ml.html)
6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning — Chapter 6: Deep Feedforward Networks*. MIT Press. [Online textbook](https://www.deeplearningbook.org/contents/mlp.html)
7. Zitkovich, B., et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. Proceedings of the 7th Conference on Robot Learning. [Paper](https://proceedings.mlr.press/v229/zitkovich23a.html)
8. Ha, D., & Schmidhuber, J. (2018). *World Models*. [Paper](https://arxiv.org/abs/1803.10122)

---

# Chapter 5 Reinforcement Learning Fundamentals

Reinforcement learning (RL) studies how agents improve behavior through interaction. Actions change the environment, which returns rewards and observations; the algorithm updates its policy. An arm's data are generated jointly by its behavior, body and environment, rather than always coming from a fixed dataset. RL is structured learning, not unrestricted trial and error.

## 5.1 Reinforcement learning and Markov decision processes

In supervised FK learning, a prediction does not change the next training example. In RL, actions change state and the distribution of future data. A policy that remains in the workspace center rarely learns about its boundaries.

Actions also have long-term effects: moving closer now can leave an awkward configuration, while a temporary detour may avoid an obstacle. Reward does not normally provide a target action as a supervised label would. Delayed rewards create a temporal credit-assignment problem.

Many RL problems are represented as a **Markov decision process (MDP)**:

\[
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
\]

The components are state space \(\mathcal S\), action space \(\mathcal A\), transition law \(P\), reward law \(R\) and discount factor \(\gamma\in[0,1]\). This definition describes sequential decisions without requiring neural networks or a particular algorithm.[1]

Reward may be written \(R(s_t,a_t,s_{t+1})\) or as a conditional distribution. States and actions can be discrete or continuous. Joint angles, velocities and torques commonly make arm problems continuous.

A one-dimensional teaching environment makes the components concrete: three positions, left/right actions, reward 1 at the goal, a −0.05 step cost and discount 0.9.

```python
STATES = (0, 1, 2)          # Left, middle, goal
ACTIONS = (-1, 1)           # Move left, move right
GOAL = 2
GAMMA = 0.9

def transition(state, action):
    if state not in STATES or action not in ACTIONS:
        raise ValueError("Invalid state or action")

    next_state = max(0, min(GOAL, state + action))
    terminated = next_state == GOAL
    reward = 1.0 if terminated else -0.05
    return next_state, reward, terminated

print(transition(state=1, action=1))   # (2, 1.0, True)
print(transition(state=1, action=-1))  # (0, -0.05, False)
```

Here states are \(\{0,1,2\}\), actions \(\{-1,+1\}\), `transition()` implements transitions and reward, and `GAMMA` defines discounting. Action +1 at state 1 deterministically reaches state 2. Introducing a 10% chance of reversed actuation would make transitions stochastic.

This example discretizes away velocity, limits and tracking errors. S1 tasks share the concepts, not this simplified physical model.

For Reach, state may contain joint angles, velocities, target and obstacles. Velocity commands induce a transition distribution

\[
P(s_{t+1}\mid s_t,a_t).
\]

because friction, tracking errors and disturbances may vary outcomes. A deterministic idealization concentrates that distribution at one next state.

The Markov property is

\[
P(s_{t+1}\mid s_0,a_0,\ldots,s_t,a_t)
=P(s_{t+1}\mid s_t,a_t).
\]

It does not mean “there is no history.” Rather, a sufficient current state makes older history unnecessary for predicting the next state. Angles alone can be insufficient: identical configurations moving in opposite directions have different futures.

The policy may receive only observation \(o_t\), rather than complete state \(s_t\). Occlusion, unmeasured forces and unknown friction create partial observability.[2] Multiple frames, action history or state estimates can improve information without revealing complete state. Specify which inputs are actually used during training and evaluation.

Embodiment shapes transitions through mass, limits and gripper structure; sensors shape observations; control interfaces shape actions. Equal reward definitions do not make policies interchangeable between arms.

## 5.2 State, action, reward and policy

These concepts require separate definitions. For one possible Reach design:

| Concept | Example representation | Role and limitations |
|---|---|---|
| State \(s_t\) | Joint state, endpoint, goal and obstacles | Sufficient environment description, not necessarily policy input |
| Observation \(o_t\) | Encoders, estimated goal and previous action | Available information, possibly noisy, delayed or incomplete |
| Action \(a_t\) | Joint velocity, endpoint displacement or torque | Interface-specific dimensions, units and constraints |
| Reward \(r_{t+1}\) | Scalar feedback after a transition | Optimization signal, not task completion itself |
| Success criterion | Enter the target region and remain for a specified time | Independent rule for measuring completion |

Simulator ground truth is not automatically available on hardware. A policy trained on exact poses may need unavailable inputs in deployment.

A six-joint velocity command may be six-dimensional in rad/s; a translation increment three-dimensional in meters. Torque commands expose dynamics more directly. Moving to a lower-level interface transfers responsibility from traditional control to the policy and increases difficulty and safety requirements. Tool displacements delegate some coordination to IK and low-level control. All outputs still require physical constraint checks.

Reward can encode distance progress, smoothness or completion. It is an optimization signal; success is a task decision. A reward omitting collision may give a colliding policy a high score. A longer successful episode may also score below a short failure under some cost definitions. Record independent task metrics alongside reward.

A parameterized stochastic **policy** chooses actions by sampling:

\[
a_t\sim\pi_{\theta}(\cdot\mid o_t).
\]

Observation is \(o_t\), action is \(a_t\), and \(\theta\) contains learned parameters. The symbol \(\sim\) denotes sampling. A deterministic policy instead uses

\[
a_t=\mu_{\theta}(o_t).
\]

and returns the same action for the same observation and parameters. Policy randomness and environment randomness are independent issues.

Continuous policies often use Gaussian means and standard deviations for \(d_a\) action components:

\[
a_t\sim\mathcal N\!\left(
\boldsymbol\mu_{\theta}(o_t),
\operatorname{diag}(\boldsymbol\sigma_{\theta}^2(o_t))
\right).
\]

The mean expresses the preferred action and standard deviation its dispersion. A diagonal covariance assumes conditionally independent components. Gaussian outputs are unbounded, so practical systems transform, scale or clip actions and enforce constraints. A small standard deviation does not guarantee safety.

Action dimension and units follow the interface: six joint velocities give \(d_a=6\), three-dimensional tool translation gives \(d_a=3\). Normalized outputs are dimensionless until mapped to physical scales.

The following standard-library example follows the Gaussian → tanh → physical-scaling sequence used conceptually by S1 continuous policies:

```python
import math
import random

def sample_tanh_gaussian(mean, std, physical_scale=0.005):
    raw_action = [random.gauss(mu, sigma)
                  for mu, sigma in zip(mean, std)]
    normalized_action = [math.tanh(value) for value in raw_action]
    physical_action = [physical_scale * value
                       for value in normalized_action]
    return raw_action, normalized_action, physical_action

random.seed(7)  # Reproduce this example's output only
raw, normalized, physical = sample_tanh_gaussian(
    mean=(0.4, -0.2),
    std=(0.3, 0.3),
)

print("Raw Gaussian sample:", [round(v, 4) for v in raw])
print("Action after tanh:", [round(v, 4) for v in normalized])
print("Physical increment/m:", [round(v, 6) for v in physical])
```

Sampling supports exploration; evaluation may use a deterministic mean action. Tanh bounds normalized values but does not perform workspace, velocity or collision checks. Correct transformed log-probabilities are addressed in Chapter 6.

## 5.3 Trajectories, returns and discounting

Successive states, actions and rewards form a **trajectory**:

\[
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots,s_T).
\]

Action \(a_t\) produces next state \(s_{t+1}\) and reward \(r_{t+1}\). Some texts call the same reward \(r_t\); check indexing conventions.

The finite-episode discounted return from time \(t\) is

\[
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}.
\]

with episode end \(T\) and dimensionless discount \(\gamma\). Return inherits the designed reward scale; it is not automatically a physical quantity.

Consider three teaching rewards:

\[
-0.2,\quad -0.1,\quad 1.0.
\]

Treat the final 1.0 as reaching the goal and preceding negative values as costs:

\[
\begin{aligned}
G_0(\gamma=1.0)&=-0.2-0.1+1.0=0.7,\\
G_0(\gamma=0.9)&=-0.2+0.9(-0.1)+0.9^2(1.0)=0.52,\\
G_0(\gamma=0.5)&=-0.2+0.5(-0.1)+0.5^2(1.0)=0.
\end{aligned}
\]

These are calculations, not Run metrics. Larger discounts preserve more distant reward, but discounting also affects scale, variance and optimization. For bounded continuing rewards, a discount below 1 can keep the infinite sum finite.

Discounting has a physical time scale. Fifty steps equal one second at a 0.02 s period, but five seconds at 0.1 s. Compare discounts together with control rates and episode lengths.

Compute return backward as follows:

```python
def discounted_return(rewards, gamma):
    """Compute the discounted return from the start of the sequence."""
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
    return total


rewards = [-0.2, -0.1, 1.0]  # Illustrative teaching data

for gamma in (1.0, 0.9, 0.5):
    value = discounted_return(rewards, gamma)
    print(f"Teaching example gamma={gamma:.1f}, G_0={value:.3f}")
```

The deterministic output is

```text
Teaching example gamma=1.0, G_0=0.700
Teaching example gamma=0.9, G_0=0.520
Teaching example gamma=0.5, G_0=0.000
```

Single-step reward, discounted return, episode reward sum and success rate answer different questions. Check whether “episode return” in a log is discounted. Reducing action costs can increase return without improving success.

## 5.4 Value and advantage functions

A realized return belongs to one trajectory. A value function is an expectation under policy and environment randomness:

\[
V^{\pi}(s)=\mathbb E_{\pi}\!\left[G_t\mid s_t=s\right].
\]

It estimates return from state \(s\) while following \(\pi\), not a guarantee about one episode. The same state can have different value under skilled and collision-prone policies. Value is not an intrinsic physical property and cannot be compared independently of reward definitions.

The action-value function is

\[
Q^{\pi}(s,a)=\mathbb E_{\pi}\!\left[G_t\mid s_t=s,a_t=a\right].
\]

It fixes the first action before following the policy. State value averages over current action choices; action value evaluates a specified one. Their difference is the **advantage**:

\[
A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s).
\]

Positive advantage means better than the policy's average at that state, not absolutely safe or necessarily successful.[2] Near an obstacle, a direct move can reduce immediate distance yet have lower long-term value than a detour.

The Bellman expectation relation is

\[
V^{\pi}(s)=
\mathbb E_{\pi}\!\left[
r_{t+1}+\gamma V^{\pi}(s_{t+1})
\mid s_t=s
\right].
\]

Immediate reward and discounted next-state value jointly determine current value. Algorithms estimate them from finite data and approximating functions.

Values \(V,Q,A\) use the return scale. They are not distances or success probabilities unless explicitly defined that way. True termination normally has no continuation value; a sampling time limit may still permit future value. This distinction matters for bootstrapping.

## 5.5 Exploration and exploitation

**Exploitation** selects actions currently believed good. **Exploration** tries uncertain actions that may improve results. No exploration can trap an arm in a poor strategy; too much can cause variation, collision, excess speed or unstable grasps. Exploration must respect constraints.

Discrete policies assign action probabilities. Gaussian continuous policies vary the mean and dispersion, but sampled actions still need limits, monitoring and emergency-stop mechanisms.

Training often retains randomness for diverse experience; evaluation may use means, fixed seeds or reduced noise. Record the difference. One low-noise demonstration does not establish overall performance.

Exploration also changes visited states. Small noise stays near familiar trajectories; large noise may visit boundaries. Inspect coverage, constraint violations and failure types, not just standard deviation.

Even in simulation, unrestricted actions may create behaviors impossible on hardware. Real arms cannot explore without bounds. Safe regions, established controllers, demonstrations, existing data and simulation pretraining can reduce risk.[6]

## 5.6 Policy learning and actor–critic methods

Policy learning adjusts \(\theta\) to increase expected trajectory return:

\[
J(\theta)=
\mathbb E_{\tau\sim\pi_{\theta}}
\left[
\sum_{t=0}^{T-1}\gamma^t r_{t+1}
\right].
\]

The expectation includes initialization, policy sampling and transitions. Finite observations approximate this objective.

The **actor** receives observations and produces actions or their distribution. The **critic** estimates state or action value to assess outcomes relative to expectation. They are algorithmic roles, commonly separate or shared networks—not two robots.

1. The actor observes and acts.
2. The environment returns reward and the next observation.
3. The critic estimates value.
4. Returns and value estimates indicate relative action quality.
5. Updated actor and critic support the next data collection cycle.

The critic is an imperfect learned estimator, affected by missing information and coverage. Actor–critic methods do not automatically correct bad rewards, safety violations or the reality gap. Actor updates change visited states, so the critic must adapt; unstable estimates can destabilize actor learning. Sampling, value fitting and update size require balance.[3]

In Gymnasium, `terminated` usually denotes an MDP terminal state and `truncated` an external limit. Both end collection and require reset, but only genuine termination normally removes continuation value.[4] Time-limit truncation may require bootstrapping from the final valid observation. Task-defined finite horizons need their own treatment.[5]

Policy, body, transition, reward, return, value and exploration form one learning chain. High-return behavior still requires independent assessment of success, safety, generalization and executability.

> **Research outlook: RL on physical arms**
>
> Hardware interaction takes time, causes wear and involves collision, manual resets and sensor drift. Sample efficiency and safe exploration are central concerns.[6]
>
> Offline RL reduces new interaction by learning from existing data, but values for unsupported actions may be unreliable.[7] Demonstrations can initialize behavior before RL refinement, with limitations from demonstration quality and coverage.
>
> Simulation and hardware fine-tuning can reduce early trial costs without removing contact, friction, sensing, latency or actuator differences. None replaces deployment constraints and evaluation.

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Author-hosted edition](http://incompleteideas.net/book/RLbook2020.pdf)
2. OpenAI. *Spinning Up in Deep RL — Part 1: Key Concepts in RL*. [Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
3. OpenAI. *Spinning Up in Deep RL — Part 3: Intro to Policy Optimization*. [Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
4. Farama Foundation. *Gymnasium Env API*. [Documentation](https://gymnasium.farama.org/api/env/)
5. Farama Foundation. *Handling Time Limits*. [Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
6. Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement Learning in Robotics: A Survey. *The International Journal of Robotics Research, 32*(11), 1238–1274. [Paper](https://doi.org/10.1177/0278364913495721)
7. Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems*. [Paper](https://arxiv.org/abs/2005.01643)
8. Brunke, L., Greeff, M., Hall, A. W., Yuan, Z., Zhou, S., Panerati, J., & Schoellig, A. P. (2022). Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning. *Annual Review of Control, Robotics, and Autonomous Systems, 5*, 411–444. [Paper](https://doi.org/10.1146/annurev-control-042920-020211)

---

# Chapter 6 Training with PPO

Proximal Policy Optimization (PPO) collects a rollout with the current policy, estimates which actions were beneficial, compares new and old action densities and clips the incentive for excessive changes.[1][3]

This chapter distinguishes general equations, the **current S1 implementation** (`trainer.py`, `buffer.py`, `policy.py`, `normalizer.py`, `runMetrics.ts`), and **teaching examples**. Implementation facts refer primarily to the default `s1_push` contract; Reach and Grasp are conceptual examples where no source-specific claim is made.

## 6.1 Policy gradients: reinforcing beneficial actions

A policy defines an action distribution rather than predicting a teacher's prescribed action. A sample policy-gradient approximation is

\[
\nabla_\theta J(\theta)
\approx
\mathbb E\!\left[
\nabla_\theta\log\pi_\theta(a_t\mid o_t)\,\hat A_t
\right].
\]

Here \(J\) is expected return, \(o_t\) observation, \(a_t\) the sampled action, and \(\hat A_t\) estimated advantage. The log-density gradient changes how likely that action is; advantage controls direction and strength.[2]

Positive advantage tends to increase the action's likelihood in similar observations; negative advantage tends to decrease it; near-zero advantage contributes little. For Push, useful goal-directed object motion may have positive advantage, while sideways displacement may have negative advantage. The algorithm learns from rollout outcomes rather than knowing a correct direction in advance.

Theory maximizes a surrogate objective, while PyTorch optimizers minimize a loss:

\[
L_{\text{policy}}
=-\mathbb E\!\left[L^{\text{CLIP}}_t(\theta)\right].
\]

The negative sign converts ascent to descent; it does not punish every action. The S1 actor uses normalized rollout advantages, while raw advantages construct critic return targets.

## 6.2 Rollouts and generalized advantage estimation

PPO is **on-policy**. A rollout is a fixed collection of transitions under the current policy, not necessarily a full episode. It may span several episodes or part of one. The current default collects 32 transitions.

Each transition stores normalized observation, physical action, reward, old log-probability, value, next value and two episode-boundary masks. Old log-probabilities support density ratios; separately stored values support per-transition temporal-difference errors:

\[
\delta_t
=r_{t+1}+\gamma b_tV(o_{t+1})-V(o_t).
\]

The bootstrap mask \(b_t\) controls inclusion of next value. Positive TD error means the immediate-plus-future outcome exceeded the critic's estimate. Its scale follows reward, not physical distance or time.

One-step TD depends more on critic accuracy but tends to have lower variance. Longer realized returns rely less on the critic but fluctuate more. **Generalized advantage estimation (GAE)** balances these effects:[2]

\[
\hat A_t
=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
\]

The continuation mask is \(c_t\). Subsequent TD errors decay by powers of \(\gamma\lambda\). Smaller \(\lambda\) emphasizes short estimates; values near one use longer histories. Discount \(\gamma\) and GAE parameter \(\lambda\) have distinct roles and are not interchangeable.

| Transition | Bootstrap \(b_t\) | Continue GAE \(c_t\) | Meaning |
|---|---:|---:|---|
| Ordinary | 1 | 1 | Use next value and subsequent TD errors |
| True termination | 0 | 0 | No continuation value or cross-episode recursion |
| Time limit | 1 | 0 | Retain final-state value without importing the next episode's advantage |

A time limit can stop collection without ending the underlying task, so it permits bootstrapping. Reset starts a new episode, so its advantages must not flow backward into the previous one.[4]

At a nonterminal rollout boundary, the stored next value supplies the final TD term. Backward GAE starts with zero and does not invent observed advantages beyond the buffer. The critic target is

\[
\hat R_t=\hat A_t+V(o_t).
\]

The rollout's actor advantages are normalized by mean and population standard deviation:

\[
\tilde A_t=
\frac{\hat A_t-\mu_{\hat A}}
{\max(\sigma_{\hat A},10^{-8})}.
\]

Only the actor uses normalized advantage; the critic still regresses raw GAE returns. These are the meanings of “actor-only advantage normalization” and “raw GAE return critic target.” The following excerpt corresponds to `buffer.py`:

```python
gae = 0.0
for t in reversed(range(rollout_steps)):
    delta = reward[t] + gamma * bootstrap_mask[t] * next_value[t] - value[t]
    gae = delta + gamma * gae_lambda * continuation_mask[t] * gae
    advantage[t] = gae

return_target = advantage + value
```

## 6.3 New-to-old policy density ratios

Data were collected under \(\pi_{\theta_{\mathrm{old}}}\). As parameters change, the same sampled action has a different density:

\[
r_t(\theta)
=\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid o_t)}
=\exp\!\left(
\log p_{\text{new}}-\log p_{\text{old}}
\right).
\]

A ratio of one means unchanged density; above one increases preference, below one decreases it. For continuous actions this is a density ratio, since the probability of one exact value is zero.

The implementation stores `old_log_probs`, recomputes current log-probabilities and uses `exp(new_log_prob - old_log_prob)`. Defaults reuse a 32-step rollout for two shuffled epochs of 16-sample minibatches, so each complete sample is used twice. This improves utilization without treating on-policy rollouts as an indefinitely reusable replay buffer. New updates collect fresh data.

The current Push policy is a **tanh Gaussian**. The actor outputs Gaussian means with trainable `log_std`, samples an unconstrained action \(u_t\), then transforms it:

\[
a_t=\operatorname{bias}
+\operatorname{scale}\odot\tanh(u_t).
\]

The elementwise scale maps two XY components to [−0.005, 0.005] m per component; Z and orientation remain fixed. This is a Push-specific contract, not a universal S1 action definition.

<details>
<summary>Advanced: action densities and observation normalization in S1</summary>

Tanh and physical scaling require a change-of-variables correction in log-probability:

\[
\log \pi(a_t\mid o_t)
=\log \mathcal N(u_t;\mu_t,\sigma_t)
-\sum_i\log\!\left(
\operatorname{scale}_i(1-\tanh^2u_{t,i})+10^{-6}
\right).
\]

Without it, ratios no longer match the transformed physical-action distribution. Reevaluation inverse-scales stored actions, applies `atanh` and computes the same correction.

The actor receives normalized observations. `normalizer.py` uses Welford running statistics and clips normalized fields to [−10, 10]; previous-action fields already scaled to [−1, 1] retain their values. Statistics are checkpointed because identical weights with different normalization can produce different actions.

Before sampling, the current raw observation updates statistics and is normalized. Next-value estimation uses current statistics without counting the next observation again immediately. The rollout stores the normalized tensors used during sampling and reuses them in both epochs. Old log-probabilities, actions, values and observations thus share a consistent sampling context.

</details>

## 6.4 The clipped PPO objective

Maximizing an unclipped ratio-times-advantage objective can produce excessively large density changes. PPO-Clip uses[1][3]

\[
L^{\mathrm{CLIP}}(\theta)=
\mathbb E_t\!\left[
\min\!\left(
r_t(\theta)\hat A_t,
\operatorname{clip}\bigl(r_t(\theta),1-\varepsilon,1+\varepsilon\bigr)\hat A_t
\right)
\right].
\]

The current default \(\varepsilon=0.2\) gives ratio interval [0.8, 1.2]. This is not an action-coordinate range or a rule limiting physical motion to 20%.

For positive advantage, ratios above \(1+\varepsilon\) no longer gain extra clipped-objective benefit. For negative advantage, reducing a ratio below \(1-\varepsilon\) similarly removes further benefit. The minimum and the advantage sign produce asymmetric plateaus.

With advantage +1 and ratio 1.5, the contribution is clipped to 1.2. With advantage −1 and ratio 0.5, the objective is `min(-0.5, -0.8) = -0.8`.

Clipping removes an incentive; it does not hard-constrain parameters or keep every realized ratio in the interval. Shared parameter updates can still push some ratios outside. Nor is clipping a hard KL bound, a robot safety guarantee or an absolute guarantee of stable training. This implementation records approximate KL and clip fraction but does not use a KL threshold for early stopping.

```python
ratio = torch.exp(new_log_prob - old_log_prob)
unclipped = ratio * advantage
clipped = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage
policy_loss = -torch.minimum(unclipped, clipped).mean()
```

Here `advantage` is normalized, and the negative sign makes a maximization objective a minimization loss.

## 6.5 Actor loss, critic loss and entropy

Separate actor and critic MLPs are jointly optimized with one Adam optimizer. Total loss is

\[
L
=L_{\text{policy}}
+c_vL_{\text{value}}
-c_eH.
\]

Actor loss promotes positive-advantage actions and discourages negative-advantage actions under the clipped objective. Critic loss fits GAE return targets:

\[
L_{\text{value}}
=\mathbb E_t\!\left[
\bigl(V(o_t)-\hat R_t\bigr)^2
\right].
\]

This mean squared error depends on reward and return scales. Lower value loss does not mean the arm completed Push.

Entropy encourages continued stochasticity through the negative entropy term. The implementation reports the **pre-tanh base Gaussian entropy proxy**, not exact bounded-action entropy. It is measured in nats and is suitable for within-implementation trends, not direct comparison across distributions and scales.

The value and entropy coefficients set relative weights. Separate actor and critic parameters mean value loss does not directly update the actor. However, joint global gradient clipping can indirectly reduce actor steps when critic gradients are large. Excessive entropy can sustain too much randomness; too little can end exploration prematurely. Neither coefficient is a task success threshold.

| Parameter | Current default | Role |
|---|---:|---|
| `gamma` | 0.99 | Discount for TD and returns |
| `gae_lambda` | 0.95 | GAE horizon trade-off |
| `clip_ratio` | 0.2 | Policy density-ratio clipping |
| `value_coefficient` | 0.5 | Critic loss weight |
| `entropy_coefficient` | 0.01 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Joint gradient-norm bound |
| `rollout_steps` | 32 | Transitions collected per update |
| `update_epochs` | 2 | Optimization passes over one rollout |
| `batch_size` | 16 | Minibatch size |
| `learning_rate` | \(3\times10^{-4}\) | Adam step size |

These are implementation defaults, not universal optima. **PPO clipping** modifies the surrogate objective; **gradient clipping** limits the joint parameter-gradient norm after backpropagation. The trainer records raw actor and critic norms before clipping at 0.5. Neither replaces action limits or collision checks.

## 6.6 Training loops and diagnostic metrics

A current S1 update proceeds as follows:

1. Flatten observations under a fixed schema and normalize them.
2. Sample an action from the tanh Gaussian; estimate value.
3. Execute it and receive reward, next observation, flags and diagnostics.
4. Store 32 transitions, old log-probabilities, current/next values and masks.
5. Compute backward GAE, return targets and actor-only normalized advantages.
6. Shuffle each epoch and update actor and critic on minibatches.
7. Aggregate metrics into `training.jsonl` and episodes into `episodes.jsonl`.
8. At save intervals, write the latest checkpoint, evaluate and update the best-ranked checkpoint.
9. Collect a fresh rollout with the updated policy.

Updates require complete collected data. Cancellation is checked before environment steps; an optimizer update already in progress is completed to avoid a checkpoint containing only part of a minibatch update sequence.

A checkpoint includes parameters, optimizer state, progress, normalization statistics and configuration. Resuming also requires compatible observation, action, reward and PPO semantics.

<details>
<summary>Advanced: checkpoint contracts and version checks</summary>

Training resume validates observation schema/semantics, action schema/ranges, reward schema/weights and PPO training semantics. The documented checkpoint schema is version 4 and training semantics version 2. Strict matching prevents silently attaching old dimensions or meanings to a new Run. Evaluation loading can have different metadata requirements without implying training compatibility.

Saving writes a temporary file before replacing the final checkpoint. `model_latest.pt` is the latest saved state; `model_best.pt` follows the evaluation ranking, whose fallback can use current average reward if no evaluation is available. Neither filename alone proves formal task success.

</details>

Loss metrics are generally aggregated with minibatch-sample weighting. Read the eight main diagnostics as follows:

| Metric | Actual measurement | Possible interpretation and limitations |
|---|---|---|
| `average_reward` | Mean reward per step in the rollout | May improve under the current reward; not episode return or formal success rate; incompatible reward schemas cannot be ranked directly |
| `episode_length` | Mean length of episodes completed in the rollout, falling back to the current unfinished episode's accumulated length | Shorter can mean faster success or earlier failure; inspect ending reasons |
| `policy_loss` | Mean negative clipped objective with normalized advantages | Sign and magnitude depend on data; lower does not independently mean a better policy |
| `value_loss` | MSE against raw GAE returns | Lower may mean better fitting, but depends on scale and possibly a flawed objective |
| `entropy` | Pre-tanh Gaussian entropy proxy | Rapid decline may mean premature loss of exploration; increases may reflect unresolved behavior |
| `approx_kl` | Sample estimate of \(\mathbb E[(r_t-1)-\log r_t]\) | Sudden increases suggest large updates; no KL-based early stop is implemented; low KL is not proof of improvement |
| `clip_fraction` | Fraction satisfying \(|r_t-1|>\varepsilon\) | Some clipping is normal; persistently high values suggest aggressive changes; zero may mean almost no update |
| `explained_variance` | \(1-\operatorname{Var}(\hat R-V)/\operatorname{Var}(\hat R)\), with stored pre-update values; returns 0 for tiny return variance | Near one indicates explained variation; negative can be worse than a constant predictor; small rollouts fluctuate; not success rate |

Missing metrics are not zero. Check whether a field was produced, transmitted and applicable.

<details>
<summary>Advanced: metric files and live callbacks</summary>

`runMetrics.ts` treats KL, clip fraction and explained variance as optional finite values. `trainer.py` writes them to training JSONL, but the documented live metric callback does not include them. A live curve may therefore lack them even when files contain them. `learning_rate` and `steps_per_second` are explicitly not reported by the current Simulation callback; no curves should be invented.

</details>

Interpret metrics together. Rising reward, falling entropy and stable KL may indicate exploitation, but do not establish learning without improved independent success. Falling value loss with near-zero explained variance may reflect smaller return scale. KL and clip-fraction spikes justify checking learning rate, advantage outliers and reuse intensity, not declaring one certain cause.

The frontend accepts structurally complete deterministic evaluation records and exports success, returns, distances, timeout and failure rates separately from training events. Confidence intervals and scientific comparisons follow in Chapter 8. Diagnosability depends on sampling evidence, optimization metrics, checkpoint semantics and independent evaluation together.

> **Research outlook: alternatives to PPO**
>
> PPO is useful for continuous simulated control but is on-policy; old data quickly become unrepresentative. Contact-rich, sparse-reward Push or Grasp can remain difficult. Demonstrations can initialize behavior; SAC and TD3 reuse historical transitions; offline RL learns from fixed datasets. Each introduces its own coverage, distribution-shift and value-estimation problems and is not automatically superior.[5][6][7]

PPO optimizes a problem whose observation, action and reward are assumed fixed. Changing those interfaces changes the learning problem itself, which is the subject of the next chapter.

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. [Paper](https://arxiv.org/abs/1707.06347)
2. Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *International Conference on Learning Representations*. [Paper](https://arxiv.org/abs/1506.02438)
3. OpenAI. *Spinning Up in Deep RL — Proximal Policy Optimization*. [Tutorial](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. Farama Foundation. *Handling Time Limits*. [Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
5. Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *Proceedings of AISTATS, 15*, 627–635. [Paper](https://proceedings.mlr.press/v15/ross11a.html)
6. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *Proceedings of ICML, 80*, 1861–1870. [Paper](https://proceedings.mlr.press/v80/haarnoja18b.html)
7. Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *Proceedings of ICML, 80*, 1587–1596. [Paper](https://proceedings.mlr.press/v80/fujimoto18a.html)
8. S1 platform source: `trainer.py`, `buffer.py`, `policy.py`, `normalizer.py`  and  `runMetrics.ts` (version reviewed for this chapter).

---

# Chapter 7 Observations, Actions and Rewards

PPO defines how to update a policy, not the task itself. Observation defines decision evidence, action defines possible influence on the environment, and reward defines preferred outcomes. Changing a field's meaning, order, unit or range can change the learning problem even with the same robot and algorithm.

## 7.1 Observation content and representation

State \(s_t\) models the environment, potentially including all joints, object motion and contacts. Observation \(o_t\) is the information actually supplied after sensing, selection, transforms and encoding:[1][2]

\[
o_t=h(s_t)+\epsilon_t.
\]

Here \(h\) represents the information channel and \(\epsilon_t\) measurement or modeling error. The equation does not claim the current S1 implementation adds random noise to every field.

Networks consume numerical tensors. The current Reach schema flattens fields in this order:

\[
o_t=
[q_{1:6},\ \dot q_{1:6},\ \Delta p^{\mathrm{ee\to goal}}_{xy},\
v^{\mathrm{ee}}_{xy},\ a_{t-1}],
\]

Its 18 components are six joint angles in rad, six velocities in rad/s, two relative endpoint-to-goal XY components in m, two endpoint velocities in m/s, and two normalized previous-action components. The implementation checks field order, lengths and finiteness rather than relying on dictionary ordering.

This plain Python version returns tuples for inspection; the real function produces `torch.float32` tensors:

```python
import math

REACH_FIELDS = (
    "arm_q", "arm_dq", "ee_to_goal_xy",
    "ee_velocity_xy", "previous_action_xy",
)
REACH_SIZES = (6, 6, 2, 2, 2)

def flatten_observation(observation, fields, sizes):
    flat = []
    for name, expected_size in zip(fields, sizes):
        if name not in observation:
            raise ValueError(f"Missing observation field: {name}")
        values = tuple(float(v) for v in observation[name])
        if len(values) != expected_size:
            raise ValueError(f"{name} must have {expected_size} components")
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name} contains a nonfinite value")
        flat.extend(values)
    return tuple(flat)

reach_observation = {
    "arm_q": (0.0,) * 6,
    "arm_dq": (0.0,) * 6,
    "ee_to_goal_xy": (0.030, -0.010),
    "ee_velocity_xy": (0.002, 0.000),
    "previous_action_xy": (0.0, 0.0),
}

vector = flatten_observation(
    reach_observation, REACH_FIELDS, REACH_SIZES
)
print(len(vector))   # 18
print(vector[12:14]) # (0.03, -0.01), relative endpoint-to-goal XY position
```

Reordered dictionary keys do not change the result. Missing fields, incorrect dimensions or NaN are rejected. Swapping field meanings while retaining length 18 remains incompatible with an old policy.

Relative coordinates directly express task relationships. Translating tool and goal together can leave their difference unchanged. Reach uses relative goal direction; Push distinguishes approach-to-object from object-to-goal progress. Absolute coordinates may still be necessary for obstacles and workspace boundaries. Always specify the reference frame.

The current **24D Push observation** contains six joint angles, six joint velocities, two tool-to-object XY components, two object-to-goal XY components, two endpoint XY velocities, two object XY velocities, two persistent-target tracking errors and two previous actions. Differences use meters and velocities m/s; previous physical actions are divided by the per-axis 0.005 m scale. No separate frame field is specified, so these are described only as differences within the same simulation coordinate representation.

The **40D Grasp observation** contains six normalized arm positions; six joint velocities scaled by 10 and clipped; two normalized finger positions; two finger velocities scaled by 0.2; three workspace-normalized tool positions; three object positions scaled relative to nominal initialization; six object linear/angular velocities; three relative tool–object positions; one estimated opening; four previous actions; three grasp-corridor errors; and one object–table clearance. A value of 1 is not generally one meter or one radian after these transforms.

A contract includes names, order, units, frame meaning, scaling and version—not just dimension. Changing radians to degrees or swapping relations can invalidate a checkpoint without changing vector length.

Reach/Push maintain running normalization statistics while preserving already scaled previous actions. Grasp has task-specific environment scaling. Normalization cannot recover absent contact information, occluded poses or missing history.

**Viewer video is not policy vision.** These task schemas and flattening functions consume numerical fields, not camera tensors, even when frames are captured for people. A future visual policy would need a new contract for image shape, calibration, preprocessing and timing.

## 7.2 Partial observability and history

Real robots seldom observe a complete Markov state. Occlusion, unmeasured contact, unknown friction and velocities absent from a single frame can make distinct physical states produce identical observations.[1][2]

Simulation can directly expose object velocities and exact poses that hardware must estimate with noise and latency. Training on unavailable ground truth creates an observation gap.

Position alone cannot distinguish a stationary object from one moving through the same point. Velocity or recent positions can. Previous actions help explain tracking error but supply only limited one-step history, not complete memory.

The following example demonstrates **observation aliasing**:

```python
def position_only_observation(state):
    return (state["object_x"],)

moving_right = {"object_x": 0.45, "object_vx": 0.20}
moving_left  = {"object_x": 0.45, "object_vx": -0.20}

print(position_only_observation(moving_right)) # (0.45,)
print(position_only_observation(moving_left))  # (0.45,)
```

Seeing only `(0.45,)` cannot reveal leftward versus rightward motion. Adding `object_vx` or estimating velocity from consecutive positions can resolve this ambiguity.

Possible remedies include measured or estimated velocities/contacts, frame stacking, sensor fusion, or recurrent and Transformer histories. The needed history depends on the missing variable: a few frames may reveal velocity but not constant unknown friction.

More input is not always better. Irrelevant noisy fields increase fitting difficulty; privileged simulator information can harm transfer; duplicate fields alter sensitivity. Choose information needed for the task and reliably obtainable.

## 7.3 Action spaces and control interfaces

Actions may specify joint position, velocity, torque, tool pose, pose increments or gripper commands. These distribute responsibility differently between policy, IK and low-level control.[3]

A typical chain is policy output → scaling and workspace constraints → tool target → IK/controller → joint commands → physics or hardware. A normalized persistent-target update is

\[
x_{t+1}^{\mathrm{target}}
=
\operatorname{clip}_{\mathcal W}
\left(
x_t^{\mathrm{target}}+\Delta x_{\max}a_t
\right),
\qquad a_t\in[-1,1]^n.
\]

The target and maximum increment use meters; \(a_t\) is dimensionless; clipping constrains the target to workspace \(\mathcal W\).

**Reach** uses normalized 2D XY actions in [−1, 1], a 0.005 m per-axis scale, fixed Z/orientation and workspace clipping. Action (0.6, −0.4) increments the target by (0.003, −0.002) m; it does not guarantee immediate tool motion.

```python
def map_reach_action(current_target, action, scale=0.005,
                     workspace=((0.20, 0.70), (-0.30, 0.30))):
    if len(action) != 2 or any(abs(a) > 1.0 for a in action):
        raise ValueError("Reach action must be a 2D vector in [-1, 1]")

    proposed = (
        current_target[0] + scale * action[0],
        current_target[1] + scale * action[1],
    )
    return tuple(
        max(low, min(high, value))
        for value, (low, high) in zip(proposed, workspace)
    )

next_target = map_reach_action(
    current_target=(0.440, 0.010),
    action=(0.6, -0.4),
)
print(next_target)  # (0.443, 0.008) m
```

This function only maps actions to targets. IK, joint control and physics remain downstream. At a workspace edge, an outward command can change without changing the clipped target, producing saturation.

**Push** instead receives physical XY increments already scaled by the policy to [−0.005, 0.005] m. Do not multiply (0.003, −0.002) by 0.005 again. Only its previous-action observation is renormalized by that scale.

**Grasp** uses four normalized components. XYZ increment a persistent target; XY scale is 0.005 m, while Z becomes 0.0025 m near the table instead of its usual 0.005 m. The fourth maps to an **absolute symmetric gripper target**, 0.025–0.05, not an increment. Target synthesis then limits per-step gripper change to 0.001 and applies workspace/joint constraints. Tool orientation stays fixed.

These tasks therefore do not share one universal “normalized 2D displacement” interface. Dimensions, units, ranges and semantics must accompany the checkpoint.

## 7.4 Action scale, control frequency and constraints

Small actions can be smoother but require more decisions and make sparse goals harder to discover. Large actions explore farther but can overshoot, saturate or disturb contact. Scale must be chosen with geometry, tracking and decision period.

If one action lasts \(N_{\mathrm{substeps}}\) physics steps of duration \(\Delta t\):

\[
T_{\mathrm{action}}
=N_{\mathrm{substeps}}\Delta t,
\qquad
f_{\mathrm{action}}=\frac{1}{T_{\mathrm{action}}}.
\]

The same displacement sent every 0.02 s versus every 0.2 s produces different effective speeds and contact dynamics.

Current Reach, Push and Grasp settings advance 8, 1 and 4 physics steps per action respectively. These are environment settings, not hardware specifications. Without the underlying physical step duration, a frequency in hertz cannot be inferred.

Constraints act at several layers: policy range, action scaling, workspace and gripper-rate limits, joint limits, reachability checks and task termination for invalid states or dangerous contact.

Do not confuse **action clipping**, **PPO ratio clipping** and **gradient clipping**. They constrain commands/targets, optimization incentives and parameter gradients respectively. One does not imply the protection provided by another.

## 7.5 Reward design and shaping

Reward maps a transition to a scalar, often combining terms:

\[
r_t=\sum_i w_i\phi_i(t).
\]

Terms \(\phi_i\) are already calculated contributions; \(w_i\) are combination weights. Reward is usually a designed score. Increasing a positive weight on an already negative term increases its penalty.

A distance-progress term is

\[
r_t^{\mathrm{progress}}
=k(d_t-d_{t+1}).
\]

Distances are in meters; \(k\) maps their change to reward scale. An action-magnitude cost is

\[
r_t^{\mathrm{action}}
=-\lambda\lVert a_t\rVert_2^2,
\]

It penalizes action size, not successive action differences, so calling it a smoothness penalty would be inaccurate.

### Comparing four common reward definitions

| Definition | Example | Information | Main limitation |
|---|---|---|---|
| Sparse success | 1 on success, otherwise 0 | Direct outcome feedback | Rare success gives little exploratory direction |
| Negative distance | \(-d_{t+1}\) | Current proximity every step | Depends on units and episode length; proximity need not mean success |
| Distance progress | \(k(d_t-d_{t+1})\) | Direction of improvement | Can ignore collision, orientation and cost |
| Combination | Progress + success − costs | Progress, endpoint and control cost | Imbalanced terms invite reward exploitation |

The following example evaluates the same transitions under all four definitions. `s1_reach_like` uses default progress scale 100, success bonus 2.0 and action-cost coefficient 0.002, assuming zero joint-limit cost. Other definitions are teaching alternatives, not simultaneous production configurations.

```python
def compare_rewards(d_t, d_next, action, success):
    action_sq = sum(a * a for a in action)

    sparse = 1.0 if success else 0.0
    negative_distance = -d_next
    progress = 100.0 * (d_t - d_next)

    s1_reach_like_terms = {
        "progress_reward": progress,
        "success_bonus": 2.0 if success else 0.0,
        "action_penalty": -0.002 * action_sq,
        "joint_limit_penalty": 0.0,
    }
    return {
        "sparse": sparse,
        "negative_distance": negative_distance,
        "progress": progress,
        "s1_reach_like": sum(s1_reach_like_terms.values()),
        "terms": s1_reach_like_terms,
    }

examples = [
    ("Effective approach", 0.100, 0.080, (0.6, -0.2), False),
    ("Large action, little progress", 0.100, 0.099, (1.0, 1.0), False),
    ("Entered the success region", 0.021, 0.018, (0.2, 0.1), True),
]

for name, d_t, d_next, action, success in examples:
    result = compare_rewards(d_t, d_next, action, success)
    values = {key: round(result[key], 4)
              for key in ("sparse", "negative_distance",
                          "progress", "s1_reach_like")}
    print(name, values)
```

| Transition | Sparse | Negative distance | Progress | Reach-style combination |
|---|---:|---:|---:|---:|
| 0.100 → 0.080 m; squared action sum 0.40; no success | 0 | −0.080 | 2.000 | 1.9992 |
| 0.100 → 0.099 m; squared action sum 2.00; no success | 0 | −0.099 | 0.100 | 0.0960 |
| 0.021 → 0.018 m; squared action sum 0.05; success | 1 | −0.018 | 0.300 | 2.2999 |

Sparse reward cannot distinguish approach from inactivity before success. Progress provides direction, costs discourage excessive actions, and the success bonus remains an independently detected event. Do not infer success from a reward threshold.

Negative distance evaluates each visited state; progress evaluates change and telescopes when undiscounted with no other terms. Discounting, termination, costs and stage gates alter this relation. Calculate contributions on concrete trajectories before tuning weights.

Current **Reach** uses XY progress ×100, normalized action squared magnitude ×−0.002, protected joint-limit penalties, and a one-time success bonus after staying within 0.02 m for three policy steps.

**Push** rewards progress toward a suitable precontact point behind the object, penalizes wrong-side or misaligned approach, rewards goal-directed object displacement, penalizes lateral drift and adds a small contact-sustain bonus. It also includes success, action and joint-limit terms. Success requires 3D object–goal distance at most 0.05 m without failure; contact alone is insufficient.

**Grasp** uses stage-dependent approach and alignment, alignment-gated descent, safely positioned closure progress and bilateral-contact-gated blockage progress. Candidate and verifier events add rewards. Action size, saturation, constrained targets and non-timeout failures contribute penalties.

Shaping supplies denser guidance for approach, alignment, contact and lift. It also creates exploitable incentives: too much contact reward encourages touching only; too much action cost encourages inactivity; loose success checks invite shortcuts.

Potential-based shaping has a special form with a policy-invariance guarantee under its assumptions:[4]

\[
F(s_t,s_{t+1})
=\gamma\Phi(s_{t+1})-\Phi(s_t).
\]

The state potential is \(\Phi\). Current S1 gates, contact bonuses, costs and terminal events do not justify claiming every reward term satisfies that theory.

Keep reward and success separate. Run `completed` describes lifecycle completion, not Policy passed. Reach requires holding, Push requires object proximity, and Grasp requires verification; timeout, failure and success have different meanings.

The teaching editor presents understandable term labels and nonnegative weights from 0 to 5. Its displayed contributions are explicitly illustrative. Advanced expressions allow one `combine_reward(terms)` function, allowlisted terms, finite constants and basic arithmetic under size, complexity and output bounds—not arbitrary Python.

The editable extension supports Reach, Push and Place, not experimental Grasp. Reach exposes progress and action cost; Push exposes approach, alignment, directional push, lateral drift, contact and action cost. The backend adds protected `success_bonus` and `joint_limit_penalty`. Editing scoring does not remove object, workspace, joint or dangerous-contact termination rules.

## 7.6 Reach, Push and Grasp interface design

The following are current task settings, not permanent S1 hardware specifications:

| Task | Observation | Action | Reward focus | Success | Common non-success endings |
|---|---|---|---|---|---|
| Reach | 18D: joints, goal relation, tool velocity, previous action | 2D normalized XY increments; 0.005 m scale; fixed Z/orientation | Progress, success, action and limit costs | Within 0.02 m for 3 steps | Nonfinite state, workspace violation, 128-step timeout |
| Push | 24D: joints, tool/object/goal relations, velocities, tracking error, previous action | 2D physical increments, ±0.005 m per axis | Precontact alignment, directional progress, lateral/contact terms and costs | Object–goal distance ≤0.05 m, no failure | Object falls/leaves table; 256-step timeout; contact is not success |
| Grasp | 40D normalized arm/finger/object state, relations, opening, action history, corridor error and clearance | Normalized XYZ + absolute symmetric opening; smaller near-table Z scale; gripper-rate limit | Staged approach, alignment, descent, closure, blockage, candidate and verification | Stable candidate followed by `grasp_verified` lift verification | Nonfinite/constraint/contact failures, premature pushing, failed verification; 120 learning-step timeout |

Reach emphasizes planar geometry; Push adds contact-driven object dynamics; Grasp adds finger coordination, bilateral contact and verification. A closed gripper does not alone demonstrate a stable grasp.

Current adapters expose one `done` and report `success`, `timeout`, `failure` and reasons in `info`. `done=True` is not success. Grasp stores `TIMEOUT` in `failure_reason`, making its failure boolean true as well; prioritize timeout and explicit reasons instead of mechanically comparing booleans across tasks. Gymnasium wrappers must map termination and truncation correctly for bootstrapping.[5]

Observation bounds evidence, action bounds physical influence, reward guides optimization, and independent success and constraint rules remain essential. Changing any of these can produce a different learning problem under the same PPO implementation.

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Author-hosted edition](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)
2. OpenAI. *Spinning Up in Deep RL — Part 1: Key Concepts in RL*. [Tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
3. Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press. [Course and textbook](https://modernrobotics.northwestern.edu/nu-gm-book-resource/)
4. Ng, A. Y., Harada, D., & Russell, S. J. (1999). Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. *Proceedings of ICML*, 278–287. [Paper](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
5. Farama Foundation. *Gymnasium Env API*  and  *Spaces*. [Environment API](https://gymnasium.farama.org/api/env/); [Spaces API](https://gymnasium.farama.org/api/spaces/)
6. S1 platform source: `policy.py`, `reach_adapter.py`, `push_adapter.py`, `grasp_schema.py`, `grasp_env.py`, `normalizer.py`, `rewardTeaching.ts`, `RewardDesigner.tsx` and `reward_extension.py` (version reviewed for this chapter).

---

# Chapter 8 Metrics, Evaluation and Scientific Experiments

Training diagnostics, independent evaluation of a frozen policy and reproducible controlled experiments belong to one evidence chain. Rising reward or a convincing video cannot alone establish reliable Reach, Push or Grasp performance. Scientific evaluation specifies conditions, samples and decisions before observing results.

## 8.1 Training metrics and task metrics

A metric is meaningful only with its samples, stage and calculation defined. Training metrics diagnose optimization; task metrics measure behavior.

| Training metric | Current meaning | Diagnostic question |
|---|---|---|
| `average_reward` | Mean step reward in the rollout | Does sampled behavior fit the current reward? |
| `episode_length` | Mean length of completed rollout episodes, with the documented fallback | How long do episodes last, and why do they end? |
| `policy_loss` | Mean PPO policy loss | What surrogate objective updates the actor? |
| `value_loss` | Error against return targets | Is the critic fitting current targets? |
| `entropy` | Policy entropy proxy | How does randomness change? |
| `approx_kl` | Approximate policy change | Might updates be too large? |
| `clip_fraction` | Fraction outside the clipping interval | How many samples cross that interval? |
| `explained_variance` | Explained return variation | Does the critic capture differences between samples? |

These are not task grades. Lower value loss does not prove reaching; policy loss is not a universal leaderboard score; shorter episodes can mean success or early failure.

Formal task metrics include success, episode length/return, final/minimum distance, object displacement, timeout and explicit failure rates. Reach needs endpoint and holding evidence; Push needs object motion; Grasp distinguishes candidate from verified success.

| Task metric | What it measures | What it does not establish alone |
|---|---|---|
| Success rate | Fraction meeting the formal criterion | Guaranteed future or out-of-distribution success |
| Timeout rate | Episodes reaching their limit without success | One certain cause |
| Failure rate | Episodes explicitly marked failed | A single category for all non-success |
| Final distance | Distance at episode end | Whether it was previously smaller |
| Minimum distance | Best distance during the episode | Stable final holding |
| Episode return | Accumulated designed reward | Independent task success |
| Object displacement | Motion relative to initialization | Correct direction or goal completion |
| Candidate/verified rates | Passing preliminary/final grasp checks | Interchangeability of these stages |

Read optimization and task metrics together without substituting one for the other. Run `completed` does not imply Policy passed. Missing evaluation is not 0% success.

## 8.2 Reward curves, success and task error

Reward is an optimization signal; success rate measures a formal outcome. Shaping terms can improve reward without improving task completion.

A documented Push summary illustrates this: 100 episodes had mean return 0.1695297 but zero successes, timeout rate 1.0 and mean object displacement approximately \(2.31\times10^{-6}\) m. Positive return does not mean the policy learned to push. Nor can this return be ranked against a different reward schema.

Two Reach policies can both score 90% success while their failed episodes differ dramatically in distance. Distinguish

\[
d_{\mathrm{final}}=d_T,
\qquad
d_{\mathrm{min}}=\min_{0\le t\le T}d_t.
\]

where \(d_t\) is task-defined distance and \(T\) the final step. A small minimum but large final distance suggests approach without retention. The task schema determines whether distance means tool-to-goal or object-to-goal.

A moving average of window \(w\) is

\[
\bar x_t^{(w)}=
\frac{1}{w}\sum_{j=0}^{w-1}x_{t-j}.
\]

It smooths existing points, hides abrupt changes and introduces lag; it adds neither seeds nor episodes. Preserve raw data, window size and axis definitions.

These documented summaries explain evidence, not a cross-task ranking:

| Source | Recorded result | Supported conclusion | Unsupported conclusion |
|---|---|---|---|
| `reach_evaluation_summary.json` | 100/100 successes; Wilson 95% interval ≈[96.30%, 100%]; final/minimum distance ≈0.01373 m; no timeouts/failures | All sampled episodes met the criterion | Guaranteed 100% success in every condition |
| `push_evaluation_summary.json` | Return ≈0.16953; 0/100 success; 100% timeout; displacement ≈0.00000231 m | Positive reward did not translate to task success | Smoothing or a positive score proves learning |
| `grasp_randomized_100_summary.json` | 100 candidates, 98 verified; two `GRASP_VERIFICATION_FAILED`; Wilson ≈[93.00%, 99.45%] | Verification rejected two candidates | Contact or closure alone proves grasping |
| `grasp_fixed_tuned_summary.json` | 0/100 success, 100 `TIMEOUT`; Wilson upper bound ≈3.70% | No success in that recorded configuration | “Tuned” in a filename establishes effectiveness |

Videos reveal behavior but do not increase the statistical sample. When a successful seed exists, the evaluator replays the smallest successful seed in a separate process. This is traceable success evidence, not a representative sample of all 100 episodes. With no successful seed, unavailable replay is legitimate; training video must not replace formal success evidence.

## 8.3 Independent evaluation and random seeds

Training explores and updates parameters. Formal evaluation freezes a selected checkpoint and protocol. Continuing to tune during testing leaks test information into model selection.

S1 evaluation loads weights and normalization, freezes statistics, uses evaluation mode and `no_grad`, chooses deterministic actions, creates an environment for each prescribed seed, records episodes and writes a summary. Production evaluation uses a separate Python process after training resources close. Evaluation errors do not rewrite completed training state.

Deterministic actions remove policy-sampling noise, not environmental variation. Formal evaluation requires 100 complete episodes. The generic configuration uses unique nonnegative seeds and rejects too few, duplicate or invalid values. Versioned task scripts may use different seed sets; record the actual list or generation rule. The supplied Reach summary records target `base_seed=10000`, so do not replace it with an assumed 0–99 range.

A seed initializes pseudorandomness; it does not guarantee bitwise equality across hardware, libraries or scheduling. It controls some randomness and permits paired conditions, without replacing multiple training seeds or covering unknown physical variation.

Do not select favorable seeds or repeatedly choose checkpoints after seeing final-test outcomes. Use validation for selection and reserve a frozen final protocol.

Evaluation states are:

- **`passed`:** valid artifacts and quality thresholds satisfied.
- **`failed`:** valid evaluation evidence, but policy quality below threshold.
- **`unavailable`:** absent or invalid evidence, including process, file or numerical-contract failures.

General documented thresholds are 100 episodes, success rate at least 80%, and Wilson lower bound at least 70%. Reach uses at least 90% and 80% respectively. These are platform acceptance rules, not universal statistical laws. A completed Run can have failed or unavailable evaluation. Missing exported values remain blank, not zero.

Sample success rate is

\[
\hat p=\frac{k}{n}.
\]

Record counts \(k,n\), not only percentages: 8/10 and 80/100 have the same point estimate but different uncertainty.

## 8.4 Baselines and fair comparisons

A baseline answers “Improved relative to what?” It may be default PPO, a rule controller, an ablated reward or a previous checkpoint. It must be implemented correctly with reasonable budget and the same evaluation protocol—not deliberately weakened.

Change one factor while fixing task/schema, observation order, action ranges/timing, reward, success criteria, training steps, checkpoint selection, training seeds, evaluation seeds and episode count. Report relevant compute, runtime and software versions; unrecorded resources are not zero.

Different reward weights change the scale, so compare shared physical outcomes and success rules rather than ranking raw returns. Different training budgets cannot establish an algorithmic improvement; sample-efficiency claims need performance at matched data budgets.

| Experiment | Changed factor | Fixed conditions | Seeds | Budget | Main outcomes |
|---|---|---|---|---|---|
| Baseline | None | Task, schema, reward, action and protocol | Prespecified training/evaluation lists | Environment steps | Success and Wilson interval |
| Variant A | For example, learning rate | Otherwise identical | Same sets | Matched | Success, distance and failures |
| Variant B | One observation field | Otherwise identical; version the schema | Same sets | Matched | Same outcomes and explicit interface change |

Checkpoint selection also matters. The trainer can rank optional validation results by evaluation availability, success, Wilson lower bound, smaller distance and reward. However, the documented `training_task_runner.py` call does not supply that callback, so training `model_best.pt` falls back to average reward, and final evaluation runs on `model_latest.pt`. State which checkpoint was evaluated and how it was chosen; filenames alone do not prove best formal success.

## 8.5 Ablations and hyperparameter experiments

Hyperparameter studies vary how training works; reward-weight studies vary optimization incentives; **ablations** remove or replace a component to test its contribution.

Observation ablations may remove velocity, previous action or a relative position. Reward ablations may remove progress, contact or action cost. Removing input fields versions the observation contract and usually invalidates direct reuse of old weights. Changing reward invalidates direct numerical comparison of returns.

An ablation should state what changed, why it should matter, what remained fixed, whether differences exceed training/evaluation variation, and how failure types changed. Lower Push reward after removing directionality does not alone identify the cause; inspect lateral motion, timeouts and multiple training seeds.

Prespecify hyperparameter candidates, select on training/validation conditions, then freeze before final testing. With limited compute, reduce candidates and acknowledge uncertainty rather than reporting a lucky best Run.

Keep tasks distinct. Reach, Push and Grasp differ in observations, rewards and success rules; candidate and verified grasp rates are not equivalent. Cross-task generalization needs a defined shared metric and sampling protocol.

## 8.6 Uncertainty, failure analysis and reproducible reports

Finite episodes provide finite evidence. Let \(\hat p\) be observed success rate and \(p\) the unknown long-run rate. The platform uses a Wilson score interval with \(z=1.959963984540054\):

\[
D=1+\frac{z^2}{n},
\qquad
C=\frac{\hat p+\frac{z^2}{2n}}{D},
\]

\[
H=\frac{z}{D}
\sqrt{
\frac{\hat p(1-\hat p)}{n}
+\frac{z^2}{4n^2}
}.
\]

The interval is

\[
\left[
\max(0,C-H),
\min(1,C+H)
\right].
\]

All these quantities are dimensionless. The implementation explicitly sets the lower bound to zero for zero successes and the upper bound to one for all successes.

For 98/100, the interval is approximately [93.00%, 99.45%]; for 100/100, [96.30%, 100%]; for 0/100, the upper bound is approximately 3.70%. For 0/10 it is about 27.75%. No observed success does not establish a true rate of zero, nor do all observed successes guarantee perfection.

This standard-library function matches the documented calculation:

```python
import math

def wilson_interval(k, n, z=1.959963984540054):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Require n > 0 and 0 <= k <= n")

    p_hat = k / n
    denominator = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)
    )
    lower = 0.0 if k == 0 else max(0.0, center - half_width)
    upper = 1.0 if k == n else min(1.0, center + half_width)
    return p_hat, lower, upper

for k, n in [(98, 100), (100, 100), (0, 100), (0, 10)]:
    p, low, high = wilson_interval(k, n)
    print(f"{k}/{n}: p={p:.2%}, 95% CI=[{low:.2%}, {high:.2%}]")
```

It checks numerical summaries, not protocol quality. Selected, dependent or incompatible episodes still produce numbers without satisfying the intended sampling assumptions.

A frequentist 95% confidence interval does not assign 95% probability to the fixed parameter lying in this particular interval. Under repeated use of the sampling protocol, about 95% of constructed intervals would cover it. Selected seeds, changed environments or strongly dependent episodes undermine that interpretation.

The interval measures evaluation sampling uncertainty, not variability of training. Compare algorithms using independent training seeds, evaluating each resulting checkpoint under the same fixed protocol.

Failure analysis should distinguish:

- **Timeout:** exhausted episode budget; not automatically collision or boundary failure.
- **Explicit failure:** task-defined invalid outcome, with a recorded reason.
- **Distance shortfall:** never approached versus approached and left, using final/minimum distance.
- **Action saturation:** frequent interface-boundary outputs, a diagnostic rather than failure proof.
- **Constraint termination:** workspace, joint or contact violations, separate from ordinary timeout.
- **Candidate without verification:** preliminary grasp did not survive later tests.
- **Unavailable evaluation:** incomplete files, hashes, records or processes; missing values are not zero.

Preserve episode records and state whether categories overlap. One failure video loses frequency; one aggregate failure rate loses behavior structure. The 100-candidate/98-verified grasp funnel and Push's 100% timeout with 0% explicit failure illustrate why categories matter.

| Report item | Required information |
|---|---|
| Task contract | Task ID and observation/action/reward/success versions |
| Code and environment | Commit or image, dependencies, OS, CPU/GPU and simulator |
| Model identity | Checkpoint path/type/SHA-256 and frozen normalization |
| Training | PPO settings, reward weights, scales, timing and initialization/transfer source |
| Randomness | All training seeds and fixed evaluation seeds or reproducible rule |
| Budget | Environment/update steps and recorded runtime/resources |
| Evaluation | Count, deterministic setting, limits, thresholds and checkpoint selection |
| Results | Counts, rates, Wilson interval, distance/displacement, returns and failures |
| Raw evidence | Original summaries, episodes, curves and missing-field explanations |
| Behavior | Traceable success and representative failure artifacts with seed and hash |
| Limitations | Untested distributions, physics, sensors, hardware and known anomalies |

Label teaching arrays separately from real Run data. Describe only fields actually present in a real summary; neither filenames nor appearance fill missing evidence.

Integrity checks require matching summaries and 100 episode records, consistent counts/rates/intervals, finite values, checkpoint SHA-256 and replay seed/hash consistency. These do not guarantee a good policy, but establish which model and records support a claim.

A defensible conclusion links task definitions, optimization diagnostics, independent evaluation, controls, uncertainty and failures. Claims beyond tested conditions require further evidence.

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Author-hosted edition](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)
2. Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning That Matters. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1). [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
3. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. *Advances in Neural Information Processing Systems, 34*. [Paper](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
4. Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association, 22*(158), 209–212. [DOI](https://doi.org/10.1080/01621459.1927.10502953)
5. PyTorch. *Reproducibility*. [Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)
6. Stable-Baselines3. *Reinforcement Learning Tips and Tricks*. [Documentation](https://stable-baselines3.readthedocs.io/en/stable/guide/rl_tips.html)
7. Farama Foundation. *Gymnasium Env API*. [Documentation](https://gymnasium.farama.org/api/env/)
8. S1 platform source: `runMetrics.ts`, `trainer.py`, `evaluation.py`, `evaluation_subprocess.py`, `training_task_runner.py`, `task_registry.py` and `api.ts`; tests: `test_evaluation.py`, `test_reach_evaluation.py`, `test_evaluation_subprocess.py` and `RunEvaluation.test.tsx`; recorded artifacts: `reach_evaluation_summary.json`, `push_evaluation_summary.json`, `grasp_randomized_100_summary.json` and `grasp_fixed_tuned_summary.json` (version reviewed for this chapter).

The next chapter asks how far evidence extends when goals, objects, physics, sensors or embodiment change. Generalization claims depend on the controlled evaluation and uncertainty principles established here.

---

# Chapter 9 Transfer Learning and Generalization

Passing evaluation establishes performance under specified tasks, parameters and samples. Changed goals, friction or hardware can change observations and action consequences. Generalization asks where learned behavior remains valid; transfer asks whether existing knowledge reduces learning cost under target conditions. Neither is proved by loading a checkpoint or changing a seed.

## 9.1 Generalization, transfer and distribution shift

RL data depend on policy–environment interaction. Write training and test joint distributions as

\[
(s,a,r)\sim p_{\mathrm{train}},
\qquad
(s,a,r)\sim p_{\mathrm{test}}.
\]

This does not assume state, action and reward independence; policy, initialization, physics and sensors jointly determine experience.

**Generalization** means accomplishing the same task under unseen conditions while maintaining its interface and success definition. New Reach goals inside the covered region test interpolation; goals or properties outside it test extrapolation or out-of-distribution behavior.

**Transfer learning** explicitly reuses source parameters, representations, normalization or skills to help target learning. The target can be a harder version of the same task or a new stage. Evidence requires better performance, fewer samples or lower cost relative to a scratch baseline—not merely a loaded file.

**Distribution shift** can involve initialization, size, mass, friction, camera view, sensing noise, latency, contact or robot morphology. It can affect observations, dynamics or the task itself.

Seeds are not domain parameters. Changing a seed cannot vary friction if the environment never samples it. In the current Place evaluation, seeds index a fixed reset sequence rather than random scenes. Conversely, manually changing mass or calibration shifts the domain even under the same seed.

Success across sampled seeds establishes stability for that sampling process, not necessarily robustness to unseen physics. Measure and describe these claims separately.

## 9.2 Pretraining, fine-tuning and parameter reuse

Checkpoints may include actor, critic, normalizer, optimizer and progress. Loading choices determine the operation:

| Operation | Relationship | Retain | Reset | Purpose |
|---|---|---|---|---|
| Resume | Same task and training semantics | Networks, normalizer, optimizer, counters | Temporary rollout state | Continue interrupted training |
| Transfer initialization | New Run or target conditions | Compatible networks/statistics | Optimizer and counters | Initialize target learning |
| Fine-tuning | Optimize on target data | Some or all transferred parameters | Experiment-dependent | Adapt to the target distribution |

Generic S1 resume checks observation, action, reward and PPO semantics before restoring complete training state. Equal tensor shapes do not imply equal meanings.

Grasp's `initialize_from_p2_r8` checks a locked parent SHA-256, task, schemas, reward, success, verifier and curriculum stage; only prescribed initial-XY ranges and seeds may differ. It copies actor, critic and normalizer, creates a new optimizer, resets counters and records parent identity. This is new-Run initialization, not resume.

The following teaching sketch isolates the distinction while omitting file I/O, hashes and full validation:

```python
def load_checkpoint_state(payload, mode, policy, normalizer,
                          optimizer, make_optimizer):
    # The real platform also validates observation, action and reward schemas here.
    policy.load_state_dict(payload["policy"])
    normalizer.load_state_dict(payload["normalizer_state"], frozen=False)

    if mode == "resume":
        optimizer.load_state_dict(payload["optimizer"])
        training_steps = payload["training_steps"]
        update_steps = payload["update_steps"]
    elif mode == "transfer":
        optimizer = make_optimizer(policy.parameters())
        training_steps = 0
        update_steps = 0
    else:
        raise ValueError("mode must be resume or transfer")

    return optimizer, training_steps, update_steps
```

Both branches load networks and normalization; only resume restores optimizer momentum and counters. Real compatibility checks are stricter, so this sketch must not replace the platform loader.

Normalization is part of the learned input interpretation. Fresh statistics can immediately change what unchanged weights see; source statistics can also be inappropriate after a semantic change. Field order, units, dimensions and scaling must be compatible.

Fine-tuning means optimization on target data. It may update all layers or freeze some, but the supplied Grasp/Lift implementations do not establish layer freezing. They create new optimizers and optimize transferred policies. Resetting selected output rows in Lower/Release or Place completion is not the same as freezing other layers.

**Curriculum learning** progressively changes initialization, goal ranges or difficulty. Grasp curriculum loading can preserve networks, statistics, optimizer and accumulated counters while changing reset stage. That continuous curriculum differs experimentally from transfer with a fresh optimizer and counters.

## 9.3 Task transfer and transfer between robot bodies

A transfer initialization can be written

\[
\theta_{\mathrm{target}}^{(0)}
=
\mathcal T\!\left(\theta_{\mathrm{source}}\right).
\]

where \(\mathcal T\) copies or maps compatible source parameters to target initialization. This relation alone provides no performance guarantee.

Grasp, Lift and Place have natural skill dependencies, but **skill order is not proof of parameter inheritance**. Current production Grasp initializes from a verified Grasp checkpoint; Lift from a verified Lift checkpoint. There is no evidence that Lift directly loads the Grasp network. Place transport creates a new trainer without a parent checkpoint.

Explicit stage transfer occurs from Place to Lower/Release: both share 32D observation and 4D action schemas. Source Place activates XY; the target activates Z descent and gripper control. Loading retains shared actor layers, critic, normalizer and learned XY rows, conservatively initializes newly active rows, and starts a fresh optimizer and counters.

Place completion also combines stages. Release-only uses rules to reach a legal release height while the learned policy triggers release; Full Place activates the full action space. This is staged coordination of rules and learning, not evidence of an end-to-end Grasp→Lift→Place weight chain.

Different robot bodies can change dimensions, joint order, link lengths, grippers, workspace, torque and control rate. Transfer may require body-independent representations, joint mappings, task-space interfaces or trained adapters.

The supplied implementation supports S1 with six arm joints, two gripper joints and differential IK converting Cartesian targets to joint-position targets. No second morphology or formal cross-body evaluation is supplied; cross-body transfer remains a general concept, not a demonstrated platform capability.

## 9.4 The simulation-to-reality gap

Sim-to-real applies a simulation-trained policy to hardware. Its reality gap has several sources:

| Source | Possible difference | Consequence |
|---|---|---|
| Geometry | Dimensions, mounting and joint zero offsets | Misaligned target/contact locations |
| Dynamics | Mass, inertia, friction and damping | Changed acceleration, slip and stability |
| Contact | Stiffness, collision, sliding and rebound | Changed grasp/push outcomes |
| Actuation | Delay, dead zones, saturation and tracking | Different displacement and feedback lag |
| Perception | Noise, occlusion, light and calibration | Shifted or missing inputs |
| Timing | Periods, latency and dropped frames | Different command duration and observation age |
| Safety | Speed, collision and emergency-stop rules | A stricter executable action set |

High simulation success does not guarantee hardware success. A policy may exploit simulator-specific contact or privileged information, while hardware filtering and low-level controllers alter closed-loop dynamics.

Current S1 live video is simulation evidence, not hardware or sim-to-real evidence. Deployment would require measurable compatible inputs, safe action mapping, calibration, speed/workspace bounds, emergency stops, slow single-step checks and staged acceptance.

Copying a checkpoint to another machine does not establish transfer. Ask how every input is measured, which controller executes actions, whether timing matches, and how independent sensing verifies success.

## 9.5 Domain randomization and system identification

Two complementary approaches broaden training conditions or calibrate simulation from real data. **Domain randomization** samples environment parameters:

\[
\xi\sim p(\xi),
\qquad
\max_\theta
\mathbb E_{\xi\sim p(\xi)}
\left[J(\pi_\theta;\xi)\right].
\]

Parameters \(\xi\) may include mass, friction, damping, initialization, visual noise or delay. The objective becomes performance over a specified family, not one fixed setting.

Choose ranges from physical knowledge and measurements. Too narrow may miss hardware; too wide may include unrealistic combinations. Record distributions, bounds, sampling frequency and correlations.

A changed seed is not automatically physical randomization. Current Grasp `randomize_xy` randomizes initialization. Configurable density, friction, restitution, stiffness and damping are fixed at scene creation in the supplied code; no per-episode physical-parameter randomization is demonstrated.

**System identification** estimates model parameters from real input–output data:

\[
\hat\xi
=
\arg\min_\xi
\sum_t
\left\|
y_t^{\mathrm{real}}
-y_t^{\mathrm{sim}}(\xi)
\right\|_2^2.
\]

The estimate \(\hat\xi\) minimizes discrepancy between measured and simulated outputs, for example joint responses under constrained test inputs. Poor excitation can make parameters unidentifiable; noise, unmodeled flexibility and timing errors also matter. Identification can inform randomization ranges without eliminating every model error.

The supplied S1 code does not collect hardware identification data or implement this optimization. It configures and records simulation properties; physical identification remains an extension.

## 9.6 Evaluating transfer

A transfer experiment needs a source, target and scratch baseline. Measure zero-shot performance, post-adaptation performance, steps to a threshold, matched-budget savings and any degradation on the source task.

A generalization gap can be defined as

\[
G=M_{\mathrm{train}}-M_{\mathrm{test}}.
\]

The sign depends on whether larger or smaller metric values are desirable. Do not mix success rates and distances or omit units and direction.

At matched target task, budget and protocol, transfer gain is

\[
\Delta_{\mathrm{transfer}}
=
M_{\mathrm{transfer}}
-M_{\mathrm{scratch}}.
\]

For success, positive is usually better; for error or required steps, negative may be better. Report original metrics, differences and uncertainty.

For illustrative—not measured—results, scratch achieves 62% and transfer 78% after 20,000 steps:

\[
\Delta_{\mathrm{transfer}}=0.78-0.62=0.16,
\]

This is 16 **percentage points**, not a 16% relative increase. Reaching 80% in 30,000 instead of 50,000 steps saves

\[
\frac{50,000-30,000}{50,000}=40\%.
\]

These measure different benefits. If transfer instead scores 55% against scratch's 62%, that is negative transfer of seven percentage points. Real claims require multiple training seeds and evaluation uncertainty.

Compare complete matched-budget learning curves, not only final best points. Transfer can accelerate early learning yet plateau later. **Negative transfer** can arise when source behavior inhibits target exploration, such as planar transport preferences during fine descent and release. Keep a scratch baseline to detect it.

| Group | Parameters | Normalizer | Optimizer | Comparison |
|---|---|---|---|---|
| Scratch | Random | New | New | Target baseline |
| Transfer | Source checkpoint | Compatible source state | New | Benefit of source initialization |
| Transfer without normalizer | Source checkpoint | New | New | Contribution of normalization |
| Resume control | Same-Run checkpoint | Restored | Restored | Distinguish continuation from transfer |

Use equal budgets and fixed evaluation seeds/success rules. Test unseen goals, objects and physics separately, with rates, errors and failure reasons. Goal heatmaps or parameter grids should identify training and extrapolation regions.

The supplied randomized Grasp summary has 98/100 verified successes; the fixed-tuned summary has 0/100 with all timeouts. Their model hashes and conditions differ, so this is not a controlled transfer comparison and cannot establish a benefit caused by randomization. No formal Lift or Place transfer summaries are supplied; their effects remain unconfirmed.

Transfer evidence must identify reused knowledge, interface compatibility, adaptation changes, scratch comparison and conditions under which benefits persist.

## References

1. Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. *IEEE Transactions on Knowledge and Data Engineering, 22*(10), 1345–1359. [DOI](https://doi.org/10.1109/TKDE.2009.191)
2. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems*. [DOI](https://doi.org/10.1109/IROS.2017.8202133)
3. Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. *2018 IEEE International Conference on Robotics and Automation*, 3803–3810. [DOI](https://doi.org/10.1109/ICRA.2018.8460528)
4. Muratore, F., Ramos, F., Turk, G., Yu, W., Gienger, M., & Peters, J. (2022). Robot Learning from Randomized Simulations: A Review. *Frontiers in Robotics and AI, 9*. [DOI](https://doi.org/10.3389/frobt.2022.799893)
5. Salvato, E., Fenu, G., Medvet, E., & Pellegrino, F. A. (2021). Crossing the Reality Gap: A Survey on Sim-to-Real Transferability of Robot Controllers in Reinforcement Learning. *IEEE Access, 9*, 153171–153187. [DOI](https://doi.org/10.1109/ACCESS.2021.3126658)
6. Ljung, L. (1999). *System Identification: Theory for the User* (2nd ed.). Prentice Hall. [Author page](https://www.rt.isy.liu.se/en/books/sysid/)
7. Lee, T., Kwon, J., Wensing, P. M., & Park, F. C. (2024). Robot Model Identification and Learning: A Modern Perspective. *Annual Review of Control, Robotics, and Autonomous Systems, 7*, 311–334. [DOI](https://doi.org/10.1146/annurev-control-061523-102310)
8. S1 platform source: `trainer.py`, `policy.py`, `normalizer.py`, `grasp_ppo.py`, `lift_ppo.py`, `place_ppo.py`, `lower_release_ppo.py`, `place_completion.py`, `cartesian_controller.py`, `s1_isaac_gym_adapter.py`, `scene.py` and `training_task_runner.py`; tests: corresponding Grasp curriculum, Grasp PPO, Lift PPO, Place PPO, Lower/Release PPOand manipulation scene tests; recorded artifacts: `grasp_randomized_100_summary.json` and `grasp_fixed_tuned_summary.json` (version reviewed for this chapter).

The first nine chapters establish concepts. The practical chapters now trace one data chain: task requirements become interfaces, interaction enters PPO, and a checkpoint receives independent evaluation. Each exercise begins with a prediction, a small function and CPU tests before discussing integration and training. Teaching code is not a drop-in implementation, teaching data are not Run evidence, and the browser does not execute arbitrary Python.

---

# Chapter 10 From a Task Requirement to a Trainable Program

“Make the S1 endpoint reach the goal” states intent, not a trainable program. We must define initialization, available information, allowed commands, scoring and completion. Follow one Reach transition through this reusable sequence:

> Requirement → inputs and outputs → examples → equation → function → tests → environment interaction → training → independent evaluation

## 10.1 Decomposing a task into five interfaces

| Interface | Question | Reach example |
|---|---|---|
| `reset` | How does an episode begin? | Restore arm, sample goal, clear action/holding history |
| Observation | What does the policy know? | Joint state, relative goal, tool velocity and previous action |
| Action | What can it do? | Normalized XY target increments |
| Reward | Which outcomes should be encouraged? | Approach progress, success and action/limit costs |
| `done` | Why does the episode end? | Success, failure or time limit |

Without action units, a network cannot infer whether 0.5 means meters, radians or half the allowed command.

| Defined by the programmer | Learned during training |
|---|---|
| Reset and initialization distribution | Actor/critic parameters |
| Observation fields, order and units | How observed features inform action/value |
| Action dimension, range and mapping | Action preferences for similar observations |
| Reward terms and weights | Which sampled actions become more likely |
| Success, failure and timeout rules | A policy operating under those fixed rules |
| Architecture and PPO configuration | Numerical network weights |

The policy is

\[
a_t\sim\pi_\theta(\cdot\mid o_t).
\]

PPO updates parameters from experience; the programmer does not supply the final correct-action function.

**Quick exercise: assign responsibility.** Which is learned: A. success-distance threshold; B. 2D action range; C. actor weights; D. timeout steps?

Reference answer: C. The others are task/training contracts. Changing them silently after training changes the problem and can invalidate a checkpoint.

## 10.2 How a Reach transition is generated

Use teaching values: distance changes from 0.080 to 0.076 m, normalized action is (0.5, −0.2), no success is detected, and joints are away from limits.

Reach's 18D observation contains six angles, six velocities, two relative goal components, two tool velocities and two previous actions. Its 2D output changes the endpoint target, not motor current. With scale 0.005 m per component:

\[
\Delta \mathbf x
=0.005(0.5,-0.2)
=(0.0025,-0.0010)\ \mathrm m.
\]

The persistent target changes by +2.5 mm in X and −1 mm in Y, subject to workspace clipping. Actual endpoint tracking must be measured, not assumed.

After execution, the environment observes again and evaluates reward. A minimal transition is

```python
transition = {
    "observation": observation,
    "action": (0.5, -0.2),
    "reward": 0.39942,
    "next_observation": next_observation,
    "done": False,
}
```

Observation explains the decision context; action records the choice; reward scores it; next observation captures its consequences; and `done` identifies an episode boundary. Removing a field loses part of this relationship.

**Prediction exercise.** Can `next_observation` always be removed when `done=True`?

Reference answer: No. Termination versus truncation can require a final observation/value for bootstrapping. The platform stores next value and separate masks. Retain information and let the trainer apply the correct semantics.

The minimal environment interface is

```python
observation = env.reset()
action = policy(observation)
next_observation, reward, done, info = env.step(action)
```

Reset returns initial information. Step executes, advances physics and returns observations, reward and ending information. Current adapters also report success, failure and timeout in `info`; always ask why an episode ended.

## 10.3 Writing a first reward function from a requirement

> Encourage approaching the goal, discourage moving away, and add a bonus on success.

Start with inputs: `previous_distance`, `distance`, and `success`. Current distance alone cannot distinguish progress from regression. Write a signature:

```python
def reach_reward(previous_distance, distance, success):
    ...
```

Then specify expectations:

| Change | Expected result |
|---|---|
| 0.10 → 0.09 m | Positive progress reward |
| 0.10 → 0.10 m | Zero progress reward |
| 0.10 → 0.11 m | Negative progress reward |

The simplest matching expression is

\[
r_t^{\text{progress}}=d_{t-1}-d_t.
\]

Both distances use meters. Translate the subtraction into Python and keep success as a separate term:

```python
def reach_reward(previous_distance, distance, success):
    progress = previous_distance - distance
    bonus = 1.0 if success else 0.0
    return progress + bonus
```

For 0.080 → 0.076 m without success, progress and total reward are 0.004. Preserve intended behavior with tests:

```python
def test_reach_reward_direction():
    assert reach_reward(0.10, 0.09, False) > 0
    assert reach_reward(0.10, 0.10, False) == 0
    assert reach_reward(0.10, 0.11, False) < 0
    assert reach_reward(0.02, 0.01, True) > 1.0
```

These test direction, no movement and success. Passing them establishes the pure function's behavior on these cases, not a trained policy.

The actual Reach formula additionally scales progress and includes costs:

\[
r_t=100(d_{t-1}-d_t)
+2\mathbf 1[\text{success}]
-0.002\lVert a_t\rVert_2^2
-0.01q_t^{\text{limit}}.
\]

Action is normalized, while the joint-limit proximity factor lies in [0,1]. Returning named terms makes the contributions inspectable:

```python
def reach_reward_terms(previous_distance, distance, action,
                       success, joint_limit_proximity):
    return {
        "progress_reward": 100.0 * (previous_distance - distance),
        "success_bonus": 2.0 if success else 0.0,
        "action_penalty": -0.002 * sum(x * x for x in action),
        "joint_limit_penalty": -0.01 * joint_limit_proximity,
    }
```

For this sample, progress is 0.4, action cost is −0.00058, the other terms are zero, and total reward is **0.39942**. This is a calculation from the formula, not a Run measurement.

**Completion exercise.** Reject negative input distances with `ValueError`. Why test this before training?

Reference answer: Check `previous_distance < 0 or distance < 0`. A pure-function test takes milliseconds; discovering the issue during GPU training costs more and confounds reward, environment and optimization errors.

The real implementation is `reach_adapter.py::ReachAdapter.compute_reward_terms`. Step supplies distances, action, success and joint state. Validate scoring independently before integration.

## 10.4 From transitions to rollouts

PPO collects a rollout under one policy version before updating:

```python
observation = env.reset()

for _ in range(num_steps):
    action, value, log_prob = policy.act(observation)
    next_observation, reward, done, info = env.step(action)
    rollout.add(observation, action, reward, value, log_prob, done)
    observation = env.reset() if done else next_observation

advantages, returns = rollout.compute_returns()
trainer.update(rollout)
save_checkpoint(policy)
```

This is a teaching skeleton, not a replacement trainer. It omits flattening/normalization, next values, timeouts, minibatches, callbacks and atomic saves.

Real buffers store `value` for predicted continuation and `log_prob` for old-policy density. `RolloutBuffer` also stores next values and bootstrap/continuation masks, then computes returns and GAE backward.

**Prediction exercise.** Can one rollout be optimized indefinitely until loss is tiny?

Reference answer: No. It was sampled under an older policy. Repeated changes make ratios and visitation mismatch grow. PPO permits a limited number of minibatch epochs but remains on-policy.

<details>
<summary>Advanced: termination masks in the current S1 rollout</summary>

The buffer uses

\[
\delta_t=r_{t+1}+\gamma b_tV(o_{t+1})-V(o_t),\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}
\]

True termination disables bootstrap; time limits may retain it. Every episode boundary disables continuation into the next episode. The actor uses normalized advantages and the critic raw returns. See `RolloutBuffer.compute_returns_and_advantages`.

</details>

## 10.5 What PPO actually updates

PPO compares an action's outcome with current expectation rather than writing a correct-action label into the network. Positive advantage increases likelihood; negative advantage decreases it:

\[
\nabla_\theta J(\theta)
\approx
\mathbb E[\nabla_\theta\log\pi_\theta(a_t\mid o_t)\hat A_t].
\]

Positive reward alone does not guarantee positive advantage: the critic may have expected more. The complete rollout, return and value provide context.

1. Reset to obtain an observation.
2. Sample an actor action.
3. Step the environment and record reward and ending reason.
4. Store interaction evidence.
5. Compute returns and advantages.
6. Update actor and critic in minibatches.
7. Save a checkpoint.
8. Freeze it and evaluate independently on prescribed seeds.

**Judgment exercise.** Does PPO learn to improve the reward function itself?

Reference answer: Not in this implementation. The programmer defines reward; PPO updates policy/value parameters and may become better at exploiting flaws in that reward.

<details>
<summary>Platform implementation: where the data chain lives</summary>

| Step | Source | Input → output |
|---|---|---|
| Observation | `ReachAdapter.get_observation` | Simulation state → five fields/18D semantics |
| Policy | `MLPActorCritic.act` | Normalized observation → action, log-probability and value |
| Action mapping | `ReachActionMapper.map` and `ReachAdapter.step` | Normalized XY → constrained target → next state |
| Reward/endings | `ReachAdapter.compute_reward_terms/step` | Distances, action, success, joints → terms, reward and info |
| Storage | `collect_rollout`, `RolloutBuffer.add` | Transition, values, densities and masks → ordered data |
| Estimation | `compute_returns_and_advantages` | Rollout → advantage and return |
| Optimization | `PPOTrainer.update` | Minibatches → updated parameters |
| Saving/evaluation | `save_checkpoint`, task runner, evaluation subprocess | Checkpoint + protocol → summary and result |

The documented trainer is single-process, single-environment. Vectorized environments are a common extension, not an established implementation fact here.

</details>

<details>
<summary>Advanced: continuous densities, global gradient clipping and checkpoints</summary>

- Gaussian samples pass through tanh and scaling; transformed `log_prob` needs the density correction.
- Separate actor/critic networks do not share direct value-loss updates, but joint gradient clipping can indirectly reduce actor step size.
- Checkpoints include optimizer, normalization, counts, configuration and semantic versions, not only weights.
- Generic `PPOConfig` defaults can differ from task-registry production settings. Use the final persisted Run configuration for reproduction.

</details>

## 10.6 Testing, training and evaluation are distinct gates

| Stage | What it establishes | What it does not establish |
|---|---|---|
| Reward unit tests | Signs, boundaries and calculations | Correct environment or learnability |
| Environment smoke test | Reset/step, basic shapes/endings | Long-term stability or effective training |
| Short training | Data/update chain does not immediately fail | Convergence or superiority |
| Full training | Checkpoint/artifacts under a budget | Reliable unseen-condition performance |
| Frozen formal evaluation | Performance under the fixed protocol | Hardware capability or broader generalization |

The authenticated `GET /simulation/task-catalog` is generated from the production registry. The frontend uses local fallback only for HTTP 404/405 and labels server versus fallback sources. Source code establishes this contract, but a fixture, fallback or HTTP 200 log is not a saved live response. The original teaching material does not include that response and therefore does not invent every field displayed by the runtime catalog.

`ppo_short_train.py` is a real controlled-development CLI for **simulated S1 Push** short PPO training. It requires rollout length, update count, project/run IDs and artifact root, using `cuda:0`. It is not a Reach-specific CLI or proof of the canonical web Run workflow. The accompanying README describes mock components in its end-to-end acceptance setup.

```bash
# Controlled-development example; adapt repository paths. This command was not run here.
PYTHONPATH=src python3 <path-to-ppo_short_train.py> \
  --rollout-steps 32 --num-updates 1 \
  --project-id tutorial --run-id push-practice-check \
  --artifacts-root ./artifacts --run-mode smoke
```

A run ID cannot change the environment created by this script. Reward extensions remain separate: Reach exposes only progress and action cost; success and joint-limit terms are protected; only allowlisted arithmetic is executable.

**Controlled-variable exercise.** When changing progress scaling from 100 to 50, what should stay fixed?

Reference answer: Observation/action schema, success rules, PPO/network configuration, budget, training/evaluation seed sets and other reward terms. Otherwise outcome changes cannot be attributed to scaling.

## Chapter 10 references and source index

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). [Author-hosted edition](http://incompleteideas.net/book/the-book-2nd.html)
2. Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
3. Gymnasium. [Env API](https://gymnasium.farama.org/api/env/)
4. S1 platform source: `reach_adapter.py`, `policy.py`, `buffer.py`, `trainer.py`, `normalizer.py`, `reward_extension.py`, `canonicalRunRequest.ts`, `http_api.py`, `task_catalog.py`, `simulationCatalog.ts`, `SimulationTaskConfigurator.tsx` and `ppo_short_train.py`.
5. S1 platform tests: `test_reach_reward.py`, `test_reach_evaluation.py`.

---

# Chapter 11 Reach: Your First Robot Learning Task

Six experiments turn the preceding workflow into practice. Begin with GPU-free predictions and tests before deciding whether to train. The aim is a reusable task-design method, not memorizing one reward function.

## Experiment 11-1 Understand the task before using a neural network

For endpoint (0.340, 0.030) m and goal (0.360, 0.020) m, the relative goal is

\[
\mathbf g-\mathbf p=(0.020,-0.010)\ \mathrm m,
\]

Distance is approximately 0.02236 m, outside the default 0.02 m success radius. Geometry already indicates increasing X and decreasing Y. Normalized action (0.5, −0.2) at scale 0.005 m changes the persistent target by (0.0025, −0.0010) m:

```python
def reach_target(target_xy, action, delta=0.005):
    proposed = tuple(
        value + delta * command
        for value, command in zip(target_xy, action)
    )
    return proposed
```

This teaching function returns an unclipped target. The platform additionally clips X to [0.30, 0.50] m and Y to [−0.13, 0.13] m. Starting at X=0.499 m with action component 1 gives 0.504 m before clipping and 0.50 m afterward.

**Prediction exercise.** Can a larger outward action move the clipped target beyond its workspace?

Reference answer: No. The action can be recorded as saturated while the target stays at the boundary. Inspect saturation, actual endpoint and reward together.

**Coding exercise.** Add `workspace_min` and `workspace_max` to `reach_target` and test that 0.499 + 0.005 clips to 0.50. This runs on ordinary CPU Python without starting a robot or simulation.

## Experiment 11-2 Understand the observation

The current Reach policy receives an 18D numerical vector, not camera images. Video is for human inspection.

| Field | Dimension | Units | Role |
|---|---:|---|---|
| `arm_q` | 6 | rad | Configuration and relationship to joint limits |
| `arm_dq` | 6 | rad/s | Motion trend |
| `ee_to_goal_xy` | 2 | m | Goal minus endpoint: direction and distance |
| `ee_velocity_xy` | 2 | m/s | Approaching or overshooting the goal |
| `previous_action_xy` | 2 | Dimensionless | Limited one-step action history |

The network receives ordered numbers, not field names:

```python
FIELDS = ("arm_q", "arm_dq", "ee_to_goal_xy",
          "ee_velocity_xy", "previous_action_xy")

def flatten_reach(observation):
    values = []
    for name in FIELDS:
        values.extend(float(x) for x in observation[name])
    assert len(values) == 18
    return values
```

Without relative goal coordinates, the same arm configuration can require different actions for different episode goals while looking identical to the policy. Joint state still matters: configurations, velocities, lag and limits affect feasible actions even at the same endpoint.

**Prediction exercise.** Is `ee_to_goal_xy` alone always sufficient?

Reference answer: It may be for an ideal planar point, but not necessarily an arm with configuration, velocity, lag and limits. Observation design should supply necessary information, not merely minimize length.

**Debugging exercise.** Swap `arm_q` and `arm_dq`. Length remains 18, but an old checkpoint may fail. Define a schema test for names, order, dimensions and units, not only total shape.

Reference approach: Use an immutable expected field/size tuple and compare it at loading. Unit changes also require a new schema or semantic version.

## Experiment 11-3 Write a reward pure function from scratch

Specify three properties: approaching is positive, staying still is zero, and moving away is negative. Complete:

```python
def progress_reward(previous_distance, distance):
    if previous_distance < 0 or distance < 0:
        raise ValueError("distance must be non-negative")
    return ________
```

Calculate these examples before viewing the answer:

```text
0.10 → 0.09: should be positive
0.10 → 0.10: should be zero
0.10 → 0.11: should be negative
```

Reference answer: `previous_distance - distance`, directly implementing \(r_t=d_{t-1}-d_t\).

```python
def test_progress_signs():
    assert progress_reward(0.10, 0.09) > 0
    assert progress_reward(0.10, 0.10) == 0
    assert progress_reward(0.10, 0.11) < 0
```

Next add success and action cost. For action (0.5, −0.2) and coefficient 0.002:

\[
r_t^{\text{action}}=-\lambda\lVert a_t\rVert_2^2
=-0.002(0.25+0.04)=-0.00058.
\]

Return named terms rather than only a total:

```python
def teaching_reach_reward(prev_d, d, action, success):
    parts = {
        "progress": 100.0 * (prev_d - d),
        "success": 2.0 if success else 0.0,
        "action": -0.002 * sum(x * x for x in action),
    }
    return sum(parts.values()), parts
```

**Small exercise.** Add `joint_limit_proximity` and `-0.01 * joint_limit_proximity`. Test that changing proximity from 0 to 1 lowers total reward by 0.01 without changing other terms.

Reference answer: Named terms make an isolated test possible; a total alone can hide an error through cancellation.

## Experiment 11-4 Compare five rewards on one trajectory

Use exactly this teaching distance sequence:

```text
0.100 → 0.080 → 0.060 → 0.061 → 0.040 m
```

Actions are (0.5, 0), (0.4, 0), (0.1, 0), and (0.5, 0). Set success threshold 0.05 m, exponential coefficient 20 m⁻¹ and action-cost coefficient 0.002. These are instructional values.

\[
\begin{aligned}
r_t^{\text{sparse}}&=\mathbf 1[d_t<\varepsilon],\\
r_t^{\text{distance}}&=-d_t,\\
r_t^{\text{progress}}&=d_{t-1}-d_t,\\
r_t^{\text{exp}}&=\exp(-\alpha d_t),\\
r_t^{\text{action}}&=-\lambda\lVert a_t\rVert_2^2.
\end{aligned}
\]

| Transition | Sparse | Negative distance | Progress | Exponential | Action penalty |
|---|---:|---:|---:|---:|---:|
| 0.100 → 0.080 | 0 | −0.080 | +0.020 | 0.20190 | −0.00050 |
| 0.080 → 0.060 | 0 | −0.060 | +0.020 | 0.30119 | −0.00032 |
| 0.060 → 0.061 | 0 | −0.061 | −0.001 | 0.29523 | −0.00002 |
| 0.061 → 0.040 | 1 | −0.040 | +0.021 | 0.44933 | −0.00050 |
| **Total** | **1** | **−0.241** | **+0.060** | **1.24765** | **−0.00134** |

On the third step, sparse reward remains zero, negative distance stays negative, progress explicitly reports regression, and exponential reward stays positive but decreases. Each encodes different preferences.

- Sparse success is clear but difficult to discover randomly.
- Negative distance supplies a signal everywhere, but proximity without completion may be acceptable without independent success/termination.
- Unmodified undiscounted progress telescopes to initial minus final distance; staying still scores zero and round trips cancel.
- Exponential reward is positive and bounded; its coefficient controls how concentrated the signal is near the goal. It is not a success rule.
- Action cost should accompany task rewards. Too much can favor inactivity.

Recompute the table with:

```python
from math import exp

def reward_variants(prev_d, d, action, eps=0.05, alpha=20.0):
    return {
        "sparse": float(d < eps),
        "negative_distance": -d,
        "progress": prev_d - d,
        "exponential": exp(-alpha * d),
        "action_penalty": -0.002 * sum(x * x for x in action),
    }
```

**Parameter exercise.** Change the exponential coefficient from 20 to 5. Does reward at 0.10 m increase or decrease?

Reference answer: It rises from \(e^{-2}\approx0.135\) to \(e^{-0.5}\approx0.607\). The signal reaches farther, but near/far discrimination changes too.

## Experiment 11-5 Let tests discover mistakes first

Check at least three failure modes before expensive training.

**Reversed sign:**

```python
def wrong_progress(previous_distance, distance):
    return distance - previous_distance
```

Direction tests fail immediately. Without them, PPO can diligently learn to move away.

**Mixed units:** passing 8.0 for eight centimeters instead of 0.08 scales progress by 100. Test that 80 mm → 76 mm converts to 0.080 → 0.076 m and unscaled progress 0.004 m.

**Repeated success bonuses:** rewarding every in-range step differs from detecting success after a hold. Current step logic counts in-range steps, succeeds at three and resets on leaving. Test the first two, third and reset cases:

```python
def update_hold(hold, distance, eps=0.02, required=3):
    hold = hold + 1 if distance <= eps else 0
    return hold, hold >= required

def test_hold_requires_three_steps():
    hold = 0
    for expected in (False, False, True):
        hold, success = update_hold(hold, 0.019)
        assert success is expected
```

**Find the bug.** Why does `reward != 0` fail to detect reversed signs?

Reference answer: Both positive and negative values pass. Test semantic order, such as `toward > still > away`, rather than mere numerical output.

The supplied `test_reach_reward.py` checks toward, away, still and success. Passing it establishes these formula contracts, not a trained success rate.

## Experiment 11-6 From short training to formal evaluation

Proceed in increasing cost order:

1. **Environment smoke:** check reset/steps, dimensions, action bounds, finite rewards and ending categories.
2. **Short training:** exercise rollouts, backpropagation, checkpoints and metric files. Do not claim learning from this alone.
3. **Full training:** freeze budget, seeds, PPO and task schemas; create traceable checkpoints.
4. **Independent evaluation:** freeze the checkpoint, use deterministic actions and 100 prescribed seeds, and report success, Wilson bounds, distance, episode length, timeout and failure.

| Change | Process metrics | Task metrics | Warning sign |
|---|---|---|---|
| More progress weight | Return and minimum distance | Success and final distance | Reward increases without stable success |
| More action cost | Action norm and saturation | Time to success and timeout | Smaller actions with many timeouts |
| Different success bonus | Bonus return | Success/holding conditions | Higher bonus without changed success |
| Different learning rate | Policy/value loss, KL and clipping | Multi-seed success | Smoother curve but worse task outcomes |

A supplied Reach summary records 100/100 successes, Wilson lower bound about 0.963, final distance about 0.0137 m, and no timeout/failure. It is evidence for that checkpoint, not a reference answer every student short run should match. Pretrained models can support evaluation, resume or adaptation; studying whether a reward learns from scratch still requires a scratch group.

### Practice at three levels

**Basic: change one permitted reward weight in the web editor.**

- Edit an allowed Reach progress or action-cost coefficient; actual fields follow the server catalog.
- State a directional hypothesis, such as doubling action cost reducing mean action norm.
- Use the experiment module's Run submission chain. The learning page itself does not train or execute arbitrary Python; historical README mock acceptance does not establish production integration.
- Record terms, norm, minimum/final distance, success and timeout; use independent evaluation for conclusions.
- Avoid changing two weights, budgets or selecting only a favorable seed.

**Code: add a reward-term test.**

- Work in an allowed development environment, not arbitrary browser execution.
- Check joint-limit cost monotonically at 0, 0.5 and 1.0 while preserving existing direction tests.
- Use the documented CPU entry point: `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
- The short-training CLI is for simulated Push, not a renamed Reach CLI.
- Keep term names/schema consistent and do not subtract an already negative penalty again.

**Experiment design: ablate progress scaling.**

- Hypothesis: lower scaling reduces dense feedback relative to success bonus.
- Change only `progress_scale`; fix task, PPO, budget and seeds.
- Record terms, error, success, Wilson bounds and failures.
- If differences do not exceed multi-seed variation, report insufficient evidence rather than selecting the best curve.

## Chapter 11 references and source index

1. Gymnasium. [Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)
2. OpenAI Spinning Up. [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
3. Ng, A. Y., Harada, D., & Russell, S. (1999). [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
4. S1 platform source: `reach_adapter.py::ReachAdapter.reset/get_observation/step/compute_reward_terms`, `policy.py`, `trainer.py` and `reward_extension.py`.
5. S1 platform evidence: `test_reach_reward.py`, `test_reach_evaluation.py`, `reach_evaluation_summary.json`.

---

# Chapter 12 Push: From Distance to Staged Rewards

Reach mainly controls endpoint–goal error. Push adds a contact chain: action moves the endpoint, suitable contact moves the object, and object motion approaches the goal. Replacing tool–goal distance with tool–object distance teaches approach, not pushing. Build rewards incrementally, testing what each addition fixes and what it can exploit.

## 12.1 Understand Push inputs and outputs

The current Push observation has 24 components:

| Field | Dimension | Units | Information |
|---|---:|---|---|
| `arm_q`, `arm_dq` | 6 + 6 | rad, rad/s | Configuration and motion |
| `ee_to_object_xy` | 2 | m | Tool–object relation |
| `object_to_goal_xy` | 2 | m | Desired object motion |
| `ee_velocity_xy` | 2 | m/s | Endpoint movement |
| `object_velocity_xy` | 2 | m/s | Whether the object is moving |
| `target_tracking_error_xy` | 2 | m | Low-level target tracking |
| `previous_action_xy` | 2 | Dimensionless | Previous action |

Contact booleans support reward and diagnostics but are not in the 24D policy vector. Contact must be inferred indirectly. Viewer video is not a PPO input.

Push outputs physical XY increments bounded by ±0.005 m, while its previous-action observation is normalized by 0.005. Reach instead receives normalized actions before mapping. Equal two-dimensional shapes do not make checkpoints compatible.

**Prediction exercise.** Does `object_to_goal_xy` alone reveal which side of the object the endpoint occupies?

Reference answer: No. It describes where the object should go, not how to establish contact. Both robot–object and object–goal relations matter.

## 12.2 Version A: reward only endpoint approach

A first attempt rewards reduced tool–object distance:

\[
r_t^A=d_{t-1}^{ee,obj}-d_t^{ee,obj}.
\]

```python
def reward_a(previous_ee_object, ee_object):
    return previous_ee_object - ee_object
```

Distance 0.080 → 0.060 m gives +0.020 even if object–goal distance stays 0.200 m and the object does not move. This correctly rewards approach but imposes no pushing requirement. With action cost, stopping near the object can be attractive.

Expose the gap explicitly:

```python
def test_version_a_cannot_see_object_motion():
    reward = reward_a(0.080, 0.060)
    assert reward > 0
    object_displacement = 0.0
    assert object_displacement == 0.0
```

The function satisfies its own contract, but that contract omits the task outcome.

**Modification exercise.** Compare approaching a stationary object with unchanged tool distance while the object moves toward the goal. Which does A prefer?

Reference answer: It favors the first or fails to reward the second because it sees only tool distance. Object motion is needed.

## 12.3 Version B: reward only object progress

Use object–goal distance instead:

\[
r_t^B=d_{t-1}^{obj,goal}-d_t^{obj,goal}.
\]

```python
def reward_b(previous_object_goal, object_goal):
    return previous_object_goal - object_goal
```

Changing 0.200 → 0.198 m scores +0.002; moving to 0.202 scores −0.002. This better reflects the final objective.

Before contact, however, random actions often leave the object still and reward repeatedly zero. The policy gets no indication that approaching is promising. The objective can be correct yet too sparse:

```python
def test_version_b_is_zero_before_contact():
    rewards = [reward_b(0.200, 0.200) for _ in range(5)]
    assert rewards == [0.0] * 5
```

**Prediction exercise.** Does a huge success bonus automatically solve missing precontact guidance?

Reference answer: No guarantee. If exploration never completes approach–contact–push, the bonus is never observed. Intermediate guidance must help without replacing success criteria.

## 12.4 Version C, step 1: approach from a useful pushing position

Approaching the object's center from the goal side may push it away. The platform defines a precontact position behind the object along goal direction \(\mathbf u\). For precontact error \(e_t\):

\[
p_t=e_{t-1}-e_t.
\]

A geometric gate in [0,1] reduces rewards from wrong-side or misaligned approach:

```python
def precontact_terms(previous_error, error, geometry_gate):
    progress = previous_error - error
    positive = max(progress, 0.0)
    negative = min(progress, 0.0)
    return {
        "precontact": 2.0 * (positive * geometry_gate + negative),
        "wrong_side": -1.0 * positive * (1.0 - geometry_gate),
    }
```

For error 0.050 → 0.047 m and gate 0.8, precontact reward is 0.0048 and wrong-side cost −0.0006, totaling 0.0042. With gate zero, the same approach totals −0.003 instead.

Test ordering:

```python
def test_aligned_approach_beats_wrong_side():
    aligned = sum(precontact_terms(0.050, 0.047, 1.0).values())
    wrong = sum(precontact_terms(0.050, 0.047, 0.0).values())
    assert aligned > 0 >= wrong
```

Overly strict gates can remove nearly all useful early feedback. The real implementation also fades positive progress near precontact to discourage repeated reward farming; moving away retains negative progress.

**Completion exercise.** Validate the gate's [0,1] bounds and calculate the gate-0.5 result before training. Boundary tests separate contract bugs from learning behavior.

## 12.5 Version C, step 2: directional pushing and lateral costs

Decompose object displacement into goal-aligned and perpendicular components:

\[
\Delta o_t^{\parallel}=\Delta\mathbf o_t^\top\mathbf u,
\qquad
\Delta o_t^{\perp}=\Delta\mathbf o_t^\top\mathbf u_\perp.
\]

The platform gives signed directional reward and penalizes absolute lateral displacement:

```python
def push_motion_terms(parallel_motion, lateral_motion):
    return {
        "directional": 20.0 * parallel_motion,
        "lateral": -5.0 * abs(lateral_motion),
    }
```

Moving 0.0015 m toward the goal with 0.0002 m lateral drift gives +0.030 and −0.001, totaling +0.029. Reversing the motion gives −0.030 directionally. Moving a lot is not the same as moving correctly.

```python
def test_motion_order():
    forward = sum(push_motion_terms(0.001, 0.0).values())
    lateral = sum(push_motion_terms(0.0, 0.001).values())
    backward = sum(push_motion_terms(-0.001, 0.0).values())
    assert forward > lateral > backward
```

A brief goal-directed displacement can still be followed by drift or result from violent contact. Keep success, failure, action cost and episode-level metrics.

**Numerical exercise.** Increase lateral coefficient from 5 to 20.

Reference answer: The same sample totals \(0.030-20\times0.0002=0.026\). Stronger penalties can also suppress useful contact adjustments, so higher is not automatically better.

## 12.6 Version C, step 3: contact, success and constraints

Contact helps sustain interaction but is not success. The default contact bonus is 0.002 per contact step, smaller than 1 mm of effective directional pushing: 0.020. Success adds 1.0 once. Action cost uses the mean squared action after normalization by 0.005 m; joint-limit cost remains protected.

```python
def completion_terms(contact, success_once,
                     normalized_action_cost, joint_proximity):
    return {
        "contact": 0.002 if contact else 0.0,
        "success": 1.0 if success_once else 0.0,
        "action": -0.001 * normalized_action_cost,
        "joint_limit": -0.01 * joint_proximity,
    }
```

For physical action (0.0025, −0.0010) m, normalized action is (0.5, −0.2), with squared mean 0.145 and action cost −0.000145. With contact, no success and zero joint proximity, the combined teaching result is

\[
0.0042+0.029+0.002-0.000145=0.035055.
\]

Entering the success radius without failure adds the one-time bonus. Object falling or leaving the table causes explicit failure; exhausting the episode limit without success/failure is timeout.

**Debugging exercise.** What happens with `total -= action_penalty` when the penalty is already negative?

Reference answer: It becomes a positive incentive for large actions. Test names, signs and combination together.

<details>
<summary>The eight reward terms in the actual Push implementation</summary>

1. `precontact_progress`: geometry-gated approach, fading positive progress near precontact while retaining negative retreat.
2. `wrong_side_or_alignment_term`: negative contribution for positive approach from unsuitable geometry.
3. `directional_push_progress`: signed object motion along the preceding goal direction, default scale 20.
4. `lateral_push_penalty`: absolute lateral motion, default scale −5.
5. `contact_sustain_bonus`: 0.002 during dynamic contact.
6. `success_bonus`: 1.0 on first success.
7. `action_penalty`: normalized 2D mean square ×−0.001.
8. `joint_limit_penalty`: joint-limit proximity ×−0.01.

Geometry, contact detection and success latching require environment state and do not belong in an isolated beginner function. Existing tests cover proper-side ordering, forward versus lateral/backward movement, modest contact bonus, one-time success, falling and boundary failures.

</details>

## 12.7 From reward terms to trustworthy experiments

Before asking which weights are best, identify evidence left by each failed design:

| Symptom | Process evidence | Task evidence | Unsupported shortcut |
|---|---|---|---|
| Stops near the object | Precontact error, contact rate, action norm | Object displacement and success | “PPO is broken” |
| Rarely contacts | Minimum precontact error and near-contact steps | Timeout rate | “The success bonus is too small” |
| Moves object the wrong way | Directional/lateral displacement | Final/minimum distance | “Any movement means learning” |
| Reward rises, success does not | Individual terms and action magnitude | Success, timeout, failure | “Training succeeded” |
| Frequent falls/boundary exits | Reasons and saturation | Failure rate | “Increase the success bonus” |

The web editor permits constrained combinations of six Push terms: approach, wrong-side/alignment, directional push, lateral penalty, contact and action cost. Success and joint-limit terms remain protected. Adding terms or changing contact/success logic requires controlled development, schema changes and tests—not arbitrary browser Python.

### A single-variable experiment

Hypothesis: doubling the directional-push combination weight increases physical displacement toward the goal.

| Item | Design |
|---|---|
| Change | `directional_push_progress` weight 1 → 2 |
| Fix | All other interfaces, rewards, PPO settings, budgets, seeds and success rules |
| Process outcomes | Directional/lateral motion, contact, action norm and reward terms |
| Task outcomes | Success, Wilson bounds, distances, timeout/failure |
| Falsification | Higher weighted reward without more physical progress, or more progress without better success |

**Design exercise.** Why not declare improvement from higher cumulative reward?

Reference answer: The changed weight changes the scoring scale even for identical physical trajectories. Compare shared physical outcomes under the same protocol.

### Examine real evidence before explaining a failure

The supplied `push_evaluation_summary.json` records 0/100 successes, timeout rate 1.0, failure rate 0, displacement about \(2.31\times10^{-6}\) m and mean return about 0.1695. It establishes failure to pass that evaluation, not which reward term caused it.

A source correction matters: the accompanying 418-line `episodes.jsonl` must not be called Push training evidence. Its accompanying manifest identifies `s1_reach`, Run `phase-m2-s1-reach-20k-seed7-v2`. More strictly, that manifest registers `metrics/training.jsonl` at 116,050 bytes with SHA-256 `44f6…490b1`; the episode file is 313,892 bytes with hash `0b8a…218`. The exact file is therefore not registered. It cannot serve as stepwise Push evidence or be merged into Reach evaluation without a completed provenance chain.

<details>
<summary>Advanced: verifying evidence from the same Run</summary>

| Check | Supplied evidence | Consequence |
|---|---|---|
| Each file has a matching path, size and SHA-256 | Missing reward-step/catalog entries; episode file differs from registered training file | Similar names or one upload batch do not establish provenance |
| Run/task/environment/config/checkpoint identities agree | Reach manifest and Push summary do not share a verified identity; 656 bytes/`26f2…40e7` differs from the manifest's 1,137 bytes/`5e81…61e6` | These are not one verified evidence set |
| Per-step reward records contain necessary fields | Episode aggregates lack step index, actions, per-step terms and endings | They support episode trends, not reconstruction of individual reward effects |
| Catalog is a real authenticated response | Route and fallback exist in source, but no response body is supplied | Fixtures, fallback and HTTP status logs cannot substitute for the response |

A useful `reward_steps.jsonl` includes episode/step indices, total reward, complete terms, action, distances and ending flags/reasons. The manifest must register it and the actual catalog response, with matching configuration and checkpoint hashes throughout. Only then can reward changes be connected auditably to results from the same model.

</details>

### Practice at three levels

**Basic: change one constrained reward weight.**

- Use terms allowed by the actual catalog/editor; state a mechanism-based hypothesis.
- Compare baseline and variant with matched budgets and seeds through formal Run submission.
- Record terms, contact, directional displacement, distance and all ending categories.
- Require physical and task evidence, not one best reward.
- Do not change multiple coefficients or success rules, or fill missing metrics with zero.

**Code: add a backward-direction test.**

- Work in a pure function or `test_push_adapter.py` within the allowed development environment.
- Check `forward > lateral > backward` with displacement in meters.
- Run CPU tests with a fake backend before GPU simulation.
- Preserve consistent reference direction and use incremental motion, not absolute object coordinates.

**Experiment: ablate precontact progress.**

- Hypothesis: removing it makes effective contact harder within a fixed budget.
- Set only that combination weight to zero; fix every other condition.
- Record precontact error, near-contact steps, contact rate, directional movement, success and timeout.
- Establish the approach/contact mechanism first, then final task effects. Report insufficient evidence when differences fall within multi-seed variation.
- Do not resume from incompatible old-reward checkpoints and thereby confound initialization.

For a new task, repeat the method: define the physical outcome, locate missing exploration signals, add one interpretable term at a time, test signs and bounds, inspect process mechanisms and evaluate task capability independently.

## Chapter 12 references and source index

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). [Author-hosted edition](http://incompleteideas.net/book/the-book-2nd.html)
2. Ng, A. Y., Harada, D., & Russell, S. (1999). [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
3. Gymnasium. [Env API](https://gymnasium.farama.org/api/env/)
4. S1 platform source: `push_adapter.py::PushAdapter.get_observation/step/compute_reward_terms/reward_geometry`, `reward_extension.py`, `rewardTeaching.ts`, `canonicalRunRequest.ts` and `SimulationTaskConfigurator.tsx`.
5. S1 platform tests: `test_push_adapter.py`.
6. S1 artifacts: `push_evaluation_summary.json`, `summary.json`, `episodes.jsonl`, `training.jsonl` and `manifest.json`. See section 12.7: the new manifest identifies Reach, does not register the exact 418-line file, and has no verified hash link to the Push summary.
7. Interface and engineering sources: `http_api.py`, `task_catalog.py`, `task_registry.py`, `simulationCatalog.ts`, `SimulationTaskConfigurator.tsx`, `ppo_short_train.py` and `README.md`. They establish source contracts, not a recorded response or artifacts from a particular Run.
