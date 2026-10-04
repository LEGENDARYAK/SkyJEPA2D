# Planar dynamics and conventions

The side view has world coordinates x (right) and z (up), in meters. Positive
pitch theta rotates counterclockwise in this view: the right motor rises and
the thrust axis tilts left. The body-axis unit thrust vector is
[-sin(theta), cos(theta)]. Pitch rate omega is in radians/second.

Observed state is [x, z, vx, vz, theta, omega]. Actual left/right motor forces
are two additional hidden actuator states. Actions are requested left/right
motor-group forces, in newtons, held constant for one control interval. They
represent a planar reduction of four individual quadrotor motors.

With total actual force F = fL + fR and arm length l:

$$\dot{x}=v_x,\qquad\dot{z}=v_z$$

$$m\dot{v}_x=-F\sin\theta-c_v(v_x-w_x)$$

$$m\dot{v}_z=F\cos\theta-mg-c_v(v_z-w_z)$$

$$\dot{\theta}=\omega,\qquad J\dot{\omega}=l(f_R-f_L)-c_\omega\omega$$

$$\dot{f}_i=(k_i a_i-f_i)/\tau$$

Here a is the clipped command, k is the actuator gain, tau is motor response
time, and w is wind velocity. When tau is zero, actual forces equal scaled
commands immediately. At zero pitch and zero wind, hover requires fL = fR = mg/2.

The simulator integrates the eight variables with fixed-step fourth-order
Runge-Kutta. Angles wrap to [-pi, pi); angular velocity remains continuous.
Rendering and input handling never write the physical state. The plant stores
float64; recorded training trajectories use float32, with float64 timestamps.

The collision radius is a deliberately simple center-position margin. Contacts
with the ground or flight-area boundaries terminate the episode. There is no
bounce, landing dynamics, obstacle geometry, propeller aerodynamic model,
battery model, or real-drone calibration.

The conventional controller maps position error to a bounded desired velocity,
then velocity error to desired acceleration. Desired lateral acceleration sets
pitch, and an attitude/rate PD loop allocates torque. The manual controller uses
the same type of attitude stabilization, but the human directly requests lean
and collective thrust. It has no position-hold loop.

The learned prober uses a separate differentiable midpoint-style integrator for
its nominal dynamics. Agreement with the RK4 plant is approximate, especially
during fast pitch changes. This mismatch and randomized actuator properties are
part of what the residual network must learn; they are measured in evaluation.

