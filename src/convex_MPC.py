import numpy as np
import quadprog
import mpc_osqp as convex_mpc
from scipy.integrate._ivp import base

class convex_MPC:
  """
  Convex Model Predictive Control Class for legged robots
  """

  def __init__(self):
    

    """
    self.horizon = 10
    self.A_qp = np.zeros( (13*self.horizon, 1), dtype=np.float64 )
    self.B_qp = np.zeros( (13*self.horizon, 12*self.horizon), dtype=np.float64 )
    self.S    = np.zeros( (13*self.horizon, 13*self.horizon), dtype=np.float64 )
    self.X_d  = np.zeros( (13*self.horizon, 1), dtype=np.float64 )
    self.U_b  = np.zeros( (20*self.horizon, 12*self.horizon), dtype=np.float64 )
    self.fmat = np.zeros( (20*self.horizon, 12*self.horizon), dtype=np.float64 )
    self.qH   = np.zeros( (12*self.horizon, 12*self.horizon), dtype=np.float64 )
    self.qg   = np.zeros( (12*self.horizon, 1), dtype=np.float64 )
    self.eye_12h   = np.eye( 12*self.horizon, dtype=np.float64 )

    self.H_qprog  = np.array( (12*12*self.horizon*self.horizon), dtype=np.float64 )
    self.g_qprog  = np.array( (12*1*self.horizon), dtype=np.float64 )
    self.A_qprog  = np.array( (12*20*self.horizon*self.horizon), dtype=np.float64 )
    self.lb_qprog = np.array( (20*1*self.horizon), dtype=np.float64 )
    self.ub_qprog = np.array( (20*1*self.horizon), dtype=np.float64 )
    self.q_soln   = np.array( (12*self.horizon), dtype=np.float64 )
    """

    self._desired_speed = (0, 0)
    self._desired_twisting_speed = 0
    self._FORCE_DIMENSION = 3
    self._PLANNING_HORIZON_STEPS = 10
    self._PLANNING_TIMESTEP = 0.025
    # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
    self._MPC_WEIGHTS = [5, 5, 0.2, 0, 0, 10, 0., 0., 1., 1., 1., 0., 0]
    self._qp_solver = convex_mpc.QPOASES

    self._desired_body_height = 0.29
    self._body_mass = 11.0
    self._num_legs = 4
    self._friction_coeffs = (0.45, 0.45, 0.45, 0.45)
    self._body_inertia_list = [0.05, 0, 0, 0, 0.1, 0, 0, 0, 0.1]
    self._cpp_mpc = convex_mpc.ConvexMpc(
        self._body_mass,
        self._body_inertia_list,
        self._num_legs,
        self._PLANNING_HORIZON_STEPS,
        self._PLANNING_TIMESTEP,
        self._MPC_WEIGHTS,
        1e-5,
        self._qp_solver        
    )

  def update_desired_vars(self, desired_height, desired_vel, desired_ang_vel, foot_contacts):
    self._desired_body_height = desired_height
    self._desired_speed = desired_vel
    self._desired_twisting_speed = desired_ang_vel
    self._foot_contacts = np.array(foot_contacts).astype(int)

  def update_state_vars(self, base_rpy, base_rpy_rate, com_velocity_body_frame, foot_pos_in_base_frame, jacobians ):
    self._base_rpy = base_rpy
    self._base_rpy_rate = base_rpy_rate
    self._com_velocity_body_frame = com_velocity_body_frame 
    self._foot_pos_in_base_frame = foot_pos_in_base_frame
    self.jacobians = jacobians
    
  def calc_torques(self):
    """Computes the torque for stance legs."""
    desired_com_position = np.array( (0., 0., self._desired_body_height), dtype=np.float64)
    desired_com_velocity = np.array( (self._desired_speed[0], self._desired_speed[1], 0.), dtype=np.float64)
    desired_com_roll_pitch_yaw = np.array((0., 0., 0.), dtype=np.float64)
    desired_com_angular_velocity = np.array( (0., 0., self._desired_twisting_speed), dtype=np.float64)
    foot_contact_state = self._foot_contacts

    # We use the body yaw aligned world frame for MPC computation.
    com_roll_pitch_yaw = np.array(self._base_rpy,dtype=np.float64)
    com_roll_pitch_yaw[2] = 0

    #predicted_contact_forces=[0]*self._num_legs*self._FORCE_DIMENSION
    # print("Com Vel: {}".format(self._state_estimator.com_velocity_body_frame))
    # print("Com RPY: {}".format(self._robot.GetBaseRollPitchYawRate()))
    # print("Com RPY Rate: {}".format(self._robot.GetBaseRollPitchYawRate()))
    predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
        [0],  #com_position
        np.asarray(self._com_velocity_body_frame, dtype=np.float64),  #com_velocity
        np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
        # Angular velocity in the yaw aligned world frame is actually different
        # from rpy rate. We use it here as a simple approximation.
        np.asarray(self._base_rpy_rate, dtype=np.float64),  #com_angular_velocity
        foot_contact_state,  #foot_contact_states
        np.array(self._foot_pos_in_base_frame, dtype=np.float64),  #foot_positions_base_frame
        self._friction_coeffs,  #foot_friction_coeffs
        desired_com_position,  #desired_com_position
        desired_com_velocity,  #desired_com_velocity
        desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
        desired_com_angular_velocity  #desired_com_angular_velocity
    )
    # sol = np.array(predicted_contact_forces).reshape((-1, 12))
    # x_dim = np.array([0, 3, 6, 9])
    # y_dim = x_dim + 1
    # z_dim = y_dim + 1
    # print("Y_forces: {}".format(sol[:, y_dim]))

    contact_forces = {}
    for i in range(self._num_legs):
      contact_forces[i] = np.array(
          predicted_contact_forces[i * self._FORCE_DIMENSION :(i + 1) *
                                   self._FORCE_DIMENSION ])
    action = []
    for leg_id, force in contact_forces.items():
      # While "Lose Contact" is useful in simulation, in real environment it's
      # susceptible to sensor noise. Disabling for now.
      # if self._gait_generator.leg_state[
      #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
      #   force = (0, 0, 0)
      motor_torques = self.MapContactForceToJointTorques(leg_id, force)
      for joint_id, torque in motor_torques.items():
        action.append(torque)

    return action, contact_forces

    
  def MapContactForceToJointTorques(self, leg_id, contact_force):
    """Maps the foot contact force to the leg joint torques."""
    jv = self.jacobians[leg_id]
    all_motor_torques = np.matmul(contact_force, jv)
    motor_torques = {}
    com_dof = 6
    for joint_id in range(leg_id * 3,(leg_id + 1) * 3):
      motor_torques[joint_id] = all_motor_torques[com_dof + joint_id]

    return motor_torques