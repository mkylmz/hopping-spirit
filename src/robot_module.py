import pybullet as p
import math
import numpy as np
from convex_MPC import convex_MPC
from slip2d import slip2d
import pathlib
import time
from utils import quat2euler
from convex_MPC import convex_MPC

class robot_module:
    """
    robot class
    """

    def __init__(self, init_pos, dt):

        self.planeid   = p.loadURDF("plane.urdf")

        urdf_path      = str(pathlib.Path(__file__).parent.absolute()) + "/myspirit40.urdf"
        self.robotid   = p.loadURDF(urdf_path, init_pos)
        self.reset_pos = init_pos
        self.reset_ori = [0,0,0,1]
        self.pos       = init_pos
        self.mass      = 11.00
        self.dt        = dt

        #Hardcoded urdf parameters like leg lengths etc.. Could be defined in the robot class later
        self.L1 = 0.206
        self.L2 = 0.206

        self.Kp = 40
        self.Kd = 2
        self.desired_pos = [0,0,0.30]
        self.desired_vel = [0,0,0]
        self.desired_acc = [0,0,0]
        self.friction_coeff = 0.60/math.sqrt(2)

        #Controller parameters
        self.rest_length = 0.2913 # Desired virtual leg length
        self.aoa = 0 # Desired virtual leg attack angle

        self.LEGS = ["FL","RL","FR","RR"]
        self.JOINTS = ["HIP","UPPER","LOWER","TOE"]
        self.indices = {
            self.LEGS[0] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 1,
                self.JOINTS[2] : 2,
                self.JOINTS[3] : 3
            },
            self.LEGS[1] : {
                self.JOINTS[0] : 4,
                self.JOINTS[1] : 5,
                self.JOINTS[2] : 6,
                self.JOINTS[3] : 7
            },
            self.LEGS[2] : {
                self.JOINTS[0] : 8,
                self.JOINTS[1] : 9,
                self.JOINTS[2] : 10,
                self.JOINTS[3] : 11
            },
            self.LEGS[3] : {
                self.JOINTS[0] : 12,
                self.JOINTS[1] : 13,
                self.JOINTS[2] : 14,
                self.JOINTS[3] : 15
            }
        }

        self.actuated_joint_indices = [0,1,2,4,5,6,8,9,10,12,13,14]
        self.toe_indices = [3,7,11,15]
        
        self.INITIAL_JOINT_POSITIONS = [0, math.pi/4, math.pi/2,
                                        0, math.pi/4, math.pi/2,
                                        0, math.pi/4, math.pi/2,
                                        0, math.pi/4, math.pi/2]
        self.reaction_forces = {
            self.LEGS[0] : [0,0,0,0,0,0],
            self.LEGS[1] : [0,0,0,0,0,0],
            self.LEGS[2] : [0,0,0,0,0,0],
            self.LEGS[3] : [0,0,0,0,0,0]
        }

        self.SelMat = np.concatenate( (np.zeros((12,6),np.float64), np.eye(12)), axis=1 )

        self.inContact = [False,False,False,False]

        # Variable matrices for equations of motion
        self.MassMatrix = []
        self.NMatrix = []
        self.JacT = [[],[],[],[]]
        self.JacR = [[],[],[],[]]
        self.JacTDot = [[],[],[],[]]
        self.SMatrix = [[0 for i in range(18)]]*6
        for i in range(6):
            for j in range(12):
                self.SMatrix[i][j]

        
        self.q              = np.array([0.0 for i in range(19)])
        self.qdot           = np.array([0.0 for i in range(18)])
        self.qdotdot        = np.array([0.0 for i in range(18)])
        self.old_qdot       = np.array([0.0 for i in range(18)])
        self.old_qdotdot    = np.array([0.0 for i in range(18)])
        self.target_torques = np.array([0.0 for i in range(12)])
        self.force_sel_mat  = np.array([0.0 for i in range(12)])
        self.target_acc     = np.array([0.0 for i in range(15)])

        self.q_acc = [.0 for i in range(12)] ## Fake acc matrix for pybullet

        for leg_i, leg_name in enumerate(self.LEGS):
            for joint_i, joint_name in enumerate(self.JOINTS[:-1]):
                p.resetJointState(self.robotid, self.indices[leg_name][joint_name], self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i])
                p.setJointMotorControl2(self.robotid, self.indices[leg_name][joint_name], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i], force=0)
            p.enableJointForceTorqueSensor(self.robotid, self.toe_indices[leg_i], True)
        

        for leg_i, leg in enumerate(self.LEGS):
            jt, jr = p.calculateJacobian(self.robotid, self.indices[self.LEGS[leg_i]]["TOE"], [0,0,0],  list(self.q[7:19]), list(self.qdot[6:18]), self.q_acc)
            self.JacTDot[leg_i] = np.array(jt)/self.dt
            self.JacT[leg_i] = np.array(jt)
            self.JacR[leg_i] = np.array(jr)

        self.myslip = slip2d([init_pos[0], init_pos[2], 0, 0, 0, 0], self.aoa, self.rest_length, self.dt)
        self.apex = True
        self.slipsolution = []

        self.convMPC = convex_MPC(10)

    def control(self):
        
        self.fetchState()
        self.solve_slip()
        self.track_tracjectory()

        for leg_i, leg in enumerate(self.LEGS):
            if ( self.checkContact(leg) ):
                self.handleStance(leg_i)
            else:
                self.handleFlight(leg_i)  
        

    def checkContact(self, leg):
        contact = p.getContactPoints(bodyA=self.robotid, bodyB=self.planeid, linkIndexA=self.indices[leg]["TOE"])
        if (len(contact)):
            return True
        return False

    def fetchState(self):
        ## Save the old state variables
        self.old_qdot = self.qdot
        self.old_qdotdot = self.qdotdot

        ## Get pos and vel from the simulation
        states = p.getJointStates(self.robotid, range(p.getNumJoints(self.robotid)))
        pos = [ele[0] for ele in states]
        vel = [ele[1] for ele in states]
        rf  = [ele[2] for ele in states]
        
        for i in range(16):
            leg_no = int(i/4)
            if not i%4==3: 
                self.q[7+i-leg_no]    = pos[i] 
                self.qdot[6+i-leg_no] = vel[i]
            else:
                self.reaction_forces[leg_no] = rf[leg_no*4+3]
        
        body_pos = p.getBasePositionAndOrientation(self.robotid)
        body_vel = p.getBaseVelocity(self.robotid)
        self.q[0:7]    = np.array( body_pos[0] + body_pos[1] )
        self.qdot[0:6] = np.array( body_vel[0] + body_vel[1] )
        self.qdotdot = (self.qdot - self.old_qdot)/self.dt

        # Calculate matrices for equations of motion
        self.MassMatrix = p.calculateMassMatrix(self.robotid, list(self.q[7:19]))
        self.NMatrix = list(p.calculateInverseDynamics(self.robotid, list(self.q[7:19]), list(self.qdot[6:18]), self.q_acc, flags=1))
        del self.NMatrix[6]
        for leg_i, leg in enumerate(self.LEGS):
            jt, jr = p.calculateJacobian(self.robotid, self.indices[self.LEGS[leg_i]]["TOE"], [0,0,0],  list(self.q[7:19]), list(self.qdot[6:18]), self.q_acc)
            self.JacTDot[leg_i] = (np.array(jt) - self.JacT[leg_i])/self.dt
            self.JacT[leg_i] = np.array(jt)
            self.JacR[leg_i] = np.array(jr)

    def handleStance(self,leg_index):
        
        if not self.inContact[leg_index]:
            p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["HIP"], p.POSITION_CONTROL, 0, force=0)
            p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], p.POSITION_CONTROL, 0, force=0)
            p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["LOWER"], p.POSITION_CONTROL, 0, force=0)            
            self.inContact[leg_index] = True

        min_i = leg_index*3
        max_i = leg_index*3+3
        [T_Hip_Torq, T_Upper_Torq, T_Lower_Torq] = self.target_torques[min_i:max_i]

        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["HIP"], p.TORQUE_CONTROL, force=T_Hip_Torq)
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], p.TORQUE_CONTROL, force=T_Upper_Torq) 
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["LOWER"], p.TORQUE_CONTROL, force=T_Lower_Torq)

        pass

    def handleFlight(self,leg_index):
        
        
        if self.inContact[leg_index]:
            self.inContact[leg_index] = False
        """
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["HIP"], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_index*3])
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_index*3+1])
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["LOWER"], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_index*3+2])
        """

    def checkNeedRestart(self):
        spaceKey = ord(' ')
        keys = p.getKeyboardEvents()
        if spaceKey in keys and keys[spaceKey]&p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(self.robotid, self.reset_pos, self.reset_ori)
            for leg_i, leg_name in enumerate(self.LEGS):
                for joint_i, joint_name in enumerate(self.JOINTS[:-1]):
                    p.resetJointState(self.robotid, self.indices[leg_name][joint_name], self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i])
                    p.setJointMotorControl2(self.robotid, self.indices[leg_name][joint_name], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i], force=0)

    def calcTargetAcc(self, desired_pos, desired_vel, desired_acc):
        pos_error = np.array(desired_pos)-self.q[0:3]
        vel_error = np.array(desired_vel)-self.qdot[0:3]
        target_acc = np.array(desired_acc) + self.Kp*pos_error + self.Kd*vel_error
        return target_acc


    def solve_slip(self):
        
        if not self.apex:
            if self.qdot[2] <= 0 and self.old_qdot[2] >= 0:
                self.apex = True
        
        if self.apex:
            self.apex = False
            self.myslip.update_state( self.q[0], self.q[2], self.qdot[0], self.qdot[2] )
            self.counter = 0
            self.slipsolution = self.myslip.step_apex_to_apex()
            self.max_t = len(self.slipsolution.t)-1
    
    def track_tracjectory(self):
        
        ## Get target desired state variables from slip
        try:
            cur_sol = self.slipsolution.y.T[self.counter]
        except IndexError:
            cur_sol = self.slipsolution.y.T[self.max_t]    
        self.desired_pos = [ cur_sol[0], 0, cur_sol[1] ]
        self.desired_vel = [ cur_sol[2], 0, cur_sol[3] ]
        self.desired_acc = [ 0, 0, 0 ]
        self.counter += 1

        ori_eul = quat2euler(self.q[3:7])
        pos = self.q[0:3]
        ang_vel = self.qdot[3:6]
        lin_vel = self.qdot[0:3]
        X0 = np.array([ 0,          0,          ori_eul[2],
                        pos[0],     pos[1],     pos[2],
                        ang_vel[0], ang_vel[1], ang_vel[2],
                        lin_vel[0], lin_vel[1], lin_vel[2],
                        -9.80665 ] ).reshape(13,1) 
        