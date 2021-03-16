import pybullet as p
import math
import numpy as np
from VerticalSLIP import VerticalSLIP as vslip

class MySpirit40:
    """
    robot class
    """

    def __init__(self, init_pos, dt):

        self.planeid   = p.loadURDF("plane.urdf")
        self.robotid   = p.loadURDF("./myspirit40.urdf", init_pos)
        self.reset_pos = init_pos
        self.reset_ori = [0,0,0,1]
        self.pos       = init_pos
        self.mass      = 11.00
        self.dt        = dt

        #Hardcoded urdf parameters like leg lengths etc.. Could be defined in the robot class later
        self.L1 = 0.206
        self.L2 = 0.206

        self.pos_kp  = 100
        self.pos_kd  = 10
        self.ori_kp  = 100
        self.ori_kd  = 10

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
        
        self.joint_pos = {
            self.LEGS[0] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            },
            self.LEGS[1] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            },
            self.LEGS[2] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            },
            self.LEGS[3] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            }
        } 

        self.joint_vel = {
            self.LEGS[0] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            },
            self.LEGS[1] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            },
            self.LEGS[2] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            },
            self.LEGS[3] : {
                self.JOINTS[0] : 0,
                self.JOINTS[1] : 0,
                self.JOINTS[2] : 0
            }
        }

        self.reaction_forces = {
            self.LEGS[0] : [0,0,0,0,0,0],
            self.LEGS[1] : [0,0,0,0,0,0],
            self.LEGS[2] : [0,0,0,0,0,0],
            self.LEGS[3] : [0,0,0,0,0,0]
        }

        self.inContact = [False,False,False,False]

        # Variable matrices for equations of motion
        self.MassMatrix = []
        self.NMatrix = []
        self.JacT = [[],[],[],[]]
        self.JacR = [[],[],[],[]]
        self.SMatrix = [[0 for i in range(18)]]*6
        for i in range(6):
            for j in range(12):
                self.SMatrix[i][j]
        self.target_torques = np.array([0.0 for i in range(18)])
        self.hip_height     = np.array([0.0 for i in range(4)])
        self.body_vel       = np.array([0.0 for i in range(6)])
        self.body_vel_old   = np.array([0.0 for i in range(6)])
        self.body_acc       = np.array([0.0 for i in range(6)])

        for leg_i, leg_name in enumerate(self.LEGS):
            for joint_i, joint_name in enumerate(self.JOINTS[:-1]):
                p.resetJointState(self.robotid, self.indices[leg_name][joint_name], self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i])
                p.setJointMotorControl2(self.robotid, self.indices[leg_name][joint_name], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i], force=0)
            p.enableJointForceTorqueSensor(self.robotid, self.toe_indices[leg_i], True)
        
    

    def control(self):
        
        self.fetchState()

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
        states = p.getJointStates(self.robotid, range(p.getNumJoints(self.robotid)))
        pos = [ele[0] for ele in states]
        vel = [ele[1] for ele in states]
        rf  = [ele[2] for ele in states]
        self.q_pos = [] 
        self.q_vel = [] 
        self.q_acc = [.0 for i in range(12)]
        for leg_i, leg in enumerate(self.LEGS):
            for joint_i, joint in enumerate(self.JOINTS[:-1]):
                self.joint_pos[leg][joint] = pos[leg_i*4+joint_i]
                self.q_pos.append(pos[leg_i*4+joint_i]) 
                self.joint_vel[leg][joint] = vel[leg_i*4+joint_i]
                self.q_vel.append(vel[leg_i*4+joint_i])
            self.reaction_forces[leg] = rf[leg_i*4+3]

        # Calculate matrices for equations of motion
        self.MassMatrix = p.calculateMassMatrix(self.robotid, self.q_pos)
        self.NMatrix = list(p.calculateInverseDynamics(self.robotid, self.q_pos, self.q_vel, self.q_acc, flags=1))
        del self.NMatrix[6]
        for leg_i, leg in enumerate(self.LEGS):
            [self.JacT[leg_i], self.JacR[leg_i]] = p.calculateJacobian(self.robotid, self.indices[self.LEGS[leg_i]]["TOE"], [0,0,0],  self.q_pos, self.q_vel, self.q_acc)
            self.hip_height[leg_i] = p.getLinkState(self.robotid, self.indices[self.LEGS[leg_i]]["HIP"])[0][2]
        self.body_vel_old = self.body_vel
        body_vel = p.getBaseVelocity(self.robotid)
        self.body_vel = np.array(body_vel[0] + body_vel[1])
        self.body_acc = (self.body_vel - self.body_vel_old)/self.dt
        self.target_torques[0:6] = self.body_acc


    def handleStance(self,leg_index):
        
        if not self.inContact[leg_index]:
            p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["HIP"], p.POSITION_CONTROL, 0, force=0)
            p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], p.POSITION_CONTROL, 0, force=0)
            p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["LOWER"], p.POSITION_CONTROL, 0, force=0)            
            self.inContact[leg_index] = True

        req_F = self.getTargetForce(leg_index)

        min_i = 6+leg_index*3
        max_i = 6+leg_index*3+3
        equ_M = (np.array(self.MassMatrix) @ self.target_torques.T)[min_i:max_i].T
        equ_N = np.array(self.NMatrix[min_i:max_i]).reshape(1,3)
        equ_J = np.array(self.JacT)[leg_index][:,min_i:max_i]
        equ_F = (np.transpose(equ_J) @ req_F).reshape(1,3)
        self.target_torques[min_i:max_i] = equ_M + equ_N - equ_F
        [T_Hip_Torq, T_Upper_Torq, T_Lower_Torq] = self.target_torques[min_i:max_i]

        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["HIP"], p.TORQUE_CONTROL, force=T_Hip_Torq)
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], p.TORQUE_CONTROL, force=T_Upper_Torq) 
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["LOWER"], p.TORQUE_CONTROL, force=T_Lower_Torq)

        pass

    def handleFlight(self,leg_index):
        
        
        if self.inContact[leg_index]:
            self.inContact[leg_index] = False

        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["HIP"], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_index*3])
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_index*3+1])
        p.setJointMotorControl2(self.robotid, self.indices[self.LEGS[leg_index]]["LOWER"], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_index*3+2])


    def checkNeedRestart(self):
        spaceKey = ord(' ')
        keys = p.getKeyboardEvents()
        if spaceKey in keys and keys[spaceKey]&p.KEY_WAS_TRIGGERED:
            p.resetBasePositionAndOrientation(self.robotid, self.reset_pos, self.reset_ori)
            for leg_i, leg_name in enumerate(self.LEGS):
                for joint_i, joint_name in enumerate(self.JOINTS[:-1]):
                    p.resetJointState(self.robotid, self.indices[leg_name][joint_name], self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i])
                    p.setJointMotorControl2(self.robotid, self.indices[leg_name][joint_name], p.POSITION_CONTROL, self.INITIAL_JOINT_POSITIONS[leg_i*3+joint_i], force=0)
            
    
    def calc_slip_state(self, leg_index):
        Lstate = p.getLinkState(self.robotid, self.indices[self.LEGS[leg_index]]["UPPER"], 1, 1)

        x = Lstate[0][0]
        y = Lstate[0][2]

        xdot = Lstate[6][0]
        ydot = Lstate[6][2]

        Lstate = p.getLinkState(self.robotid, self.indices[self.LEGS[leg_index]]["TOE"], 0, 1)

        toe_x = Lstate[0][0]
        toe_y = Lstate[0][2]

        return [x, y, xdot, ydot, toe_x, toe_y]

    def slip2polar(self,slip_state):
        L       =   math.sqrt(  (slip_state[0]-slip_state[4])**2 + \
                                (slip_state[1]-slip_state[5])**2 )
        aoa     =  np.arctan2(  (slip_state[0]-slip_state[4]), \
                                (slip_state[1]-slip_state[5]) )
        L_dot   =   math.cos( aoa ) * slip_state[3] + \
                    math.cos( np.pi/2 - aoa ) * slip_state[2]
        aoa_dot =  -math.sin( aoa ) * slip_state[3] + \
                    math.cos( aoa ) * slip_state[2]
        return [L, L_dot, aoa, aoa_dot]

    def getTargetForce(self, leg_index):
        
        base_force = self.mass*2.5
        reqF = np.array([0, 0, base_force])

        [base_pos, base_ori] = p.getBasePositionAndOrientation(self.robotid)
        base_ori = p.getEulerFromQuaternion(base_ori)
        [base_linvel, base_angvel] = p.getBaseVelocity(self.robotid)
        
        reqF += np.array([  -self.pos_kp*base_pos[0] - self.pos_kd*base_linvel[0], 
                            -self.pos_kp*base_pos[1] - self.pos_kd*base_linvel[1], 
                            -self.pos_kp*(self.hip_height[leg_index]-self.rest_length)  - self.pos_kd*base_linvel[2] ])
        
        if (leg_index == 0):   # FL
            reqF += np.array([0, 0, -self.ori_kp*base_ori[0] - self.ori_kd*base_angvel[0] + self.ori_kp*base_ori[1] + self.ori_kd*base_angvel[1] ])

        elif (leg_index == 1): # RL
            reqF += np.array([0, 0, -self.ori_kp*base_ori[0] - self.ori_kd*base_angvel[0] - self.ori_kp*base_ori[1] - self.ori_kd*base_angvel[1] ]) 

        elif (leg_index == 2): # FR
            reqF += np.array([0, 0,  self.ori_kp*base_ori[0] + self.ori_kd*base_angvel[0] + self.ori_kp*base_ori[1] + self.ori_kd*base_angvel[1] ]) 

        elif (leg_index == 3): # RR
            reqF += np.array([0, 0,  self.ori_kp*base_ori[0] + self.ori_kd*base_angvel[0] - self.ori_kp*base_ori[1] - self.ori_kd*base_angvel[1] ]) 

        return reqF