import pybullet as p
import math
import numpy as np
from VerticalSLIP import VerticalSLIP as vslip
import pathlib
import rbdyn as rbd
import eigen as e
import picos
import time

class MySpirit40:
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

        self.dyn       = rbd.parsers.from_urdf_file(urdf_path,fixed=False,baseLink="body")
        self.dyn.mbc.zero(self.dyn.mb)
        self.dyn.mbc.gravity = e.eigen.Vector3d(0,0,-9.8)

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

        self.myslip = vslip([init_pos[0], init_pos[2], 0, 0, 0, 0], self.aoa, self.rest_length, self.dt)
        self.apex = True
        self.slipsolution = []

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
         
        ## update position and oritentation with forward kinematics
        q = [[] for x in range(17)]
        q[0] = [ body_pos[1][3]] + list(body_pos[1][0:3]) + list(body_pos[0])
        for leg_i in range(4):
            for joint_i in range(3):
                q[1+leg_i*4+joint_i] = [self.q[7+leg_i*3+joint_i]]
        self.dyn.mbc.q = q
        rbd.forwardKinematics(self.dyn.mb,self.dyn.mbc)

        ## update velocity with forward velocity
        alpha = [[] for x in range(17)]
        alpha[0] = self.qdotdot[0:6]
        for leg_i in range(4):
            for joint_i in range(3):
                alpha[1+leg_i*4+joint_i] = [self.qdot[6+leg_i*3+joint_i]]
        self.dyn.mbc.alpha = alpha
        rbd.forwardVelocity(self.dyn.mb,self.dyn.mbc)

        ## update acceleration with forward acceleration
        alphaD = [[] for x in range(17)]
        alphaD[0] = self.qdotdot[0:6]
        for leg_i in range(4):
            for joint_i in range(3):
                alphaD[1+leg_i*4+joint_i] = [self.qdotdot[6+leg_i*3+joint_i]]
        self.dyn.mbc.alphaD = alphaD
        rbd.forwardAcceleration(self.dyn.mb,self.dyn.mbc)            

        ## Calculate CoM Jacobian and its derivative
        jac_com = rbd.CoMJacobian(self.dyn.mb)
        self.CoM_Jac = np.array(jac_com.jacobian(self.dyn.mb, self.dyn.mbc))
        self.CoM_Jac_Dot = np.array(jac_com.jacobianDot(self.dyn.mb, self.dyn.mbc))
        self.normal_acc = self.CoM_Jac_Dot@self.qdot
        #self.normal_acc = np.array(jac_com.normalAcceleration(self.dyn.mb, self.dyn.mbc)) # J_dot*q_dot

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

        ## Initialize PICOS problem
        P = picos.Problem()
        #P.options.solver = "cvxopt"
        P.options["*_tol"] = 10e-6
        qdotdot = picos.RealVariable("qdotdot", (18,1) ) 
        qdotdot.value = self.qdotdot

        ## Define motion constraint
        M = picos.Constant("M", self.MassMatrix, (18,18) )
        N = picos.Constant("N", self.NMatrix, (18,1) )
        S = picos.Constant("S", self.SelMat.T, (18,12))
        Jt_foot0 = picos.Constant( "Jt_foot0", np.array(self.JacT[0]).T, (18,3) )
        Jt_foot1 = picos.Constant( "Jt_foot1", np.array(self.JacT[1]).T, (18,3) )
        Jt_foot2 = picos.Constant( "Jt_foot2", np.array(self.JacT[2]).T, (18,3) )
        Jt_foot3 = picos.Constant( "Jt_foot3", np.array(self.JacT[3]).T, (18,3) )
        torques = picos.RealVariable("torques", (12,1) )
        torques.value = self.target_torques
        
        f_foot0 = picos.RealVariable( "f_foot0", (3,1) )
        P.add_constraint(f_foot0[2] >= 0)
        P.add_constraint(f_foot0[2]*self.friction_coeff >= abs(f_foot0[1]))
        P.add_constraint(f_foot0[2]*self.friction_coeff >= abs(f_foot0[0]))

        f_foot1 = picos.RealVariable( "f_foot1", (3,1) )
        P.add_constraint(f_foot1[2] >= 0)
        P.add_constraint(f_foot1[2]*self.friction_coeff >= abs(f_foot1[1]))
        P.add_constraint(f_foot1[2]*self.friction_coeff >= abs(f_foot1[0]))

        f_foot2 = picos.RealVariable( "f_foot2", (3,1) )
        P.add_constraint(f_foot2[2] >= 0)
        P.add_constraint(f_foot2[2]*self.friction_coeff >= abs(f_foot2[1]))
        P.add_constraint(f_foot2[2]*self.friction_coeff >= abs(f_foot2[0]))

        f_foot3 = picos.RealVariable( "f_foot3", (3,1) )
        P.add_constraint(f_foot3[2] >= 0)
        P.add_constraint(f_foot3[2]*self.friction_coeff >= abs(f_foot3[1]))
        P.add_constraint(f_foot3[2]*self.friction_coeff >= abs(f_foot3[0]))

        contact_force = Jt_foot0*f_foot0 + Jt_foot1*f_foot1 + Jt_foot2*f_foot2 + Jt_foot3*f_foot3
        P.add_constraint(M*qdotdot+N==S*torques+contact_force)
        
        ## Force constraint
        for leg_i in range(4):
            if self.inContact[leg_i]:
                self.force_sel_mat[leg_i*3:leg_i*3+3] = np.array([0,0,0])
            else:
                self.force_sel_mat[leg_i*3:leg_i*3+3] = np.array([1,1,1])
        S_f = picos.Constant("S_f", self.force_sel_mat.T, (1,12) )
        forces = f_foot0 // f_foot1 // f_foot2 // f_foot3
        P.add_constraint( S_f*forces == 0 )

        ## Torque constraint
        P.add_constraint(abs(torques) <= 10.0)

        ## Define objective
        CoM_Jac = picos.Constant("J", self.CoM_Jac, (3,18) )
        body_normal_acc = picos.Constant("normal_acc", self.normal_acc, (3,1))
        self.target_acc[0:3] = self.calcTargetAcc(self.desired_pos, self.desired_vel, self.desired_acc) 
        target_acc = picos.Constant("target_acc", self.target_acc.T, (15,1))
        J = CoM_Jac // Jt_foot0.T // Jt_foot1.T // Jt_foot2.T // Jt_foot3.T
        normal_acc_foot0 = self.JacTDot[0] @ self.qdot
        normal_acc_foot1 = self.JacTDot[1] @ self.qdot
        normal_acc_foot2 = self.JacTDot[2] @ self.qdot
        normal_acc_foot3 = self.JacTDot[3] @ self.qdot
        normal_acc = body_normal_acc // normal_acc_foot0 // normal_acc_foot1 // normal_acc_foot2 // normal_acc_foot3
        P.set_objective("min", abs(J*qdotdot+normal_acc-target_acc)**2 )
        
        ## Solve
        #print(P)
        try:
            if any(self.inContact):
                solution = P.solve()
                self.target_torques[0:12] = np.array(torques.value).reshape(12,)
        except ValueError:
            print("ValueError is given at " + str(time.time()) + " !")
        except picos.modeling.problem.SolutionFailure:
            print("Solution not found at  " + str(time.time()) + " !")
        