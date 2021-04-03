import numpy as np
import qpsolvers

class convex_MPC:
  """
  Convex Model Predictive Control Class for legged robots
  """

  def __init__(self, horizon):
    
    self.horizon = horizon

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

  pass