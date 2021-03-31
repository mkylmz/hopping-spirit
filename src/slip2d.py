import numpy as np
import math
import scipy.integrate as integrate

class slip2d():
    """
    Vertical SLIP class for simple integration
    """

    def __init__(self, init_state, aoa, rest_length, dt):

        # state vector [ x, y, xdot, ydot, toe_x, toe_y]
        self.state      = [init_state[0], init_state[1], init_state[2], init_state[3], 0, 0]
        self.gravity    = 9.81
        self.dt         = dt

        self.mass       = 11.00
        self.stiffness  = 4000.0
        self.r_length   = rest_length
        self.cur_length = rest_length
        self.aoa        = aoa

        self.state[4]   = self.state[0]+np.sin(self.aoa)*rest_length
        self.state[5]   = self.state[1]-np.cos(self.aoa)*rest_length

        self.max_time = 5

        pass

    def step_apex_to_apex(self):
        '''
        Take one step from apex to apex/failure.
        returns a sol object from integrate.solve_ivp, with all phases
        '''

        AOA = self.aoa
        GRAVITY = self.gravity
        MASS = self.mass
        RESTING_LENGTH = self.r_length
        STIFFNESS = self.stiffness
        #TOTAL_ENERGY

        SPECIFIC_STIFFNESS = STIFFNESS / MASS  # FIXME: Is this name right?

        # Note: not taken from p[]
        HALF_PI = np.pi / 2.0
        MAX_TIME = self.max_time

        def resetLegs(x):
            x[4] = x[0]+np.sin(AOA)*RESTING_LENGTH
            x[5] = x[1]-np.cos(AOA)*RESTING_LENGTH
            return x
        
        def apex_event(t, x):
            ''' Event function to reach apex '''
            return x[3]
        apex_event.terminal = True

        def fall_event(t, x):
            ''' Event function to detect the body hitting the floor (failure)
            '''
            return x[1]
        fall_event.terminal = True

        def flight_dynamics(t, x):
            ''' code in flight dynamics, xdot_ = f() '''
            return np.array([x[2], x[3], 0, -GRAVITY, x[2], x[3]])

        def liftoff_event(t, x, RESTING_LENGTH_SQ=RESTING_LENGTH**2):
            ''' Event function for maximum spring extension (transition to flight)
            '''
            return ((x[0]-x[4])**2 + (x[1]-x[5])**2) - RESTING_LENGTH_SQ
        liftoff_event.terminal = True
        liftoff_event.direction = 1

        def stance_dynamics(t, x):
            # energy = computeTotalEnergy(x,p)
            # print(energy)
            alpha = np.arctan2(x[1]-x[5], x[0]-x[4]) - HALF_PI
            leg_length = np.sqrt((x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2)
            xdotdot = -SPECIFIC_STIFFNESS * (RESTING_LENGTH - leg_length) \
                    * np.sin(alpha)
            ydotdot = SPECIFIC_STIFFNESS * (RESTING_LENGTH - leg_length) \
                    * np.cos(alpha) - GRAVITY
            return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0])

        def touchdown_event(t, x):
            ''' Event function for foot touchdown (transition to stance)
            '''
            # x[1]- np.cos(p["aoa"])*p["resting_length"] (which is = x[5])
            return x[5]
        touchdown_event.terminal = True

        APEX_EVENTS = (fall_event, apex_event)
        FLIGHT_EVENTS = (fall_event, touchdown_event)
        STANCE_EVENTS = (fall_event, liftoff_event)


        t0 = 0
        ''' Starting time '''
        x0 = self.state
        ''' Starting state '''

        # FLIGHT: simulate till touchdown
        sol = integrate.solve_ivp(
                events=FLIGHT_EVENTS,
                fun=flight_dynamics,
                max_step=self.dt,
                t_span=[t0, t0 + MAX_TIME],
                y0=x0,
        )

        # STANCE: simulate till liftoff
        x0 = sol.y[:, -1]
        sol2 = integrate.solve_ivp(
                events=STANCE_EVENTS,
                fun=stance_dynamics,
                max_step=self.dt,
                t_span=[sol.t[-1], sol.t[-1] + MAX_TIME],
                y0=x0,
        )

        # FLIGHT: simulate till apex
        x0 = resetLegs(sol2.y[:, -1])
        sol3 = integrate.solve_ivp(
                events=APEX_EVENTS,
                fun=flight_dynamics,
                max_step=self.dt,
                t_span=[sol2.t[-1], sol2.t[-1] + MAX_TIME],
                y0=x0,
        )

        # concatenate all solutions
        sol.t = np.concatenate((sol.t, sol2.t, sol3.t))
        sol.y = np.concatenate((sol.y, sol2.y, sol3.y), axis=1)
        sol.t_events += sol2.t_events + sol3.t_events
        sol.failed = any(sol.t_events[i].size != 0 for i in (0, 2, 4))
        return sol

    def step_stance(self):
        '''
        Solve stance dynamics
        returns a sol object from integrate.solve_ivp, with all phases
        '''

        AOA = self.aoa
        GRAVITY = self.gravity
        MASS = self.mass
        RESTING_LENGTH = self.r_length
        STIFFNESS = self.stiffness
        #TOTAL_ENERGY

        SPECIFIC_STIFFNESS = STIFFNESS / MASS  # FIXME: Is this name right?

        # Note: not taken from p[]
        HALF_PI = np.pi / 2.0
        MAX_TIME = self.max_time

        def resetLegs(x):
            x[4] = x[0]+np.sin(AOA)*RESTING_LENGTH
            x[5] = x[1]-np.cos(AOA)*RESTING_LENGTH
            return x
        
        def apex_event(t, x):
            ''' Event function to reach apex '''
            return x[3]
        apex_event.terminal = True

        def fall_event(t, x):
            ''' Event function to detect the body hitting the floor (failure)
            '''
            return x[1]
        fall_event.terminal = True

        def liftoff_event(t, x, RESTING_LENGTH_SQ=RESTING_LENGTH**2):
            ''' Event function for maximum spring extension (transition to flight)
            '''
            return ((x[0]-x[4])**2 + (x[1]-x[5])**2) - RESTING_LENGTH_SQ
        liftoff_event.terminal = True
        liftoff_event.direction = 1

        def stance_dynamics(t, x):
            # energy = computeTotalEnergy(x,p)
            # print(energy)
            alpha = np.arctan2(x[1]-x[5], x[0]-x[4]) - HALF_PI
            leg_length = np.sqrt((x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2)
            xdotdot = -SPECIFIC_STIFFNESS * (RESTING_LENGTH - leg_length) \
                    * np.sin(alpha)
            ydotdot = SPECIFIC_STIFFNESS * (RESTING_LENGTH - leg_length) \
                    * np.cos(alpha) - GRAVITY
            return np.array([x[2], x[3], xdotdot, ydotdot, 0, 0])

        def touchdown_event(t, x):
            ''' Event function for foot touchdown (transition to stance)
            '''
            # x[1]- np.cos(p["aoa"])*p["resting_length"] (which is = x[5])
            return x[5]
        touchdown_event.terminal = True

        STANCE_EVENTS = (fall_event, liftoff_event)


        t0 = 0
        ''' Starting time '''
        x0 = resetLegs(self.state)        
        ''' Starting state '''

        # STANCE: simulate till liftoff
        sol = integrate.solve_ivp(
                events=STANCE_EVENTS,
                fun=stance_dynamics,
                max_step=self.dt,
                t_span=[t0, t0 + MAX_TIME],
                y0=x0,
        )

        # concatenate all solutions
        sol.failed = (len(sol.t_events) != 0)
        return sol

    def update_state(self, x, y, xdot, ydot):
        self.state      = [ x, y, xdot, ydot, 
                            x+np.sin(self.aoa)*self.r_length, 
                            y-np.cos(self.aoa)*self.r_length]
