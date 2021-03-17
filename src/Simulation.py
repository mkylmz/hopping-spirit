import pybullet as p
import pybullet_data as pd
import time
from MySpirit40 import MySpirit40 as myrobot

class Simulation:
    """
    my simulation class for pybullet
    """

    def __init__(self,dt,init_pos):
        """
        main initializatio function
        """
        p.connect(p.GUI,options="--width=1920 --height=1080")
        p.setAdditionalSearchPath(pd.getDataPath())
        self.dt = dt

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        self.robot = myrobot(init_pos, dt)
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        p.resetDebugVisualizerCamera(1.5, 30, -52, [0,0,0])
        p.setRealTimeSimulation(0)   

        pass

    def runLoop(self):
        """
        main while loop function
        """

        while p.isConnected():

            start = time.time()
            
            p.stepSimulation()
            self.robot.checkNeedRestart()
            self.robot.control()

            end = time.time()
            diff = self.dt-(end - start)
            if diff > 0:
                time.sleep(diff)
        pass


    pass