from Simulation import Simulation

init_pos = [0,0,0.27]
init_ori = [0,0,0]

my_sim = Simulation(1./20,init_pos)

my_sim.runLoop()