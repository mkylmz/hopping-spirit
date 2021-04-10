from sim_module import sim_module

init_pos = [0,0,0.30]
init_ori = [0,0,0]

my_sim = sim_module(1./200,init_pos)

my_sim.runLoop()