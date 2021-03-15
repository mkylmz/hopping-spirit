
from VerticalSLIP import VerticalSLIP as vslip
import numpy as np
import matplotlib.pyplot as plt

slip = vslip([0, 0.80, 0, 0, 0, 0.51], 0, 0.29, 1/240)

sol = slip.step_apex_to_apex()

plt.plot(sol.t,sol.y[1], color='orange')
plt.show()