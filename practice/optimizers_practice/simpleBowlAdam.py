import numpy as np
import matplotlib.pyplot as plt

from practice.optimizers import Adam
from practice.simple_layers import SimpleOvalSquareFuncLayer_v2

# start at a point -3,3
params = {'x1': -1.9, 'x2': -1.9}
lr = 0.1
rho1 = 0.9
rho2 = 0.9
nesterov_ag = Adam(lr, rho1, rho2)

# do 17 iterations
loss_list = []
x1_list = []
x2_list = []
iters_num = 100
for it in range(iters_num):
    simpleLayer = SimpleOvalSquareFuncLayer_v2(params['x1'], params['x2'])
    x1_list.append(params['x1'])
    x2_list.append(params['x2'])
    loss_list.append(simpleLayer.forward())
    dx1, dx2 = simpleLayer.backward(1)
    grads = {'x1': dx1, 'x2': dx2}
    nesterov_ag.update(params, grads)

print(f"final paarmeters : {params}")
x1_arr = np.array(x1_list)
x2_arr = np.array(x2_list)
x1_coords = np.arange(-2, 5, 0.1)
x2_coords = np.arange(-2, 2, 0.1)
mx1, mx2 = np.meshgrid(x1_coords, x2_coords)
simpleLayer = SimpleOvalSquareFuncLayer_v2(mx1, mx2)
mout = simpleLayer.forward()
plt.pcolormesh(mx1, mx2, mout, cmap='jet')
plt.colorbar()
plt.plot(x1_arr, x2_arr, color='w', linestyle='dashed', marker='o')
plt.show()
