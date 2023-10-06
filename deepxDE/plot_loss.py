import numpy as np
import matplotlib.animation as animation

data=np.loadtxt("loss_vnorm.txt")
print(data.shape)

steps=data[:,0]
loss_pde_v=data[:,1]
loss_pde_w=data[:,2]
loss_data=data[:,-1]

total_loss=loss_pde_v+loss_pde_w+loss_data

import matplotlib.pyplot as plt
plt.plot(steps,loss_pde_v,label="loss_pde_v")
plt.plot(steps,loss_pde_w,label="loss_pde_w")
plt.plot(steps,loss_data,label="loss_data")
#plt.plot(steps,total_loss,label="total_loss")
plt.legend()
plt.yscale("log")
plt.savefig("loss_vnorm.png")