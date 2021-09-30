import matplotlib.pyplot as plt
import time
import random
import numpy as np
 
#ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(auto=True)
axes.set_ylim(auto=True) 
line, = axes.plot(xdata, ydata, 'r-')
max_plot = 0
for epoch in range(1, 100):
    loss = np.log(epoch) if random.random() > 0.5 else random.randint(1, 50)
    if loss > max_plot:
        max_plot = np.ceil(loss)
    xdata.append(epoch)
    ydata.append(loss)
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    axes.set_xlim(0, epoch)
    axes.set_ylim(0, max_plot)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
    
    
 
 
# add this if you don't want the window to disappear at the end
plt.show()