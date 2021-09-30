import matplotlib.pyplot as plt
import time
import random
import numpy as np
 
#ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata1 = []
ydata2 = []
ydata3 = []
 
plt.show()
plt.style.use('ggplot') 

fig, (ax1, ax2) = plt.subplots(2, 1)


line1, = ax1.plot(xdata, ydata1, color='red')
line2, = ax2.plot(xdata, ydata2, color='blue', label='Train Accuracy')
line3, = ax2.plot(xdata, ydata3, color='orange', label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax1.set_title('Train Loss')
ax2.set_title('Accuracy')
fig.legend(loc=4)
fig.tight_layout()

max_plot1 = 0
max_plot2 = 0
for epoch in range(1, 100):
    loss = -epoch if random.random() > 0.5 else random.randint(1, 50)
    accuracy = np.log(epoch) if random.random() > 0.5 else random.randint(1, 50)
    accuracy2 = np.log(epoch) if random.random() > 0.5 else random.randint(1, 50)
    if loss > max_plot1:
        max_plot1 = np.ceil(loss)
    if accuracy > max_plot2:
        max_plot2 = np.ceil(accuracy)
    if accuracy2 > max_plot2:
        max_plot2 = np.ceil(accuracy2)
        
            
    xdata.append(epoch)
    ydata1.append(loss)
    ydata2.append(accuracy)
    ydata3.append(accuracy2)
    line1.set_xdata(xdata)
    line1.set_ydata(ydata1)
    line2.set_xdata(xdata)
    line2.set_ydata(ydata2)
    line3.set_xdata(xdata)
    line3.set_ydata(ydata3)
    ax1.set_xlim(0, epoch)
    ax1.set_ylim(0, max_plot1)
    ax2.set_xlim(0, epoch)
    ax2.set_ylim(0, max_plot2)
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
    
    
 
 
# add this if you don't want the window to disappear at the end
plt.show()