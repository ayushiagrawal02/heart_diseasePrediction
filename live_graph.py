import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
x_data, y_data = [], []


line, = ax.plot([], [], lw=2)
ax.set_ylim(60, 100)
ax.set_xlim(0, 100)  # Initial x-axis limit
ax.set_ylabel('Heartbeat')
ax.set_xlabel('Time')
ax.set_title('Live Heartbeat Graph')


def update(frame):
    x_data.append(frame)
    y_data.append(random.randint(65, 95))
    
   
    if frame > 100:
        ax.set_xlim(frame - 100, frame)
    
    line.set_data(x_data[-100:], y_data[-100:])
    return line,


ani = FuncAnimation(fig, update, frames=range(1000), blit=True, interval=100)

plt.show()
