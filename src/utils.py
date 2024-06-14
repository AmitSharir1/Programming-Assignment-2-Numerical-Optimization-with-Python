import matplotlib.pyplot as plt
import numpy as np


def plot_qp_path(path, title):
    path = np.array(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Optimization Path', color='blue')
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=100, c='red', marker='x', label='Optimal Solution')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.set_zlabel('Z-axis', fontsize=12)
    
    ax.view_init(elev=50, azim=50)
    plt.legend()
    plt.show()

def plot_lp_path(path, title):
    path = np.array(path)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-2, 2, 1000)
    constraints_ineq = {
        'y=0': (x, np.zeros_like(x)),
        'y=1': (x, np.ones_like(x)),
        'x=2': (np.full_like(y, 2), y),
        'y=-x+1': (x, -x + 1)
    }

    # Define colors for each constraint
    colors = {
        'y=0': 'green',
        'y=1': 'blue',
        'x=2': 'purple',
        'y=-x+1': 'orange'
    }
    
    for label, (x_vals, y_vals) in constraints_ineq.items():
        ax.plot(x_vals, y_vals, label=label, color=colors[label])

    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], 'lightgray', label='Feasible Region')
    ax.plot(path[:, 0], path[:, 1], color='red', label='Optimization Path')
    ax.scatter(path[-1][0], path[-1][1], s=100, c='gold', marker='x', label='Optimal Solution')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X-axis', fontsize=12)
    ax.set_ylabel('Y-axis', fontsize=12)
    ax.legend(loc='best')
    plt.show()

def plot_objective_values(values, title):
    values = np.array(values)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(len(values))
    ax.plot(iterations, values, marker='o', linestyle='-', color='blue', markersize=5)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    
    plt.grid(True)
    plt.show()