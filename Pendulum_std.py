import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import sys
import os
import csv
import shutil


# switching to a pendulum
vars = dict(lagrange_factor=10 ** -5, opt_steps=500000, opt_step_sz=1 ** 0, total_time=3, time_samples=300,
            start= .3 * math.pi / 2, end= -1.2 * math.pi / 2, rand_factor= math.pi / 2, grad_range=100)
dt = vars['total_time'] / vars['time_samples']


def lagrangian(q, m=1, g=1, l=1):
    (x, xdot) = q
    return vars['lagrange_factor']*m * l * ((2 * g) ** -1 * l * xdot ** 2 + torch.cos(x) -1)


def action(x, dt=.01):
    ''' q is a 1D tensor of n values, one for each (discretized) time step'''
    xdot = (x[1:] - x[:-1]) / dt
    xdot = torch.cat([xdot, xdot[-1:]])
    return lagrangian(q=(x, xdot)).sum()


def get_path_between(x, steps=1000, step_size=1e-1, dt=.01):
    t = np.linspace(0, int(vars['total_time']), int(vars['time_samples']))
    xs = []
    for i in range(steps):
        grad = torch.autograd.grad(action(x, dt), x)
        if 1 in torch.isnan(grad[0]):
            sys.exit('NaN')
        grad_x = grad[0]
        grad_x[[0, -1]] *= 0  # fix first and last coordinates by zeroing their grads
        torch.clamp(grad_x, -abs(vars['grad_range']), abs(vars['grad_range']))
        x.data -= grad_x * step_size

        if i % (steps // 100) == 0:
            xs.append(x.clone().data)
            print('step={:04d}, S={:.3e}'.format(i, action(x).item()))
    return t, x, xs


# Experimentation with different initial conditions
xb = vars['rand_factor'] * torch.randn(vars['time_samples'], requires_grad=True)
xa = vars['start'] * torch.ones(vars['time_samples'], requires_grad=True)
x0 = xa + xb
x0[0].data *= 0.0; x0[-1].data *= 0.0  # set first and last points to zero
x0[0].data += vars['start']; x0[-1].data += vars['end'] # set first and last points to desired values

# Pendulum simulation tends to take more optimization steps to converge compared to freefall
t, x, xs = get_path_between(x0.clone(), steps=vars['opt_steps'], step_size=vars['opt_step_sz'], dt=dt)

plt.figure(dpi=90)
plt.title('Simulation as Optimization (Pendulum)')


def pend_height(y):
    height = np.ones(len(y)) - np.cos(y.detach().numpy())
    return height


plt.plot(t, pend_height(x0), 'y.-', label='Initial (random) path')
for i, xi in enumerate(xs):
    label = 'During optimization' if i == vars['opt_steps'] // 2 else None
    plt.plot(t, pend_height((xi)), alpha=0.3, color=plt.cm.viridis(1 - i / (len(xs) - 1)), label=label)
# Conversion to (one) cartesian coordinate for qualitative analysis
plt.plot(t, pend_height(x), 'b.-', label='Final (optimized) path')
plt.plot(t[[0, -1]], pend_height(x0.data[[0, -1]]), 'b+', markersize=15, label= "Points held constant")
plt.ylim(-.25, 2.25)
plt.xlabel('Time (s)')
plt.ylabel('Height (l = 1)')
plt.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig('graph.png')
plt.show()

# data collection
def save(vars):
    i = 0
    root = f'../../PendulumSim/'
    while os.path.isdir(f'{root}{i}'):
        i = i + 1
    os.mkdir(f'{root}{i}')
    w = csv.writer(open(f"{root}{i}/dict.csv", "w"))
    for key, val in vars.items():
        w.writerow([key, val])

    shutil.move('graph.png', f"{root}{i}/graph.png")
    torch.save(xs, f"{root}{i}/data.pt")


if int(input('Save trial (0 / 1)? : ')) != 0:
    save(vars)
