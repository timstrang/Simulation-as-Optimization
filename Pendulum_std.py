import numpy as np
import matplotlib.pyplot as plt
import torch
import math

# switching to a pendulum


def lagrangian(q, m=1, g=1, l=1):
    (x, xdot) = q
    return m * l * ((2 * g) ** -1 * l * xdot ** 2 + torch.cos(x) -1)


def action(x, dt=.01):
    ''' q is a 1D tensor of n values, one for each (discretized) time step'''
    xdot = (x[1:] - x[:-1]) / dt
    xdot = torch.cat([xdot, xdot[-1:]])
    return lagrangian(q=(x, xdot)).sum()


def get_path_between(x, steps=1000, step_size=1e-1, dt=.01):
    t = np.linspace(0, len(x) - 1, len(x)) * dt
    xs = []
    for i in range(steps):
        grad = torch.autograd.grad(action(x, dt), x)
        grad_x = grad[0] * .1
        grad_x[[0, -1]] *= 0  # fix first and last coordinates by zeroing their grads
        x.data -= grad_x * step_size

        if i % (steps // 100) == 0:
            xs.append(x.clone().data)
            print('step={:04d}, S={:.3e}'.format(i, action(x).item()))
    return t, x, xs

dt = .1
# Experimentation with different initial conditions
xb = 10 * torch.randn(1000, requires_grad=True)
xa = 90 * torch.ones(1000, requires_grad=True)
for i in range(len(xa)):
    if i > (len(xa) // 2):
        xa[i] = 180
    if i > 3*(len(xa) // 4):
        xa[i] = -90
x0 = xa + xb
x0[0].data *= 0.0; x0[-1].data *= 0.0  # set first and last points to zero
x0[0].data += 90; x0[-1].data += -90  # set first and last points to desired values

# Pendulum simulation tends to take more optimization steps to converge compared to freefall
t, x, xs = get_path_between(x0.clone(), steps=500000, step_size=3e-2, dt=dt)

plt.figure(dpi=90)
plt.title('Simulation as Optimization (Pendulum)')

plt.plot(t, x0.detach().numpy(), 'y.-', label='Initial (random) path')
for i, xi in enumerate(xs):
    label = 'During optimization' if i == 10000 else None
    plt.plot(t, -torch.cos(xi), alpha=0.3, color=plt.cm.viridis(1 - i / (len(xs) - 1)), label=label)
# Conversion to (one) cartesian coordinate for qualitative analysis
plt.plot(t, -np.cos(x.detach().numpy()), 'b.-', label='Final (optimized) path')
plt.plot(t[[0, -1]], x0.data[[0, -1]], 'b+', markersize=15, label='Points held constant')

plt.ylim(-1.25, 1.25)
plt.xlabel('Time (s)');
plt.ylabel('Height (l = 1)');
plt.legend(fontsize=8, ncol=3)
plt.tight_layout();
plt.show()
