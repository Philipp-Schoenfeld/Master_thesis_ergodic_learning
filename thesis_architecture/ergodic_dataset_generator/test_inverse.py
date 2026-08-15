import numpy as np

def inverse_euler(p_traj, dt):
    # p_traj shape: (T+1, 2)
    # returns u_traj shape: (T, 2), v0 shape: (2,)
    T = len(p_traj) - 1
    v = np.zeros((T+1, 2))
    u = np.zeros((T, 2))
    
    # v[t] = (p[t+1] - p[t]) / dt
    for t in range(T):
        v[t] = (p_traj[t+1] - p_traj[t]) / dt
        
    # v[T] is not defined by forward difference, just copy v[T-1]
    v[T] = v[T-1]
        
    for t in range(T):
        u[t] = (v[t+1] - v[t]) / dt
        
    return u, v[0]

dt = 0.05
T = 200
t_eval = np.linspace(0, 2*np.pi, T+1)
p_traj = np.column_stack([np.sin(t_eval), np.cos(t_eval)])

u, v0 = inverse_euler(p_traj, dt)

# forward simulate
p_sim = np.zeros((T+1, 2))
v_sim = np.zeros((T+1, 2))
p_sim[0] = p_traj[0]
v_sim[0] = v0

for t in range(T):
    p_sim[t+1] = p_sim[t] + dt * v_sim[t]
    v_sim[t+1] = v_sim[t] + dt * u[t]
    
print("Max diff:", np.max(np.abs(p_sim - p_traj)))
