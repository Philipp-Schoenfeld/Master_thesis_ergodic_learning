import numpy as np
import ompl.base as ob
import ompl.geometric as og

def init_particles(N, T, noise_std=0.02):
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(1.0)
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    
    class MyValidStateChecker(ob.StateValidityChecker):
        def __init__(self, spaceInformation):
            super().__init__(spaceInformation)

        def isValid(self, state):
            x = state[0]
            y = state[1]
            return (x - 0.5)**2 + (y - 0.5)**2 > 0.13**2

    validityChecker = MyValidStateChecker(si)
    si.setStateValidityChecker(validityChecker)
    si.setup()

    pdef = ob.ProblemDefinition(si)
    start = space.allocState()
    start[0] = 0.2
    start[1] = 0.15
    
    goal = space.allocState()
    goal[0] = 0.8
    goal[1] = 0.85
    
    pdef.setStartAndGoalStates(start, goal)

    planner = og.RRT(si)
    planner.setProblemDefinition(pdef)
    planner.setup()
    
    solved = planner.solve(5.0)
    if not solved:
        print("Warning: RRT could not find a path. Falling back to linear.")
        t = np.linspace(0, 1, T)[:, None]
        base_traj = (1 - t) * np.array([0.2, 0.15]) + t * np.array([0.8, 0.85])
    else:
        path = pdef.getSolutionPath()
        states = path.getStates()
        pts = np.array([[s[0], s[1]] for s in states])
        
        # Resample exactly to T points
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        
        if cum_dists[-1] > 0:
            cum_dists /= cum_dists[-1]
            t_orig = cum_dists
        else:
            t_orig = np.linspace(0, 1, len(pts))
            
        t_new = np.linspace(0, 1, T)
        base_traj = np.stack([
            np.interp(t_new, t_orig, pts[:, 0]),
            np.interp(t_new, t_orig, pts[:, 1])
        ], axis=-1)
        
    particles = []
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(T, 2))
        traj = base_traj + noise
        traj = np.clip(traj, 0.02, 0.98)
        particles.append(traj.ravel())
        
    return np.array(particles), base_traj
