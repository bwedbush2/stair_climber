# Finite constrained optimization problem
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import mujoco

def _solve_cftoc(A, B, P, Q, R, N, x0, xL, xU, uL, uU, bf, Af):

    # Initialize model
    model = pyo.ConcreteModel()
    nx = np.size(x0)
    nu = np.size(uU)
    model.tidx = pyo.Set(initialize=range(0, N+1))
    model.tidu = pyo.Set(initialize=range(0, N))
    model.nx = pyo.Set(initialize=range(0, nx))
    model.nu = pyo.Set(initialize=range(0, nu))

    # Create state and input variables trajectory:
    model.x = pyo.Var(model.nx, model.tidx) # x is the state vector
    model.u = pyo.Var(model.nu, model.tidu) # u is the input vector

    # Cost function
    model.P = P
    model.Q = Q
    model.R = R
    terminal_cost = sum(
        model.x[i, N] * model.P[i, j] * model.x[j, N]
        for i in model.nx
        for j in model.nx
    )
    state_cost = sum(
        model.x[i, t] * model.Q[i, j] * model.x[j, t]
        for t in model.tidu
        for i in model.nx
        for j in model.nx
    )
    control_cost = sum(
        model.u[i, t] * model.R[i, j] * model.u[j, t]
        for t in model.tidu
        for i in model.nu
        for j in model.nu
    )
    model.cost = pyo.Objective(
        expr=terminal_cost + state_cost + control_cost,
        sense=pyo.minimize
    )

def _solve_cftoc_disc(gam, tau, N, x0, xL, xU, uL, uU, bf, Af):

    nx = 3         # number of states (x,y,theta)
    nu = 2         # number of inputs u1=v, u2=omega

    model = pyo.ConcreteModel()
    model.tidx = pyo.Set(initialize=range(0, N+1)) # length of finite optimization problem
    model.xidx = pyo.Set(initialize=range(0, nx))
    model.uidx = pyo.Set(initialize=range(0, nu))

    # Create state and input variables trajectory:
    model.z = pyo.Var(model.xidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)


    # Objective:
    model.shortest_time_cost  = sum((model.z[1, t]-zf[1])**2 for t in model.tidx if t < N)
    model.turning_cost = sum((model.u[1, t])**2 for t in model.tidx if t < N)
    model.speed_cost = sum((model.u[0, t])**2 for t in model.tidx if t < N)
    model.cost = pyo.Objective(expr = model.speed_cost, sense=pyo.minimize)

    # # Initial condition
    # model.constraint1 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i, 0] == z0[i])

    # Dynamic constraints
    model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] == model.z[0, t] + Ts* (pyo.cos(model.z[2, t]) *model.u[0, t])
                                    if t < N else pyo.Constraint.Skip)
    model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] == model.z[1, t] + Ts* (pyo.sin(model.z[2, t]) *model.u[0, t])
                                    if t < N else pyo.Constraint.Skip)
    model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] == model.z[2, t] + Ts* model.u[1, t]
                                    if t < N else pyo.Constraint.Skip)
    
    # # State and input constraints
    # model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= 1
    #                                 if t <= N-1 else pyo.Constraint.Skip)
    # model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >=-1
    #                                 if t <= N-1 else pyo.Constraint.Skip)
    # model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= 1
    #                                 if t <= N-1 else pyo.Constraint.Skip)
    # model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= -1
    #                                 if t <= N-1 else pyo.Constraint.Skip)
    # #model.constraint10 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t] <= 0.4
    # #                                   if t <= N-1 else pyo.Constraint.Skip)

    # State bounds:  xL <= x[i,t] <= xU  for all i in nx, t in tidx
    model.x_upper = pyo.Constraint(model.nx, model.tidx,
        rule=lambda m, i, t: m.x[i, t] <= xU
    )
    model.x_lower = pyo.Constraint(model.nx, model.tidx,
        rule=lambda m, i, t: m.x[i, t] >= xL
    )
    # Control bounds: uL <= u[i,t] <= uU for all i in nu, t in tidx
    model.u_upper = pyo.Constraint(model.nu, model.tidu,
        rule=lambda m, i, t: m.u[i, t] <= uU
    )
    model.u_lower = pyo.Constraint(model.nu, model.tidu,
        rule=lambda m, i, t: m.u[i, t] >= uL
    )
    
    # Initial Constraints
    model.x0_con = pyo.Constraint(model.nx, rule= lambda m, i: m.x[i, 0] == x0[i])
    
    # Terminal constraint
    Af_list = Af.tolist() if hasattr(Af, "tolist") else Af
    if len(Af_list) != 0:   # set this from Af.shape or len(Af_list)
        def xf_rule(m, i):
            return sum(Af_list[i][j] * m.x[j, N] for j in m.nx) <= bf[i]
    else:
        def xf_rule(m, i):
            return m.x[i, N] == bf[i]
    model.xf_con = pyo.Constraint(model.nx, rule=xf_rule)

    # # Terminal constraint
    # model.constraint9 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i, N] == zf[i])

   
    # Now we can solve:
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    # u1 = [pyo.value(model.u[0,0])]
    # u2 = [pyo.value(model.u[1,0])]

    # num_vars = pyo.value(model.nvariables())
    # num_cons = pyo.value(model.nconstraints())

    # print("Total variables:", num_vars)
    # print("Total constraints:", num_cons)

    # OUTPUTS : feas, xOpt, uOpt, JOpt
    feas = (
        results.solver.status == SolverStatus.ok and 
        results.solver.termination_condition in (
            TerminationCondition.optimal,
            TerminationCondition.feasible
        )
    )
    xOpt = np.zeros((nx,N+1))
    for i in model.nx:
        for t in model.tidx:
            xOpt[i, t] = pyo.value(model.x[i, t])
    uOpt = np.zeros((nu, N))
    for i in model.nu:
        for t in model.tidu:
            uOpt[i, t] = pyo.value(model.u[i, t])
    JOpt = pyo.value(model.cost)

    return [feas, xOpt, uOpt, JOpt]
    

# Euler discretized unicycle model
# x is state vector, u is input, Ts is sampling period
def fdis(z, u, Ts):
    nz = z.shape(0)
    z_next = np.empty((nz,))
    z_next[0] = z[0] + Ts*(np.cos(z[2]) * u[0])
    z_next[1] = z[1] + Ts*(np.sin(z[2]) * u[0])
    z_next[2] = z[2] + Ts*u[1]
    return z_next

def flin(z, u, Ts):
    nz = z.shape(0)
    z_next = np.empty((nz,))
    z_next[0] = z[0] + Ts*(np.cos(z[2]) * u[0])
    z_next[1] = z[1] + Ts*(np.sin(z[2]) * u[0])
    z_next[2] = z[2] + Ts*u[1]
    return z_next

# Helper: get (x, y, yaw) of the car body in world frame
def _get_car_pose(model, data, body_name="car"):
    """
    Returns (x, y, yaw) of the given body in world coordinates.
    yaw is extracted from the body's rotation matrix assuming z-up.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # World position of body (3,)
    pos = data.xpos[body_id].copy()
    x, y = pos[0], pos[1]

    # World rotation matrix of body (3x3), stored flat in row-major order
    R_flat = data.xmat[body_id].copy()
    R = R_flat.reshape(3, 3)

    # Heading is body x-axis in world frame
    heading = R[:, 0]
    yaw = np.arctan2(heading[1], heading[0])

    return [x, y, yaw]

def traj_mpc(model, data, path) -> tuple[float, float] :
    '''
    This is a controller that uses MPC to optimize trajector control
    given desired waypoints

    :param model: mujoco model
    :param data: mujoco data
    :return: control inputs for drive and turn commands
    :rtype: tuple[float, float]
    '''

    # Simulation params
    Ts = model.opt.timestep     # time step for simulation model
    TFinal = 1                  # picked to be 1 second. Increase if myopic. Decrease if computation is slow
    N = TFinal / Ts             # Horizon

    # Model params
    gam = 1     # drive param
    tau = 0.5   # turn param

    # Constraints:
    z0 = _get_car_pose(model, data)

    target_idx = data.userdata[1]
    xt, yt, _ = path[target_idx]    # target waypoint
    # NEXT target waypoint
    xt_next, yt_next, _ = path[target_idx + 1] if target_idx < len(path) else [xt,yt,0]
    #yawt = 
    zf=[xt,yt,np.pi/4]

    feas, xOpt, uOpt, JOpt = _solve_cftoc_disc(gam, tau, N, z0, xL, xU, uL, uU, bf, Af)
    
    u1, u2 = uOpt[:,0]

    return u1, u2
