# Finite constrained optimization problem
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import mujoco


def _solve_cftoc_disc(gam, tau, Ts, N, x0, xL, xU, uL, uU, bf, Af):

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


    # Objective:
    w_xy  = 20.0
    w_yaw = 5.0
    w_v   = 0.2
    w_w   = 0.5

    w_xy_T  = 200.0
    w_yaw_T = 50.0

    model.track_cost = sum(
        w_xy * ((model.x[0, t] - bf[0])**2 + (model.x[1, t] - bf[1])**2)
    + w_yaw * (1 - pyo.cos(model.x[2, t] - bf[2]))
        for t in model.tidu
    )
    model.terminal_cost = (
        w_xy_T * ((model.x[0, N] - bf[0])**2 + (model.x[1, N] - bf[1])**2)
    + w_yaw_T * (1 - pyo.cos(model.x[2, N] - bf[2]))
    )
    model.shortest_time_cost  = sum((model.x[1, t]-bf[1])**2 for t in model.tidx if t < N)
    model.turning_cost = sum(w_w * (model.u[1, t]**2) for t in model.tidu)
    model.speed_cost   = sum(w_v * (model.u[0, t]**2) for t in model.tidu)
    model.cost = pyo.Objective(expr = model.track_cost + model.terminal_cost, sense=pyo.minimize)

    # # Initial condition
    # model.constraint1 = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[i, 0] == z0[i])

    # Dynamic constraints
    model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[0, t+1] == model.x[0, t] + Ts* (pyo.cos(model.x[2, t]) *model.u[0, t] * gam)
                                    if t < N else pyo.Constraint.Skip)
    model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[1, t+1] == model.x[1, t] + Ts* (pyo.sin(model.x[2, t]) *model.u[0, t] * gam)
                                    if t < N else pyo.Constraint.Skip)
    model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[2, t+1] == model.x[2, t] + Ts* model.u[1, t] * tau
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
    # #model.constraint10 = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[1, t] <= 0.4
    # #                                   if t <= N-1 else pyo.Constraint.Skip)

    # State bounds:  xL <= x[i,t] <= xU  for all i in nx, t in tidx
    # Limits are vectors
    model.x_upper = pyo.Constraint(model.nx, model.tidx,
        rule=lambda m, i, t: m.x[i, t] <= xU[i]
    )
    model.x_lower = pyo.Constraint(model.nx, model.tidx,
        rule=lambda m, i, t: m.x[i, t] >= xL[i]
    )
    # Control bounds: uL <= u[i,t] <= uU for all i in nu, t in tidx
    model.u_upper = pyo.Constraint(model.nu, model.tidu,
        rule=lambda m, i, t: m.u[i, t] <= uU[i]
    )
    model.u_lower = pyo.Constraint(model.nu, model.tidu,
        rule=lambda m, i, t: m.u[i, t] >= uL[i]
    )
    
    # Initial Constraints
    model.x0_con = pyo.Constraint(model.nx, rule= lambda m, i: m.x[i, 0] == x0[i])
    
    # Terminal constraint
    Af_list = Af.tolist() if hasattr(Af, "tolist") else Af
    if len(Af_list) == 0:   # set this from Af.shape or len(Af_list)
        model.xf_con = pyo.Constraint(model.nx, rule=lambda m,i: m.x[i,N] == bf[i])
    else:
        ncon = len(Af_list)
        model.nxf = pyo.RangeSet(0, ncon-1)
        bf_list = bf.tolist() if hasattr(bf, "tolist") else bf
        model.xf_con = pyo.Constraint(
            model.nxf,
            rule=lambda m, r: sum(Af_list[r][j]*m.x[j, N] for j in m.nx) <= bf_list[r]
        )

    # # Terminal constraint
    # model.constraint9 = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[i, N] == bf[i])

    # Initial guess to make solver run faster for nonlinear dynamics
    for t in model.tidx:
        model.x[0,t].set_value(x0[0])
        model.x[1,t].set_value(x0[1])
        model.x[2,t].set_value(x0[2])
    for t in model.tidu:
        model.u[0,t].set_value(0.0)
        model.u[1,t].set_value(0.0)
    
   
    # Now we can solve:
    solver = pyo.SolverFactory('ipopt')
    # Solver options
    solver.options["max_iter"] = 2000
    solver.options["tol"] = 1e-6
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
    nz = z.shape[0]
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

def traj_mpc(model, data, path, dt=0.05) -> tuple[float, float] :
    '''
    This is a controller that uses MPC to optimize trajector control
    given desired waypoints

    :param model: mujoco model
    :param data: mujoco data
    :return: control inputs for drive and turn commands
    :rtype: tuple[float, float]
    '''

    # Simulation params
    # STILL NEED TO FIGURE OUT OUT TO SIMULATE OVER A REDUCED HORIZON SIZE (NOT EVERY .001S)
    # AND HOW TO CONVERT DISTANCE TO TIME
    Ts = dt                 # time step - every time the controller is called
    N = 10                  # Horizon
    TFinal = 1              # picked to be 1 second. Increase if myopic. Decrease if computation is slow


    # Model params
    gam = 1     # drive param
    tau = 0.5   # turn param

    # Initial constraint
    z0 = _get_car_pose(model, data)

    # Terminal condition
    target_idx = int(data.userdata[1])
    xt, yt, _ = path[target_idx]    # target waypoint
    xt_next, yt_next, _ = path[target_idx + 1] if target_idx < len(path) else [xt,yt,0]     # NEXT target waypoint
    yaw_des = np.arctan2(yt_next - yt,
                        xt_next - xt)
    zf = [xt,yt,yaw_des]
    Af = []

    # Boundaries
    xL = [-1000, -1000, -np.pi]
    xU = [1000, 1000, np.pi]
    uL = [0, -1]     # 0.5 to limit linear speed
    uU = [0.5, 1]

    feas, xOpt, uOpt, JOpt = _solve_cftoc_disc(gam, tau, Ts, N, z0, xL, xU, uL, uU, zf, Af)
    print("feasibilty = ", feas)
    print("optimal states = ", xOpt)
    print("optimal inputs = ", uOpt)
    
    u1, u2 = uOpt[:,0]

    return u1, u2
