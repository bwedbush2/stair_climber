# Finite constrained optimization problem
import numpy as np
import pyomo.environ as pyo
import mujoco

# Euler discretized unicycle model
# x is state vector, u is input, Ts is sampling period
def fdis(z, u, Ts):
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

    nx = 3         # number of states (x,y,theta)
    nu = 2         # number of inputs u1=v, u2=omega

    model = pyo.ConcreteModel()
    model.tidx = pyo.Set(initialize=range(0, N+1)) # length of finite optimization problem
    model.xidx = pyo.Set(initialize=range(0, nx))
    model.uidx = pyo.Set(initialize=range(0, nu))

    # Create state and input variables trajectory:
    model.z = pyo.Var(model.xidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)

    # Constraints:
    z0 = _get_car_pose(model, data)

    target_idx = data.userdata[1]
    xt, yt, _ = path[target_idx]    # target waypoint
    # NEXT target waypoint
    xt_next, yt_next, _ = path[target_idx + 1] if target_idx < len(path) else [xt,yt,0]
    #yawt = 
    zf=[xt,yt,np.pi/4]

    # Objective:
    model.shortest_time_cost  = sum((model.z[1, t]-zf[1])**2 for t in model.tidx if t < N)
    model.turning_cost = sum((model.u[1, t])**2 for t in model.tidx if t < N)
    model.speed_cost = sum((model.u[0, t])**2 for t in model.tidx if t < N)


    model.cost = pyo.Objective(expr = model.speed_cost, sense=pyo.minimize)


    model.constraint1 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i, 0] == z0[i])
    model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] == model.z[0, t] + Ts* (pyo.cos(model.z[2, t]) *model.u[0, t])
                                    if t < N else pyo.Constraint.Skip)
    model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] == model.z[1, t] + Ts* (pyo.sin(model.z[2, t]) *model.u[0, t])
                                    if t < N else pyo.Constraint.Skip)
    model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] == model.z[2, t] + Ts* model.u[1, t]
                                    if t < N else pyo.Constraint.Skip)
    model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= 1
                                    if t <= N-1 else pyo.Constraint.Skip)
    model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >=-1
                                    if t <= N-1 else pyo.Constraint.Skip)
    model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= 1
                                    if t <= N-1 else pyo.Constraint.Skip)
    model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= -1
                                    if t <= N-1 else pyo.Constraint.Skip)
    model.constraint9 = pyo.Constraint(model.xidx, rule=lambda model, i: model.z[i, N] == zf[i])

    #model.constraint10 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t] <= 0.4
    #                                   if t <= N-1 else pyo.Constraint.Skip)

    # Now we can solve:
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)

    u1 = [pyo.value(model.u[0,0])]
    u2 = [pyo.value(model.u[1,0])]

    # num_vars = pyo.value(model.nvariables())
    # num_cons = pyo.value(model.nconstraints())

    # print("Total variables:", num_vars)
    # print("Total constraints:", num_cons)

    return (u1,u2)
