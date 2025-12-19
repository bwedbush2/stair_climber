# Finite constrained optimization problem
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import mujoco

_MPC_WARM = {"x": None, "u": None}

def _report_constraint_violations(model, top_k=10, tol=1e-6):
    """
    Print the most violated constraints in the current Pyomo model state.
    Works after solve() even if infeasible.
    """
    viols = []

    for c in model.component_data_objects(pyo.Constraint, active=True):
        if c.body is None:
            continue

        try:
            body = pyo.value(c.body)
        except Exception:
            continue

        lb = pyo.value(c.lower) if c.has_lb() else None
        ub = pyo.value(c.upper) if c.has_ub() else None

        v = 0.0
        if lb is not None and body < lb - tol:
            v = lb - body
        if ub is not None and body > ub + tol:
            v = max(v, body - ub)

        if v > tol:
            viols.append((v, c.name, body, lb, ub))

    viols.sort(key=lambda x: x[0], reverse=True)

    if not viols:
        print("No constraint violations found above tol.")
        return

    print(f"\nTop {min(top_k, len(viols))} constraint violations (tol={tol}):")
    for v, name, body, lb, ub in viols[:top_k]:
        print(f"  {name}: viol={v:.3e}, body={body:.6g}, lb={lb}, ub={ub}")


def _solve_cftoc_lin(gam, tau, Ts, N, x0, xN, xL, xU, uL, uU, bf, Af,
                     xbar=None, ubar=None):
    """
    Solve CFTOC with *linearized* (affine) unicycle dynamics about (xbar, ubar).

    x = [x, y, yaw]
    u = [v, w]

    Dynamics enforced:
        x_{t+1} = A_t x_t + B_t u_t + c_t

    If xbar/ubar are None, we default to a simple nominal:
      - xbar is constant at x0
      - ubar is zeros
    (Better: pass your warm-start trajectory as xbar/ubar each MPC step.)
    """

    # -------------------------
    # Initialize model (same)
    # -------------------------
    model = pyo.ConcreteModel()
    nx = np.size(x0)
    nu = np.size(uU)
    model.tidx = pyo.Set(initialize=range(0, N+1))
    model.tidu = pyo.Set(initialize=range(0, N))
    model.nx = pyo.Set(initialize=range(0, nx))
    model.nu = pyo.Set(initialize=range(0, nu))

    model.x = pyo.Var(model.nx, model.tidx)
    model.u = pyo.Var(model.nu, model.tidu)

    # -------------------------
    # Objective (unchanged)
    # -------------------------
    w_xy  = 20.0
    w_yaw = 5.0
    w_v   = 0.5
    w_w   = 0.1

    w_xy_T  = 200.0
    w_yaw_T = 50.0

    # desired stage yaw based on current position
    yaw_desired = np.arctan2(xN[1]-x0[1], xN[0]-x0[0])
    model.track_cost = sum(
        w_xy * ((model.x[0, t] - xN[0])**2 + (model.x[1, t] - xN[1])**2)
        + w_yaw * (1 - pyo.cos(model.x[2, t] - yaw_desired))
        for t in model.tidu
    )
    model.terminal_cost = (
        w_xy_T * ((model.x[0, N] - xN[0])**2 + (model.x[1, N] - xN[1])**2)
        + w_yaw_T * (1 - pyo.cos(model.x[2, N] - xN[2]))
    )
    model.turning_cost = sum(w_w * (model.u[1, t]**2) for t in model.tidu)
    model.speed_cost   = sum(w_v * (model.u[0, t]**2) for t in model.tidu)

    total_cost = model.track_cost + model.terminal_cost + model.speed_cost + model.turning_cost
    model.cost = pyo.Objective(expr=total_cost, sense=pyo.minimize)

    # -------------------------
    # Linearization data prep  (NEW)
    # -------------------------
    # Nominal trajectory defaults (if none provided)
    if xbar is None:
        xbar = np.zeros((nx, N+1), dtype=float)
        for t in range(N+1):
            xbar[:, t] = np.array(x0, dtype=float).reshape(-1)
    if ubar is None:
        ubar = np.zeros((nu, N), dtype=float)

    xbar = np.asarray(xbar, dtype=float)
    ubar = np.asarray(ubar, dtype=float)

    assert xbar.shape == (nx, N+1), f"xbar must be shape {(nx, N+1)}, got {xbar.shape}"
    assert ubar.shape == (nu, N),   f"ubar must be shape {(nu, N)}, got {ubar.shape}"

    # Precompute A_t, B_t, c_t for t=0..N-1
    A_list = []
    B_list = []
    c_list = []

    I = np.eye(nx)

    for t in range(N):
        thb = xbar[2, t]
        vb  = ubar[0, t]
        wb  = ubar[1, t]  # not needed for Jacobian except yaw dot is linear in w anyway

        # Continuous-time Jacobians about (xbar_t, ubar_t)
        # xdot = gam*cos(th)*v
        # ydot = gam*sin(th)*v
        # thdot = tau*w
        A_c = np.array([
            [0.0, 0.0, -gam * vb * np.sin(thb)],
            [0.0, 0.0,  gam * vb * np.cos(thb)],
            [0.0, 0.0,  0.0]
        ], dtype=float)

        B_c = np.array([
            [gam * np.cos(thb), 0.0],
            [gam * np.sin(thb), 0.0],
            [0.0,              tau]
        ], dtype=float)

        # Euler discretization of the *linearized error dynamics*
        # δx_{t+1} = (I + Ts A_c) δx_t + (Ts B_c) δu_t
        A_d = I + Ts * A_c
        B_d = Ts * B_c

        # Build affine term so the linear model matches the nominal step exactly:
        # x_{t+1} ≈ A_d x_t + B_d u_t + c_t
        # c_t = xbar_{t+1} - A_d xbar_t - B_d ubar_t
        # where xbar_{t+1} = xbar_t + Ts * f(xbar_t, ubar_t)
        fbar = np.array([
            gam * np.cos(thb) * vb,
            gam * np.sin(thb) * vb,
            tau * wb
        ], dtype=float)

        xbar_next = xbar[:, t] + Ts * fbar
        c_t = xbar_next - A_d @ xbar[:, t] - B_d @ ubar[:, t]

        A_list.append(A_d)
        B_list.append(B_d)
        c_list.append(c_t)

    # -------------------------
    # Dynamics constraints (CHANGED)
    # -------------------------
    def dyn_rule(m, i, t):
        if t >= N:
            return pyo.Constraint.Skip

        A_t = A_list[t]
        B_t = B_list[t]
        c_t = c_list[t]

        return m.x[i, t+1] == (
            sum(A_t[i, j] * m.x[j, t] for j in m.nx)
            + sum(B_t[i, k] * m.u[k, t] for k in m.nu)
            + float(c_t[i])
        )

    model.dyn_con = pyo.Constraint(model.nx, model.tidx, rule=dyn_rule)

    # -------------------------
    # Bounds / initial / terminal (unchanged)
    # -------------------------
    model.x_upper = pyo.Constraint(model.nx, model.tidx,
        rule=lambda m, i, t: m.x[i, t] <= xU[i]
    )
    model.x_lower = pyo.Constraint(model.nx, model.tidx,
        rule=lambda m, i, t: m.x[i, t] >= xL[i]
    )
    model.u_upper = pyo.Constraint(model.nu, model.tidu,
        rule=lambda m, i, t: m.u[i, t] <= uU[i]
    )
    model.u_lower = pyo.Constraint(model.nu, model.tidu,
        rule=lambda m, i, t: m.u[i, t] >= uL[i]
    )

    model.x0_con = pyo.Constraint(model.nx, rule=lambda m, i: m.x[i, 0] == x0[i])

    Af_list = Af.tolist() if hasattr(Af, "tolist") else Af
    if len(Af_list) == 0:
        model.xf_con = pyo.Constraint(model.nx, rule=lambda m, i: m.x[i, N] == bf[i])
    else:
        ncon = len(Af_list)
        model.nxf = pyo.RangeSet(0, ncon-1)
        bf_list = bf.tolist() if hasattr(bf, "tolist") else bf
        model.xf_con = pyo.Constraint(
            model.nxf,
            rule=lambda m, r: sum(Af_list[r][j] * m.x[j, N] for j in m.nx) <= bf_list[r]
        )

    # Optional: initialize with nominal (helps solver even though it's affine)
    for i in model.nx:
        for t in model.tidx:
            model.x[i, t].set_value(float(xbar[i, t]))
        model.x[i, 0].set_value(float(x0[i]))
    for i in model.nu:
        for t in model.tidu:
            model.u[i, t].set_value(float(ubar[i, t]))

    # -------------------------
    # Solve (unchanged)
    # -------------------------
    solver = pyo.SolverFactory('ipopt')
    solver.options["max_iter"] = 1000
    solver.options["tol"] = 1e-6

    results = solver.solve(model)
    print("IPOPT status:", results.solver.status)
    print("IPOPT term:", results.solver.termination_condition)

    feas = (
        results.solver.status == SolverStatus.ok and
        results.solver.termination_condition in (
            TerminationCondition.optimal,
            TerminationCondition.feasible
        )
    )

    if not feas:
        _report_constraint_violations(model, top_k=15, tol=1e-6)

    xOpt = np.zeros((nx, N+1))
    for i in model.nx:
        for t in model.tidx:
            xOpt[i, t] = pyo.value(model.x[i, t])

    uOpt = np.zeros((nu, N))
    for i in model.nu:
        for t in model.tidu:
            uOpt[i, t] = pyo.value(model.u[i, t])

    JOpt = pyo.value(model.cost)
    return [feas, xOpt, uOpt, JOpt]


def _solve_cftoc_disc(gam, tau, Ts, N, x0, xN, xL, xU, uL, uU, bf, Af, xbar, ubar):

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
    w_v   = 0.5
    w_w   = 0.1

    w_xy_T  = 200.0
    w_yaw_T = 50.0

    model.track_cost = sum(
        w_xy * ((model.x[0, t] - xN[0])**2 + (model.x[1, t] - xN[1])**2)
        + w_yaw * (1 - pyo.cos(model.x[2, t] - xN[2]))
        for t in model.tidu
    )
    model.terminal_cost = (
        w_xy_T * ((model.x[0, N] - xN[0])**2 + (model.x[1, N] - xN[1])**2)
        + w_yaw_T * (1 - pyo.cos(model.x[2, N] - xN[2]))
    )
    model.turning_cost = sum(w_w * (model.u[1, t]**2) for t in model.tidu)
    model.speed_cost   = sum(w_v * (model.u[0, t]**2) for t in model.tidu)
    total_cost = model.track_cost + model.terminal_cost + model.speed_cost + model.turning_cost
    model.cost = pyo.Objective(expr = total_cost, sense=pyo.minimize)

    # # Initial condition
    # model.constraint1 = pyo.Constraint(model.xidx, rule=lambda model, i: model.x[i, 0] == z0[i])

    # Dynamic constraints
    model.dyn_con_x = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[0, t+1] == model.x[0, t] + Ts* (pyo.cos(model.x[2, t]) *model.u[0, t] * gam)
                                    if t < N else pyo.Constraint.Skip)
    model.dyn_con_y = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[1, t+1] == model.x[1, t] + Ts* (pyo.sin(model.x[2, t]) *model.u[0, t] * gam)
                                    if t < N else pyo.Constraint.Skip)
    model.dyn_con_th = pyo.Constraint(model.tidx, rule=lambda model, t: model.x[2, t+1] == model.x[2, t] + Ts* model.u[1, t] * tau
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

    # Optional: initialize with nominal (helps solver even though it's affine)
    for i in model.nx:
        for t in model.tidx:
            model.x[i, t].set_value(float(xbar[i, t]))
        model.x[i, 0].set_value(float(x0[i]))
    for i in model.nu:
        for t in model.tidu:
            model.u[i, t].set_value(float(ubar[i, t]))
   
    # Now we can solve:
    solver = pyo.SolverFactory('ipopt')
    # Solver options
    solver.options["max_iter"] = 1000
    solver.options["tol"] = 1e-6
    # solver.options["warm_start_init_point"] = "yes"
    # solver.options["print_level"] = 0
    # solver.options["max_iter"] = 60                  # 30–100 is typical for MPC
    # solver.options["tol"] = 1e-3                     # loosen
    # solver.options["acceptable_tol"] = 1e-2          # accept “good enough”
    # solver.options["acceptable_iter"] = 5
    # solver.options["max_cpu_time"] = 0.05            # seconds (match your dt_drive)

    results = solver.solve(model)
    print("IPOPT status:", results.solver.status)
    print("IPOPT term:", results.solver.termination_condition)


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
    if not feas:
        _report_constraint_violations(model, top_k=15, tol=1e-6)

    # extract values to return
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
    #Ts = 0.2
    N = 10                  # Horizon
    TFinal = Ts*N           # picked to be 1 second. Increase if myopic. Decrease if computation is slow


    # Model params
    gam = 1     # drive param
    tau = 1     # turn param

    # Initial constraint
    z0 = _get_car_pose(model, data)

    # Terminal condition
    target_idx = int(data.userdata[1])
    xt, yt, _ = path[target_idx]    # target waypoint
    xt_next, yt_next, _ = path[target_idx + 1] if target_idx < len(path)-1 else [xt,yt,0]     # NEXT target waypoint
    yawt = np.arctan2(yt_next - yt,
                        xt_next - xt)
    
    # terminal constraint point
    zf = np.array([xt, yt, yawt], dtype=float)

    # Terminal constraint box half-widths
    eps_xy  = 0.1   # meters
    eps_yaw = 0.05   # radians

    # Af * x_N <= bf encodes:
    #  x <= xt+eps_xy,  -x <= -xt+eps_xy
    #  y <= yt+eps_xy,  -y <= -yt+eps_xy
    #  yaw <= yaw_des+eps_yaw,  -yaw <= -yaw_des+eps_yaw
    Af = np.array([
        [ 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0,-1.0, 0.0],
        [ 0.0, 0.0, 1.0],
        [ 0.0, 0.0,-1.0],
    ], dtype=float)

    bf = np.array([
        xt + eps_xy,
        -xt + eps_xy,
        yt + eps_xy,
        -yt + eps_xy,
        yawt + eps_yaw,
        -yawt + eps_yaw,
    ], dtype=float)

    # Boundaries
    xL = [-1000, -1000, -np.pi]
    xU = [1000, 1000, np.pi]
    uL = [0, -1]
    uU = [0.5, 1]   # 0.5 to limit linear speed

    x_old = _MPC_WARM["x"] if _MPC_WARM["x"] is not None else np.zeros((3, N+1), dtype=float)
    u_old = _MPC_WARM["u"] if _MPC_WARM["u"] is not None else np.zeros((2, N), dtype=float)
    feas, xOpt, uOpt, JOpt = _solve_cftoc_lin(gam, tau, Ts, N, z0, zf, xL, xU, uL, uU, bf, Af, x_old, u_old)
    print(f"Current position: ({z0[0]}, {z0[1]}, {z0[2]})")
    print(f"Target position: ({xt}, {yt}, {yawt})")
    print("optimal states = ", xOpt)
    print("optimal inputs = ", uOpt)
    # after you solve and extract xOpt, uOpt:
    _MPC_WARM["x"] = xOpt
    _MPC_WARM["u"] = uOpt

    
    u1, u2 = uOpt[:,0]

    return u1, u2
