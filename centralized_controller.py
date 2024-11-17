import numpy as np
import struct
import math
import casadi as ca
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D

def attitude_jacobian_bar(phi, theta, psi):
  H = ca.MX(3, 3) 
  H[0, 0] = ca.cos(psi) / ca.cos(theta)
  H[0, 1] = ca.sin(psi) / ca.cos(theta)
  H[0, 2] = 0
  H[1, 0] = -ca.sin(psi)
  H[1, 1] = ca.cos(psi)
  H[1, 2] = 0
  H[2, 0] = ca.cos(psi) * ca.tan(theta)
  H[2, 1] = ca.sin(psi) * ca.tan(theta)
  H[2, 2] = 1
  return H

def attitude_jacobian(phi, theta, psi):
  H = ca.MX(3, 3) 
  H[0, 0] = 1
  H[0, 1] = ca.sin(phi) * ca.tan(theta)
  H[0, 2] = ca.cos(phi) * ca.tan(theta)
  H[1, 0] = 0
  H[1, 1] = ca.cos(phi)
  H[1, 2] = -ca.sin(phi)
  H[2, 0] = 0
  H[2, 1] = ca.sin(phi) / ca.cos(theta)
  H[2, 2] = ca.cos(phi) / ca.cos(theta)

  return H

def rotation_matrix(phi, theta, psi):
  R_x = ca.MX(3, 3)
  R_x[0, 0] = 1
  R_x[0, 1] = 0
  R_x[0, 2] = 0
  R_x[1, 0] = 0
  R_x[1, 1] = ca.cos(phi)
  R_x[1, 2] = -ca.sin(phi)
  R_x[2, 0] = 0
  R_x[2, 1] = ca.sin(phi)
  R_x[2, 2] = ca.cos(phi)

  R_y = ca.MX(3, 3)
  R_y[0, 0] = ca.cos(theta)
  R_y[0, 1] = 0
  R_y[0, 2] = ca.sin(theta)
  R_y[1, 0] = 0
  R_y[1, 1] = 1
  R_y[1, 2] = 0
  R_y[2, 0] = -ca.sin(theta)
  R_y[2, 1] = 0
  R_y[2, 2] = ca.cos(theta)

  R_z = ca.MX(3, 3)
  R_z[0, 0] = ca.cos(psi)
  R_z[0, 1] = -ca.sin(psi)
  R_z[0, 2] = 0
  R_z[1, 0] = ca.sin(psi)
  R_z[1, 1] = ca.cos(psi)
  R_z[1, 2] = 0
  R_z[2, 0] = 0
  R_z[2, 1] = 0
  R_z[2, 2] = 1

  R = ca.mtimes(R_z, ca.mtimes(R_y, R_x))
  return R

# Quadrotor parameters
mass = 1.75
L = 0.238537
J = ca.DM([0.01252, 0.01743, 0.0238])
J = ca.diag(J)
kf = 8.54858e-06
km = 1.6e-2

# Bar parameters
Lb = 1.5      
d1 = Lb / 2
d2 = Lb / 2      
massb = 0.5
Jb = massb*Lb*Lb / 12        

# Cable parameters
Lc = 0.6
l1 = Lc
l2 = Lc

g = 9.81
M = 2 * mass + massb

Q_uav = np.array([30,30,30,30,30,30,10,10,20,10,10,100]) 
Qb = np.array([50,50,50,50,50,50,10,10,10,1,1,5]) 
Q = np.diag(np.hstack([Qb, Q_uav, Q_uav]))
Qn = 10000*Q
R = np.diag(np.array([1,100,100,50,1,100,100,50]))


def dynamicsC(x, u):
    # State variables
    pb = x[0:3].reshape((3, 1))
    vb = x[3:6].reshape((3, 1))
    thetab = x[6:9].reshape((3, 1))
    omegab = x[9:12].reshape((3, 1))
    Hb = attitude_jacobian_bar(x[6], x[7], x[8])

    p1 = x[12:15].reshape((3, 1))
    v1 = x[15:18].reshape((3, 1))
    theta1 = x[18:21].reshape((3, 1))
    omega1 = x[21:24].reshape((3, 1))
    R1 = rotation_matrix(x[18], x[19], x[20])
    H1 = attitude_jacobian(x[18], x[19], x[20])

    p2 = x[24:27].reshape((3, 1))
    v2 = x[27:30].reshape((3, 1))
    theta2 = x[30:33].reshape((3, 1))
    omega2 = x[33:36].reshape((3, 1))
    R2 = rotation_matrix(x[30], x[31], x[32])
    H2 = attitude_jacobian(x[30], x[31], x[32])

    nb = (p2 - p1) / ca.norm_2(p2 - p1)
    n1 = (p1 - pb + d1 * nb) / l1
    n2 = (p2 - pb - d2 * nb) / l2

    # d1/d2 for tension calculation
    d1_ = -d1
    d2_ = d2

    S_w_n = ca.cross(omegab, nb)
    a = ca.cross(nb, n1)
    b = ca.cross(nb, n2)

    M_z = ca.MX(2,2)
    M_z[0, 0] = massb / mass
    M_z[0, 1] = 0
    M_z[1, 0] = 0
    M_z[1, 1] = massb / mass

    M_z[0, 0] += ca.mtimes(n1.T, n1)
    M_z[0, 1] += ca.mtimes(n1.T, n2)
    M_z[1, 0] += ca.mtimes(n1.T, n2)
    M_z[1, 1] += ca.mtimes(n2.T, n2)

    M_z[0, 0] += (massb * d1_ * d2 / Jb) * ((d1_ / d2) * ca.norm_2(a) ** 2)
    M_z[0, 1] += (massb * d1_ * d2 / Jb) * (ca.mtimes(a.T, b))
    M_z[1, 0] += (massb * d1_ * d2 / Jb) * (ca.mtimes(a.T, b))
    M_z[1, 1] += (massb * d1_ * d2 / Jb) * ((d2 / d1_) * ca.norm_2(b) ** 2)

    T_z11 = ca.MX(2,6)
    T_z11[0, 0] = ((massb / mass) * n1.T)[0]
    T_z11[0, 1] = ((massb / mass) * n1.T)[1]
    T_z11[0, 2] = ((massb / mass) * n1.T)[2]
    T_z11[0, 3] = 0
    T_z11[0, 4] = 0
    T_z11[0, 5] = 0
    T_z11[1, 0] = 0
    T_z11[1, 1] = 0
    T_z11[1, 2] = 0
    T_z11[1, 3] = ((massb / mass) * n2.T)[0]
    T_z11[1, 4] = ((massb / mass) * n2.T)[1]
    T_z11[1, 5] = ((massb / mass) * n2.T)[2]

    T_z12 = ca.MX(6,1)
    T_z12[0, 0] = 0
    T_z12[1, 0] = 0
    T_z12[2, 0] = u[0]
    T_z12[3, 0] = 0
    T_z12[4, 0] = 0
    T_z12[5, 0] = u[4]

    T_z1 = ca.mtimes(T_z11, T_z12)

    T_z2 = ca.MX(2,1)
    T_z2[0, 0] = massb * ca.norm_2(v1 - (vb + d1_ * S_w_n)) ** 2 / l1
    T_z2[1, 0] = massb * ca.norm_2(v2 - (vb + d2 * S_w_n)) ** 2 / l2

    T_z3 = ca.MX(2,1)
    T_z3[0, 0] = massb * ca.norm_2(omegab) ** 2 * d1_ * ca.mtimes(nb.T, n1)
    T_z3[1, 0] = massb * ca.norm_2(omegab) ** 2 * d2_ * ca.mtimes(nb.T, n2)
    
    T_z = T_z1 + T_z2 + T_z3

    T12 = ca.solve(M_z, T_z)
    T1 = T12[0]
    T2 = T12[1]

    # Control inputs
    F1 = u[0]
    tau1 = ca.vertcat(u[1], u[2], u[3])
    F2 = u[4]
    tau2 = ca.vertcat(u[5], u[6], u[7])

    # Dynamics
    pb_d = vb
    vb_d = (T1 * n1 + T2 * n2 - massb * g * ca.DM([0, 0, 1])) / massb
    thetab_d = Hb @ omegab
    cross_pr_ = ca.cross(nb, (d2 * T2 * n2 - d1 * T1 * n1))
    omegab_d = ca.solve(ca.diag(ca.DM([1e-06, 1e-06, Jb])), cross_pr_)

    p1_d = v1
    v1_d = (F1 * R1 @ ca.DM([0, 0, 1]) - mass * g * ca.DM([0, 0, 1]) - T1 * n1) / mass
    theta1_d = H1 @ omega1
    cross_pr = ca.cross(omega1, J @ omega1)
    omega1_d = ca.solve(J, tau1 - cross_pr)

    p2_d = v2
    v2_d = (F2 * R2 @ ca.DM([0, 0, 1]) - mass * g * ca.DM([0, 0, 1]) - T2 * n2) / mass
    theta2_d = H2 @ omega2
    cross_pr = ca.cross(omega2, J @ omega2)
    omega2_d = ca.solve(J, tau2 - cross_pr)

    ret = ca.vertcat(pb_d, vb_d, thetab_d, omegab_d, p1_d, v1_d, theta1_d, omega1_d, p2_d, v2_d, theta2_d, omega2_d)
    return ret


def get_eqb(pb = [0, 0, 2], yawb = 0, yaw1 = 0, yaw2 = 0, thetaf = 0, thetal = 0):
  # print(thetaf, thetal)
  p1 = np.array(pb) + np.array([(d1 + l1*ca.sin(thetal))*ca.cos(yawb), (- d1 - l1*ca.sin(thetal))*ca.sin(yawb), l1*ca.cos(thetal)])
  p2 = np.array(pb) + np.array([(-d2 - l2*ca.sin(thetal))*ca.cos(yawb), (d2 + l2*ca.sin(thetal))*ca.sin(yawb), l2*ca.cos(thetal)])

  roll1 = 0
  roll2 = 0
  pitch1 = thetaf
  pitch2 = -thetaf

  x1 = np.hstack((p1, [0, 0, 0, roll1, pitch1, yaw1, 0, 0, 0]))
  x2 = np.hstack((p2, [0, 0, 0, roll2, pitch2, yaw2, 0, 0, 0]))
  xb = np.hstack((pb, [0, 0, 0, 0, 0, yawb, 0, 0, 0]))

  M = 2*mass + massb
  F = g*M / (2*ca.cos(thetaf)) # eqb thrust

  x = np.hstack((xb, x1, x2))
  u = np.array([F, 0, 0, 0, F, 0, 0, 0]).reshape(8,)

  return np.array(x), np.array(u)


def equations(vars):
    thetal, thetaf = vars
    T_expr = (massb * g * math.cos(thetaf - thetal) * (massb + 2 * mass)) / (
        (2 * math.cos(thetaf) * (2 * mass * math.cos(thetal) ** 2 + massb))
    )
    F_expr = g * M / (2 * math.cos(thetaf))
    
    e1 = F_expr * math.sin(thetaf) - T_expr * math.sin(thetal)
    e2 = T_expr * math.cos(thetal) - massb * g / 2
    return [e1, e2]

def get_thetas(initial_guess = [0.1, 0.1]):
    solution = fsolve(equations, initial_guess)

    thetal_solution, thetaf_solution = solution

    F = g * M / (2 * math.cos(thetaf_solution))
    T = (massb * g * math.cos(thetaf_solution - thetal_solution) * (massb + 2 * mass)) / ((2 * math.cos(thetaf_solution) * (2 * mass * math.cos(thetal_solution) ** 2 + massb)))
    eq1 = F * math.sin(thetaf_solution) - T * math.sin(thetal_solution)
    eq2 = T * math.cos(thetal_solution) - massb * g / 2

    assert eq1 < 1e-08 and eq2 < 1e-08, "solution found does not solve the system of equations!"
    return thetal_solution, thetaf_solution

def linearize(f, x, x0):
  x_lim = ca.MX.sym('x_lim', x.sparsity())
  return ca.substitute(f + ca.jtimes(f, x, x_lim - x0), ca.vertcat(x_lim, x), ca.vertcat(x, x0))

def controller(A, B, fnext, x0, u0, xinit, uinit, dt):
  nx = 36
  nu  = 8 

  N = 30
  
  opti = ca.Opti()
  X = opti.variable(nx, N+1)
  U = opti.variable(nu, N)
  p = opti.parameter(nx, 1)

  # Dynamic constraints and input bounds

  opti.subject_to(X[:, 0] == p)

  for k in range(N):

    opti.subject_to(X[:, k+1] == fnext(X[:, k], U[:, k]))
    
    opti.subject_to(U[0, k] >= 0)
    opti.subject_to(U[1:4, k] <= 1)
    opti.subject_to(U[1:4, k] >= -1)
    opti.subject_to(U[4, k] >= 0)
    opti.subject_to(U[5:8, k] <= 1)
    opti.subject_to(U[5:8, k] >= -1)

  

  # Cost function
  cost = 0
  for k in range(N):
      # Stage costs with regularization to avoid numerical issues
      state_error = X[:, k] - x0
      control_error = U[:, k] - u0
      state_cost = state_error.T @ Q @ state_error
      control_cost = control_error.T @ R @ control_error
      cost += state_cost + control_cost

  # Terminal cost
  terminal_error = X[:, N] - x0
  terminal_cost = terminal_error.T @ Qn @ terminal_error
  cost += terminal_cost

  opti.minimize(cost)

  # Set up solver with more robust options
  opts = {
      'ipopt.max_iter': 1000,
      'ipopt.tol': 1e-4,
      'ipopt.acceptable_tol': 1e-9,
      'ipopt.print_level': 5,  # Increased print level for debugging
      'print_time': 0,
      'ipopt.acceptable_obj_change_tol': 1e-12,
      'ipopt.check_derivatives_for_naninf': 'yes',
  }

  opti.solver('ipopt', opts)

  # Initialize decision variables to help convergence
  opti.set_initial(X, (np.array(xinit)).reshape((nx, 1)) @ np.ones((1, N+1)))  # Initialize states to x0
  opti.set_initial(U, (np.array(uinit)).reshape((nu, 1)) @ np.ones((1, N)))    # Initialize controls to u0

  # Set parameter value
  opti.set_value(p, (np.array(xinit)).reshape((nx, 1)))

  try:
      # Solve the optimization problem
      sol = opti.solve()
      
      # Extract solution
      objective_cost_value = sol.value(opti.f)
      print("Objective (Cost) Value:", objective_cost_value)

      X_sol = sol.value(X)
      U_sol = sol.value(U)
      # print(X_sol[:, 0])
      return U_sol[:, 0]
      
  except Exception as e:
      print("Optimization failed!")
      print("Error message:", str(e))
      
      # Debug information
      print("\nLast variable values:")
      print("X:", opti.debug.value(X))
      print("U:", opti.debug.value(U))
      print("Cost:", opti.debug.value(cost))
      
      # Check for any NaN or Inf in the matrices
      print("\nChecking matrices for NaN/Inf:")
      print("Q has NaN:", np.any(np.isnan(Q)))
      print("R has NaN:", np.any(np.isnan(R)))
      print("Qn has NaN:", np.any(np.isnan(Qn)))
      print("x0 has NaN:", np.any(np.isnan(x0)))
      print("u0 has NaN:", np.any(np.isnan(u0)))


if __name__ == "__main__":
  thetal, thetaf = get_thetas()
  print("Thetaf:", thetaf, "Thetal:", thetal)
  x0, u0 = get_eqb(pb = [1, 1, 2], thetal=thetal, thetaf=thetaf) #ref variables
  xinit, uinit = get_eqb(thetal=thetal, thetaf=thetaf) #initial variables
  # x0 = np.array(x0.tolist())
  # u0 = np.array(u0.tolist())
  
  # print("x0:", x0)
  # print("xinit:", xinit)

  nx = 36
  nu  = 8 
  dt = 0.3

  x = ca.MX.sym('x', nx, 1)
  u = ca.MX.sym('u', nu, 1)

  dx = x - x0
  du = u - u0
  J_x = ca.jacobian(dynamicsC(x, u), x)
  J_u = ca.jacobian(dynamicsC(x, u), u)
  
  Ac = ca.Function('fA', [x, u], [J_x])
  Bc = ca.Function('fB', [x, u], [J_u])
  
  A = Ac(x0+1e-20,u0-1e-20)
  B = Bc(x0+1e-20,u0-1e-20)

  xcurr = ca.MX.sym('xcurr', nx, 1)
  ucurr = ca.MX.sym('ucurr', nu, 1)
  f = dynamicsC(x0 +1e-20, u0 +1e-20) + ca.mtimes(A, xcurr - x0) + ca.mtimes(B, ucurr - u0)
  F = ca.Function('f', [xcurr, ucurr], [f])

  k1 = F(xcurr, ucurr)
  k2 = F(xcurr + dt/2 * k1, ucurr) 
  k3 = F(xcurr + dt/2 * k2, ucurr)
  k4 = F(xcurr + dt * k3, ucurr)
  xnext = xcurr + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
  fnext = ca.Function('fnext', [xcurr, ucurr], [xnext])
  
  N_sim = 60
  trajectory_pb = []
  trajectory_p1 = []
  trajectory_p2 = []

  for t in range(N_sim):
    xprev = xinit
    res = controller(A, B, fnext, x0, u0, xinit, uinit, dt) #get the first control input
    print(res)

    xinit = np.array(fnext(xinit, res)).reshape((nx, 1))

    state_names = ['pb', 'vb', 'Θb', 'ωb', 'p1', 'v1', 'Θ1', 'ω1', 'p2', 'v2', 'Θ2', 'ω2']
    print(f"{'State':<10} {'Initial':<70} {'Final':<70}")
    print("-" * 150)
    for i, name in enumerate(state_names):
        val0 = xprev[i*3:i*3+3]
        val_res = xinit[i*3:i*3+3]
        print(f"{name:<10} ({val0[0]}, {val0[1]}, {val0[2]}){' ' * (70 - len(f'({val0[0]}, {val0[1]}, {val0[2]})'))} ({val_res[0]}, {val_res[1]}, {val_res[2]})")
    
    # Store xinit values in arrays
    trajectory_pb.append(xinit[0:3].flatten())
    trajectory_p1.append(xinit[12:15].flatten())
    trajectory_p2.append(xinit[24:27].flatten())

  if N_sim > 1:
    trajectory_pb = np.array(trajectory_pb)
    trajectory_p1 = np.array(trajectory_p1)
    trajectory_p2 = np.array(trajectory_p2)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_pb[:, 0], trajectory_pb[:, 1], trajectory_pb[:, 2], label='pb')
    ax.plot(trajectory_p1[:, 0], trajectory_p1[:, 1], trajectory_p1[:, 2], label='p1')
    ax.plot(trajectory_p2[:, 0], trajectory_p2[:, 1], trajectory_p2[:, 2], label='p2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Position Trajectories')
    ax.legend()

    # Draw the bar and cables
    for i in range(N_sim):
      pb = trajectory_pb[i]
      p1 = trajectory_p1[i]
      p2 = trajectory_p2[i]

      # Calculate the ends of the bar
      bar_end1 = pb + 0.75 * (p1 - p2) / np.linalg.norm(p1 - p2)
      bar_end2 = pb + 0.75 * (p2 - p1) / np.linalg.norm(p2 - p1)

      # Draw the bar
      ax.plot([bar_end1[0], bar_end2[0]], [bar_end1[1], bar_end2[1]], [bar_end1[2], bar_end2[2]], 'k-', linewidth=0.5)

      # Draw the cables
      ax.plot([bar_end1[0], p1[0]], [bar_end1[1], p1[1]], [bar_end1[2], p1[2]], 'k--', linewidth=0.5)
      ax.plot([bar_end2[0], p2[0]], [bar_end2[1], p2[1]], [bar_end2[2], p2[2]], 'k--', linewidth=0.5)

    plt.show()
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    labels = ['X', 'Y', 'Z']
    for i in range(3):
      axs[0, i].plot(trajectory_pb[:, i], label='pb')
      axs[0, i].plot(np.ones(N_sim) * x0[i], label='x0[pb]')
      axs[0, i].set_title(f'pb vs x0[pb] - {labels[i]}')
      axs[0, i].legend()

      axs[1, i].plot(trajectory_p1[:, i], label='p1')
      axs[1, i].plot(np.ones(N_sim) * x0[12 + i], label='x0[p1]')
      axs[1, i].set_title(f'p1 vs x0[p1] - {labels[i]}')
      axs[1, i].legend()

      axs[2, i].plot(trajectory_p2[:, i], label='p2')
      axs[2, i].plot(np.ones(N_sim) * x0[24 + i], label='x0[p2]')
      axs[2, i].set_title(f'p2 vs x0[p2] - {labels[i]}')
      axs[2, i].legend()

    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_pb[:, 0], trajectory_pb[:, 1], trajectory_pb[:, 2], label='pb')
    ax.plot(trajectory_p1[:, 0], trajectory_p1[:, 1], trajectory_p1[:, 2], label='p1')
    ax.plot(trajectory_p2[:, 0], trajectory_p2[:, 1], trajectory_p2[:, 2], label='p2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Position Trajectories without Cable/Bar Lines')
    ax.legend()
    plt.show()