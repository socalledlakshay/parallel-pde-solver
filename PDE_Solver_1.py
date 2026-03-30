import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import time

start_time = time.perf_counter()

mpi_comm = MPI.COMM_WORLD
comm = PETSc.COMM_WORLD 
rank = mpi_comm.Get_rank()

# user input
if len(sys.argv) > 1:
    system = int(sys.argv[1])
    solver = int(sys.argv[2])
    boundary_condition = int(sys.argv[3])
    problem = int(sys.argv[4])
    xmin = float(sys.argv[5])
    xmax = float(sys.argv[6])
    final_time = float(sys.argv[7])
    number_of_cells = int(sys.argv[8])
    CFL = float(sys.argv[9])

N = number_of_cells
T = final_time
L = xmax - xmin

def f(u):
    if system == 1: return 1.0 * u
    elif system == 2: return 0.5 * u**2

def wave_speed(u):
    if system == 1: return 1.0
    elif system == 2: return np.abs(u)

def numerical_flux(uL, uR, dx, dt):
    fL = f(uL)  #n
    fR = f(uR)  #n+1

    # 1. Lax-Friedrichs
    if solver == 1:
        alpha = dx / dt
        return 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        
    # 2. Rusanov (Local Lax-Friedrichs)
    elif solver == 2:
        alpha = np.maximum(np.abs(wave_speed(uL)), np.abs(wave_speed(uR)))
        return 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)

    # 3. Godunov (Exact Riemann Solver)
    elif solver == 3:
        if system == 1:
            return fL # Upwind for a > 0
        elif system == 2: # Burgers
            if uL >= uR: # Shock
                s = 0.5 * (uL + uR)
                return fL if s >= 0 else fR
            else: # Rarefaction
                if uL >= 0: return fL
                elif uR <= 0: return fR
                else: return 0.0 # Sonic point (f(0) = 0)

    # 4. Roe Scheme
    elif solver == 4:
        if system == 1:
            return fL # Upwind
        elif system == 2: # Burgers
            if uL == uR:
                a_roe = wave_speed(uL)
            else:
                a_roe = (fR - fL) / (uR - uL) # Roe average speed
            return 0.5 * (fL + fR) - 0.5 * np.abs(a_roe) * (uR - uL)

    # 5. Engquist-Osher (EO)
    elif solver == 5:
        if system == 1:
            return fL # Upwind
        elif system == 2: # Burgers
            # F_EO = int_0^uL max(f'(s),0)ds + int_0^uR min(f'(s),0)ds + f(0)
            uL_plus = np.maximum(uL, 0.0)
            uR_minus = np.minimum(uR, 0.0)
            return 0.5 * (uL_plus**2) + 0.5 * (uR_minus**2)

    #6. Upwind
    elif solver == 6:
        if system == 1:
            # Linear Transport: Wave speed is a=1.0 (positive).
            # Information flows from left to right, so we take the Left flux.
            return fL 
        elif system == 2:
            # Burgers: Wave speed is f'(u) = u. 
            # We determine the "wind direction" using the average state.
            a_avg = 0.5 * (uL + uR)
            return fL if a_avg >= 0 else fR

comm = PETSc.COMM_WORLD
rank = comm.getRank()
dx = (xmax - xmin) / N

# Boundary Type Mapping
b_type = PETSc.DM.BoundaryType.PERIODIC if boundary_condition == 2 else PETSc.DM.BoundaryType.GHOSTED

da = PETSc.DMDA().create(dim=1, sizes=[N], stencil_width=1, 
                         boundary_type=b_type, comm=comm)

g_u = da.createGlobalVec()
l_u = da.createLocalVec()
g_unext = g_u.duplicate()

# Initial Condition (U_0)
(istart, iend), = da.getRanges()
x_local = xmin + (np.arange(istart, iend) + 0.5) * dx

if problem == 2:
    u0 = np.where(x_local < 0.5, 2.0, 1.0)
else:
    u0 = np.sin(2 * np.pi * x_local)

g_u.setValues(range(istart, iend), u0)
g_u.assemblyBegin()
g_u.assemblyEnd()

t = 0.0
while t < final_time:
    # 2. CFL CALCULATION FIX
    # Use the array directly for the local max
    da.globalToLocal(g_u, l_u)
    u_local = l_u.getArray()[1:-1]   # exclude ghost cells
    local_max_a = np.max(np.abs(wave_speed(u_local)))
    max_a = mpi_comm.allreduce(local_max_a, op=MPI.MAX)
    
    dt = CFL * dx / max(max_a, 1e-10)
    if t + dt > final_time: dt = final_time - t
    lam = dt / dx
    
    da.globalToLocal(g_u, l_u)
    u_ghost = l_u.getArray() 
    
    # 3. MANUALLY APPLY NEUMANN BCs (if not periodic)
    if boundary_condition == 1:
        # If we are the first process, set the left ghost to match the first cell
        if istart == 0:
            u_ghost[0] = u_ghost[1]
        # If we are the last process, set the right ghost to match the last cell
        if iend == N:
            u_ghost[-1] = u_ghost[-2]

    unext = g_unext.getArray()
    
    # Conservative Update
    for i in range(istart, iend):
        local_i = i - istart + 1  # shift for ghost
        F_plus  = numerical_flux(u_ghost[local_i], u_ghost[local_i+1], dx, dt)
        F_minus = numerical_flux(u_ghost[local_i-1], u_ghost[local_i], dx, dt)
        unext[i - istart] = u_ghost[local_i] - lam * (F_plus - F_minus)

    g_unext.assemblyBegin()
    g_unext.assemblyEnd()
    g_u, g_unext = g_unext, g_u
    t += dt

end_time = time.perf_counter()
elapsed_time = end_time - start_time

scatter, seq_u = PETSc.Scatter.toZero(g_u)
scatter.scatter(g_u, seq_u, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

if rank == 0:
    u_final = seq_u.getArray()
    x_full = np.linspace(xmin + 0.5*dx, xmax - 0.5*dx, len(u_final))
    u_exact = np.zeros_like(x_full)

    
    
    # 1. INITIAL VALUES BASED ON PROBLEM
    if problem == 2:
        uL, uR = 2.0, 1.0  # Step Function
    else:
        # For Sine wave, we use the initial condition shifted by a*t
        # Note: Burgers sine wave exact solution requires a Newton solver (Method of Characteristics)
        # We will focus on the Step Function generalization here.
        uL, uR = 0, 0 

    # 2. CALCULATE EXACT SOLUTION
    if system == 1: # LINEAR TRANSPORT
        s = 1.0 # Constant wave speed a
        x_shift = (x_full - s * final_time - xmin) % (xmax - xmin) + xmin
        if problem == 2:
            u_exact = np.where(x_shift < 0.5, uL, uR)
            
    elif system == 2: # BURGERS EQUATION
        if uL > uR: # SHOCK CASE (Figure 4.8)
            s = 0.5 * (uL + uR) # Rankine-Hugoniot speed
            x_s = (0.5 + s * final_time - xmin) % (xmax - xmin) + xmin
            u_exact = np.where(x_full < x_s, uL, uR)
            
        elif uL < uR: # RAREFACTION CASE (Figure 4.9)
            # Rarefaction fan spreads between x = 0.5 + uL*t and x = 0.5 + uR*t
            xL = 0.5 + uL * final_time
            xR = 0.5 + uR * final_time
            
            for i, x in enumerate(x_full):
                if x <= xL: u_exact[i] = uL
                elif x >= xR: u_exact[i] = uR
                else: u_exact[i] = (x - 0.5) / final_time # The "Fan"
    
    # 3. RENDERING
    
    print(f"Execution time: {elapsed_time:.4f} seconds")
    plt.plot(x_full, u_exact, 'k--', label='Exact Solution')

    solver_names = {1: "Lax-Friedrichs", 2: "Rusanov", 3: "Godunov", 4: "Roe", 5: "Engquist-Osher", 6: "Upwind"}
    scheme_label = solver_names.get(solver, "Unknown")
    plt.plot(x_full, u_final, 'r-', label=f'{scheme_label} (N={N})')
    
    pde_name = "Transport" if system == 1 else "Burgers"
    plt.title(f"{pde_name} Equation at T={final_time}")
    plt.legend()
    plt.grid(True)
    plt.show()