import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameter definitions
a_no_hopf = 0.5   # Monotone/stable regime (no Hopf)
a_hopf = 0.1      # Low 'a' regime where Hopf bifurcation occurs
b = 1.0
c = 1.0

# ---------------------------------------------------------
# 1. System Dynamics & Equilibria
# ---------------------------------------------------------
def system_ode(t, state, a, b, c, G):
    u, v = state
    f = -v - u*(v + 1) + G
    du = -a*u + b*v + c*f
    dv = f
    return [du, dv]

def get_equilibria(a, b, G):
    # Quadratic equation for v*: v^2 + (1 + a/b)v - (a/b)G = 0
    discriminant = (1 + a/b)**2 + 4*(a/b)*G
    if discriminant < 0:
        return []
    v_upper = -0.5*(1 + a/b) + 0.5*np.sqrt(discriminant)
    v_lower = -0.5*(1 + a/b) - 0.5*np.sqrt(discriminant)
    u_upper = (b/a)*v_upper
    u_lower = (b/a)*v_lower
    return [(u_upper, v_upper), (u_lower, v_lower)]

# ---------------------------------------------------------
# 2. One-Parameter Bifurcation Diagrams (v vs G)
# ---------------------------------------------------------
G_vals = np.linspace(0.01, 5.0, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for ax, a_val, title in zip([ax1, ax2], [a_no_hopf, a_hopf], [f"a = {a_no_hopf} (No Hopf)", f"a = {a_hopf} (Hopf Bifurcation)"]):
    v_upper_branch = []
    v_lower_branch = []
    
    for G in G_vals:
        eqs = get_equilibria(a_val, b, G)
        if eqs:
            v_upper_branch.append(eqs[0][1])
            v_lower_branch.append(eqs[1][1])
        else:
            v_upper_branch.append(np.nan)
            v_lower_branch.append(np.nan)
            
    ax.plot(G_vals, v_upper_branch, 'b-', label='Upper Equilibrium ($eq_0$)')
    ax.plot(G_vals, v_lower_branch, 'r--', label='Lower Equilibrium / Saddle ($eq_1$)')
    ax.set_title(f"Bifurcation Diagram: {title}")
    ax.set_xlabel("Glucose Load $G$")
    ax.set_ylabel("Glycemia State $v^*$")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("bifurcation_diagrams.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Phase Space & Controllability Safe Zone Analysis
# ---------------------------------------------------------
G_test = 2.0
fig, ax = plt.subplots(figsize=(8, 6))

# Grid for vector field
u_range = np.linspace(-15, 10, 200)
v_range = np.linspace(-10, 15, 200)
U, V = np.meshgrid(u_range, v_range)

f_grid = -V - U*(V + 1) + G_test
dU = -a_no_hopf*U + b*V + c*f_grid
dV = f_grid

# Vector field streamlines
ax.streamplot(U, V, dU, dV, color='lightgrey', density=1.2, linewidth=0.7)

# Nullclines
v_null = np.linspace(-9.9, 14.9, 400)
v_null = v_null[np.abs(v_null + 1) > 0.05] # avoid division by zero
u_null_dv = (G_test - v_null)/(v_null + 1)
ax.plot(u_null_dv, v_null, 'orange', label=r'$\dot{v} = 0$ Nullcline')

u_null_du = (b*v_null - c*v_null + c*G_test) / (a_no_hopf + c*(v_null + 1))
ax.plot(u_null_du, v_null, 'blue', label=r'$\dot{u} = 0$ Nullcline')

# Equilibria plot
eqs = get_equilibria(a_no_hopf, b, G_test)
if eqs:
    ax.plot(eqs[0][0], eqs[0][1], 'go', markersize=8, label='Stable Node ($eq_0$)')
    ax.plot(eqs[1][0], eqs[1][1], 'ro', markersize=8, label='Saddle Point ($eq_1$)')

# Sample Trajectories (Controlled vs Divergent)
t_span = (0, 15)
t_eval = np.linspace(0, 15, 500)

sol_safe = solve_ivp(system_ode, t_span, [-5.0, 5.0], args=(a_no_hopf, b, c, G_test), t_eval=t_eval)
ax.plot(sol_safe.y[0], sol_safe.y[1], 'g-', linewidth=2, label='Safe Trajectory')

sol_diverge = solve_ivp(system_ode, t_span, [-15.0, -5.0], args=(a_no_hopf, b, c, G_test), t_eval=t_eval)
ax.plot(sol_diverge.y[0], sol_diverge.y[1], 'r-', linewidth=2, label='Divergent Trajectory')

ax.set_xlim([-15, 10])
ax.set_ylim([-10, 15])
ax.set_xlabel("Insulin State $u$")
ax.set_ylabel("Glycemia State $v$")
ax.set_title(f"Phase Space Dynamics & Controllability Region (G = {G_test}, a = {a_no_hopf})")
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("phase_portrait_controllability.png", dpi=300)
plt.close()

print("Execution complete. Figures 'bifurcation_diagrams.png' and 'phase_portrait_controllability.png' generated successfully.")