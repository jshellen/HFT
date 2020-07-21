import numpy as np
import matplotlib.pyplot as pp

from src.hjb_solvers import (
    MM_Model_Parameters,
    AS3P_Finite_Difference_Solver,
)


def create_as3p_model_solutions():
    
    lambda_m = 50
    lambda_p = 50
    kappa_m = 100
    kappa_p = 100
    delta = 0
    phi = 0.000001
    alpha = 0.0001
    q_min = -25
    q_max = 25
    cost = 0.005
    rebate = 0.0025
    T = 5 # minutes
    n = 5*60 # one step per second
    
    
    d_grid = np.linspace(0, 0.1, 100)
    fig, ax = pp.subplots(figsize=[3.5, 3]);
    pp.plot(d_grid, lambda_p*np.exp(-kappa_p*d_grid),
            color='blue', lw=3)
    ax.set_xlabel('distance from mid')
    ax.set_ylabel('fill rate (fills/minute)')
    
    parameters = MM_Model_Parameters(lambda_m, lambda_p, kappa_m, kappa_p, delta,
                                   phi, alpha, q_min, q_max, T, cost, rebate)
    
    impulses, model = AS3P_Finite_Difference_Solver.solve(parameters, N_steps=n)
    
    # Plot the value function
    Y = model.q_grid
    X = model.t_grid
    X, Y = np.meshgrid(X,Y)
    f = pp.figure(figsize=[5, 4]);
    pp3d = pp.axes(projection="3d", elev=20, azim=50);
    pp3d.set_title("Value function");
    pp3d.set_xlabel("Minute");
    pp3d.set_ylabel("Inventory");
    pp3d.set_zlabel("Value");
    pp3d.plot_surface(X, Y, model.h, cmap='magma');
    f.savefig("./graphs/value_function.pdf", bbox_inches='tight')
    
    # Plot the impulse regions
    from matplotlib import colors
    import matplotlib.patches as mpatches
    mycolors = ['white', 'blue', 'red']
    cmap = colors.ListedColormap(mycolors)
    
    f, ax = pp.subplots(figsize=[4, 4])
    ax.imshow(impulses, cmap=cmap, aspect='auto')
    ax.set_xticks([0, 0.5*n, n])
    ax.set_xticklabels([0, int(0.5*n), n],fontsize=8);
    ax.set_yticks(np.arange(0, len(model.q_grid), 2));
    ax.set_yticklabels(model.q_grid[::2],fontsize=8);
    ax.set_ylabel('Inventory')
    ax.set_xlabel('Second')
    f.savefig("./graphs/impulse_regions.pdf", bbox_inches='tight')
    
    # Plot the ask spread for continuation region
    f, ax = pp.subplots(figsize=[5, 4])
    for q in range(3, -4, -1):
        ax.plot(model.l_p[q], label=f'q={q}')
    ax.set_title("Ask to mid spread")
    pp.legend()
    f.savefig("./graphs/ask_to_mid_spread.pdf", bbox_inches='tight')

    # Plot the bid spread for continuation region
    fig, ax = pp.subplots(figsize=[5, 4])
    for q in range(3, -4, -1):
        ax.plot(model.l_m[q], label=f'q={q}')
    ax.set_title("Bid to mid spread")
    pp.legend()
    f.savefig("./graphs/bid_to_mid_spread.pdf", bbox_inches='tight')
    
    pp.show()
    
    
if __name__ == '__main__':
    create_as3p_model_solutions()










