import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polytope as pc
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_ellipses(ellipses_list, interactive=False):
    """
    Plots one or more ellipses, automatically handling degenerate ellipses 
    and perfectly adapting the space for the external legend.
    """
    fig, ax = plt.subplots(figsize=(9, 6)) # Slightly widened the default figure
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_title("Ellipsoidal Uncertainty Evolution")
    
    colors = plt.colormaps['tab10']
    
    if interactive:
        plt.ion()
        plt.show()

    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf

    for i, tup in enumerate(ellipses_list):
        if len(tup) == 3:
            P, theta_hat, label = tup
        elif len(tup) == 2:
            P, theta_hat = tup
            label = f'{i}'
        
        color = colors(i % 10)
        theta_hat = np.array(theta_hat).flatten()
        
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        
        degeneration_threshold = 1e-8
        is_degenerate = False
        
        eigenvalues_plot = np.copy(eigenvalues)
        for j in range(len(eigenvalues_plot)):
            if eigenvalues_plot[j] < degeneration_threshold:
                eigenvalues_plot[j] = 1e8  
                is_degenerate = True
                
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues_plot)
        
        ell = mpatches.Ellipse(
            xy=theta_hat, width=width, height=height, angle=angle,
            edgecolor=color, facecolor='none', linewidth=2, label=label
        )
        
        ax.add_patch(ell)
        ax.plot(theta_hat[0], theta_hat[1], marker='x', color=color)

        if not is_degenerate:
            dx = np.sqrt(P[0, 0])
            dy = np.sqrt(P[1, 1])
            
            min_x = min(min_x, theta_hat[0] - dx)
            max_x = max(max_x, theta_hat[0] + dx)
            min_y = min(min_y, theta_hat[1] - dy)
            max_y = max(max_y, theta_hat[1] + dy)

        if min_x != np.inf:
            margin_x = (max_x - min_x) * 0.15 if max_x > min_x else 1.0
            margin_y = (max_y - min_y) * 0.15 if max_y > min_y else 1.0
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)

        if interactive:
            # Position the legend center right
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # Tell tight_layout to use only the left 80% of the figure for the plot
            plt.tight_layout(rect=[0, 0, 0.99, 1])
            fig.canvas.draw()
            print(f"[{i+1}/{len(ellipses_list)}] {label} plotted. Press any key on the figure...")
            plt.waitforbuttonpress()
            
    if not interactive:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, .5))
        # Leave 20% space on the right (from 0.8 to 1.0) for the legend
        # plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.show()
    else:
        plt.ioff()
        print("Ellipses plotting completed.\n")
        plt.close()


def plot_ellipsoids(ellipsoids_list, interactive=False):
    """
    Plots one or more 3D ellipsoids automatically handling degenerate ellipsoids 
    and adapting the space for the external legend.
    """
    fig = plt.figure(figsize=(10, 8))
    # Set 3D projection
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    ax.set_title("Ellipsoidal Uncertainty Evolution (3D)")
    
    colors = plt.colormaps['tab10']
    
    if interactive:
        plt.ion()
        plt.show()

    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    min_z, max_z = np.inf, -np.inf

    # In 3D plot_surface doesn't support direct labels well for the legend, 
    # so we create proxy "patches" for the legend.
    legend_patches = []

    for i, tup in enumerate(ellipsoids_list):
        if len(tup) == 3:
            P, theta_hat, label = tup
        elif len(tup) == 2:
            P, theta_hat = tup
            label = f'{i}'
        
        color = colors(i % 10)
        theta_hat = np.array(theta_hat).flatten()
        
        # 1. Generate points of a unit sphere
        u = np.linspace(0.0, 2.0 * np.pi, 40)
        v = np.linspace(0.0, np.pi, 40)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones_like(u), np.cos(v))
        
        # Flatten for matrix transformation (shape: 3 x N)
        sphere_points = np.vstack((x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()))
        
        # 2. Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        
        degeneration_threshold = 1e-8
        is_degenerate = False
        eigenvalues_plot = np.copy(eigenvalues)
        
        for j in range(len(eigenvalues_plot)):
            if eigenvalues_plot[j] < degeneration_threshold:
                eigenvalues_plot[j] = 1e8  
                is_degenerate = True
                
        # 3. Transformation: calculate the deformation matrix A = V * sqrt(Lambda)
        radii = np.sqrt(eigenvalues_plot)
        A = eigenvectors @ np.diag(radii)
        
        # Deform the sphere into an ellipsoid and translate the center to theta_hat
        ellipsoid_points = A @ sphere_points
        X = ellipsoid_points[0, :] + theta_hat[0]
        Y = ellipsoid_points[1, :] + theta_hat[1]
        Z = ellipsoid_points[2, :] + theta_hat[2]
        
        # Reshape to the original grid shape for surface plotting
        X = X.reshape(x_sphere.shape)
        Y = Y.reshape(y_sphere.shape)
        Z = Z.reshape(z_sphere.shape)
        
        # Plot the surface (with alpha transparency to see through) and the center
        ax.plot_surface(X, Y, Z, color=color, alpha=0.3, edgecolor=color, linewidth=0.2)
        ax.scatter(*theta_hat, color=color, marker='x', s=50)
        
        legend_patches.append(mpatches.Patch(color=color, label=label, alpha=0.5))

        # 4. Update dynamic limits
        if not is_degenerate:
            dx = np.sqrt(P[0, 0])
            dy = np.sqrt(P[1, 1])
            dz = np.sqrt(P[2, 2])
            
            min_x = min(min_x, theta_hat[0] - dx)
            max_x = max(max_x, theta_hat[0] + dx)
            min_y = min(min_y, theta_hat[1] - dy)
            max_y = max(max_y, theta_hat[1] + dy)
            min_z = min(min_z, theta_hat[2] - dz)
            max_z = max(max_z, theta_hat[2] + dz)

        if min_x != np.inf:
            margin_x = (max_x - min_x) * 0.15 if max_x > min_x else 1.0
            margin_y = (max_y - min_y) * 0.15 if max_y > min_y else 1.0
            margin_z = (max_z - min_z) * 0.15 if max_z > min_z else 1.0
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.set_zlim(min_z - margin_z, max_z + margin_z)
            
            # Force cubic proportions to avoid visual distortion of ellipsoids
            ax.set_box_aspect((1, 1, 1))

        if interactive:
            ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.1, 0.5))
            plt.tight_layout(rect=[0, 0, 0.80, 1])
            fig.canvas.draw()
            print(f"[{i+1}/{len(ellipsoids_list)}] {label} plotted. Press any key on the figure...")
            plt.waitforbuttonpress()
            
    if not interactive:
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    else:
        plt.ioff()
        print("3D ellipsoids plotting completed.\n")
        plt.show()

def plot_polytopes(polytopes_list, interactive=False):
    """
    Plots one or more polytopes extracting vertices with polytope 
    and plotting them directly with matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title("Evolution of Polytopic Sets (Ax <= b)")
    
    colors = plt.colormaps['tab10']
    legend_patches = []
    
    if interactive:
        plt.ion()
        plt.show()

    for i, (A, b, label) in enumerate(polytopes_list):
        color = colors(i % 10)
        
        # 1. Use polytope only to find vertices
        p = pc.Polytope(A, b)
        vertices = pc.extreme(p)
        
        if vertices is not None and len(vertices) > 0:
            # 2. Order vertices counterclockwise using ConvexHull
            # (crucial so Matplotlib doesn't draw intersecting polygons)
            hull = ConvexHull(vertices)
            ordered_vertices = vertices[hull.vertices]
            
            # 3. Create native Matplotlib Polygon patch (note closed=True keyword)
            poly = mpatches.Polygon(
                ordered_vertices, closed=True, 
                facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5
            )
            ax.add_patch(poly)
            
            # Update axis limits based on new vertices
            ax.update_datalim(ordered_vertices)
            
        # Create a proxy artist for the legend
        patch = mpatches.Patch(color=color, alpha=0.4, label=label)
        legend_patches.append(patch)

        if interactive:
            ax.autoscale_view()
            ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1))
            fig.canvas.draw()
            print(f"[{i+1}/{len(polytopes_list)}] {label} plotted. Press any key on the figure...")
            plt.waitforbuttonpress()
            
    if not interactive:
        ax.autoscale_view()
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
    else:
        plt.ioff()
        print("Polytopes plotting completed.\n")
        plt.close()

def plot_trajectory(*trajectories):
    """
    Takes a variable number of ndarrays of shape (N, 2) as input 
    and plots the trajectories on the x1, x2 plane.
    """
    if len(trajectories) == 0:
        print("No trajectory passed as input.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Iterate over all matrices passed to the function
    for i, X in enumerate(trajectories):
        # Check that each matrix has the correct shape
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"Trajectory at position {i+1} is not an ndarray of shape (N, 2)")

        # Extract coordinates
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        # Plot the current trajectory
        ax.plot(x1, x2, marker='.', linestyle='-', alpha=0.7, label=f'Trajectory {i+1}')
    
    # Plot details
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title("Trajectories Comparison")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Maintain geometric proportions
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.show()

def plot_parameter_polytopes_3d(theta_list, M_theta, interactive=False, true_theta=None):
    """
    Plots 3D parametric uncertainty polytopes and their nominal centers.
    
    :param theta_list: List of tuples (mu_t, theta_bar) from the simulation.
    :param M_theta: Fixed matrix of the directions of the polytope faces.
    :param interactive: If True, shows a frame-by-frame animation.
    :param true_theta: Optional array with true parameters (red star).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    ax.set_title("Evolution of parameter set")
    
    colors = plt.colormaps['tab10']
    legend_patches = []
    
    if interactive:
        plt.ion()
        plt.show()

    min_limits = np.array([np.inf, np.inf, np.inf])
    max_limits = np.array([-np.inf, -np.inf, -np.inf])

    for i, (mu_t, theta_bar) in enumerate(theta_list):
        color = colors(i % 10)
        label = f'Iteration {i}'
        
        # 1. Use polytope to robustly extract vertices
        p = pc.Polytope(M_theta, mu_t)
        vertices = pc.extreme(p)
        
        if vertices is not None and len(vertices) >= 4:
            try:
                # 2. Convex hull to create 3D faces (triangulation)
                hull = ConvexHull(vertices)
                faces = [vertices[simplex] for simplex in hull.simplices]
                
                # 3. Create Poly3DCollection to draw the 3D volume
                poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.2)
                ax.add_collection3d(poly3d)
                
                # Update scene limits
                min_limits = np.minimum(min_limits, np.min(vertices, axis=0))
                max_limits = np.maximum(max_limits, np.max(vertices, axis=0))
            except Exception as e:
                print(f"Warning: Impossible to create ConvexHull at step {i}: {e}")
        
        # Always draw the nominal point (projection)
        ax.scatter(*theta_bar, color=color, marker='x', s=50)
        legend_patches.append(mpatches.Patch(color=color, label=label, alpha=0.5))

        if interactive:
            # Scale axes to the current frame
            for j, set_lim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                margin = (max_limits[j] - min_limits[j]) * 0.15 if max_limits[j] > min_limits[j] else 1.0
                if not np.isinf(margin):
                    set_lim(min_limits[j] - margin, max_limits[j] + margin)
                    
            ax.set_box_aspect((1, 1, 1))
            ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.1, 0.5))
            plt.tight_layout(rect=[0, 0, 0.80, 1])
            fig.canvas.draw()
            print(f"[{i+1}/{len(theta_list)}] {label} plotted. Press any key on the figure...")
            plt.waitforbuttonpress()
            
    if true_theta is not None:
        ax.scatter(*true_theta, color='red', marker='*', s=200, label='true value')
        legend_patches.append(mpatches.Patch(color='red', label='true value'))

    # Finalize limits for static drawing
    if not np.any(np.isinf(min_limits)):
        margins = (max_limits - min_limits) * 0.15
        margins[margins == 0] = 1.0
        ax.set_xlim(min_limits[0] - margins[0], max_limits[0] + margins[0])
        ax.set_ylim(min_limits[1] - margins[1], max_limits[1] + margins[1])
        ax.set_zlim(min_limits[2] - margins[2], max_limits[2] + margins[2])
        ax.set_box_aspect((1, 1, 1))

    if not interactive:
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    else:
        plt.ioff()
        print("All polytopes have been plotted.\n")
        plt.close()


if __name__ == "__main__":
    
    print("--- Test 1: Plotting Ellipses ---")
    # Data taken from Example 1 of the paper for the initial ellipsoid
    P_1 = np.array([[4.0, 1.0], 
                    [1.0, 2.0]])
    theta_1 = np.array([0.8, 0.7])
    
    # Artificially reduce uncertainty to simulate subsequent steps
    P_2 = P_1 * 0.5
    theta_2 = np.array([0.5, 0.4])
    
    P_3 = P_2 * 0.3
    theta_3 = np.array([0.2, 0.1]) # Convergence towards the true value [0.2, 0.1]
    
    ellipses_data = [
        (P_1, theta_1, "Iteration 1 (Prior)"),
        (P_2, theta_2, "Iteration 2"),
        (P_3, theta_3, "Iteration 5 (Convergence)")
    ]
    
    # Change to interactive=True if you want to see the step-by-step animation!
    plot_ellipses(ellipses_data, interactive=True)
    
    
    print("--- Test 2: Plotting Polytopes ---")
    # Polytope 1: A large square (e.g., physical limits of the system)
    A_1 = np.array([[ 1,  0], 
                    [-1,  0], 
                    [ 0,  1], 
                    [ 0, -1]])
    b_1 = np.array([5, 5, 5, 5])
    
    # Polytope 2: A narrower rectangle (e.g., first invariant set)
    A_2 = A_1
    b_2 = np.array([2, 1, 3, 2])
    
    # Polytope 3: A triangle (e.g., intersection with an oblique constraint)
    A_3 = np.array([[ 1,  1], 
                    [-1,  0], 
                    [ 0, -1]])
    b_3 = np.array([2, 0, 0])
    
    polytopes_data = [
        (A_1, b_1, "System constraints"),
        (A_2, b_2, "Invariant Set t=1"),
        (A_3, b_3, "Invariant Set t=5")
    ]
    
    plot_polytopes(polytopes_data, interactive=True)