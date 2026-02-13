import matplotlib.tri as mtri
import matplotlib.pyplot as plt
def my_plot(solution,Vh):
    fig = plt.figure(figsize=(6, 6))
    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triangulation = mtri.Triangulation(Vh.space_mesh.node[:, 0], Vh.space_mesh.node[:, 1])

    # Plot the surface.
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = plt.axes(projection='3d')
    # ax.plot_trisurf(triangulation, solution.T[0], cmap=plt.cm.CMRmap)
    trisurf = ax.plot_trisurf(triangulation, solution.T[0],     # type: ignore
                            cmap=plt.cm.coolwarm,  # coolwarm, CMRmap, cmap=plt.cm.YlGnBu_r, gist_earth  # type: ignore
                            linewidth=0.2,
                            antialiased=True,
                            edgecolor='grey')
    # fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')      # type: ignore
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # ax.view_init(30, 37.5)
    ax.view_init(30, 37.5)  # type: ignore

    plt.show()

    # plt.savefig('The numerical solution.png', dpi=300)
    # plt.savefig('The numerical solution.png', bbox_inches='tight')
    # plt.savefig('foo.png')
    plt.savefig('foo.pdf')