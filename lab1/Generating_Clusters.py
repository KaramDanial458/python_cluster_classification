import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

if __name__ == '__main__':
    #-- Example usage -----------------------
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
            mean=(1,1), cov=[[0.4, 9],[9, 10]], size=1000
            )
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro')

    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.5, color='green')

    plt.show()

def boundary(x, means):
    if len(means) == 2:
        z_k = means[0]
        z_l = means[1]
        assert isinstance(x, tuple)
        """See equation 3.6 for MED classifier."""
        return np.matmul(np.transpose(np.subtract(z_k, z_l)), x) \
               + 0.5 * (np.matmul(np.transpose(z_l), z_l) - np.matmul(np.transpose(z_k), z_k))
    elif len(means) == 3:
        z = [0, 0, 0]
        for i in np.arange(3):
            z[i] = 0.5 * (np.matmul(np.transpose(means[i]), means[i])) - np.matmul(np.transpose(means[i]), x)
        z.sort()
        return z[1] - z[0]


mean_class_a = (5, 10)
cov_class_a = [[8, 0], [0, 4]]
mean_class_b = (10, 15)
cov_class_b = [[8, 0], [0, 4]]
mean_class_c = (5, 10)
cov_class_c = [[8, 4], [4, 40]]
mean_class_d = (15, 10)
cov_class_d = [[8, 0], [0, 8]]
mean_class_e = (10, 5)
cov_class_e = [[10, -5], [-5, 20]]

x_class_a, y_class_a = np.random.multivariate_normal(mean_class_a, cov_class_a, 200).T
x_class_b, y_class_b = np.random.multivariate_normal(mean_class_b, cov_class_b, 200).T
x_class_c, y_class_c = np.random.multivariate_normal(mean_class_c, cov_class_c, 100).T
x_class_d, y_class_d = np.random.multivariate_normal(mean_class_d, cov_class_d, 200).T
x_class_e, y_class_e = np.random.multivariate_normal(mean_class_e, cov_class_e, 150).T

fig, ax = plt.subplots()
ax.plot(x_class_a, y_class_a, "b.")
ax.plot(x_class_b, y_class_b, 'r.')
x = np.linspace(-5, 20, 25)
y = np.linspace(0, 25, 25)
X, Y = np.meshgrid(x, y)
Z = np.zeros((25, 25))
for i in np.arange(len(x)):
    for j in np.arange(len(y)):
        Z[i][j] = boundary((x[i], y[j]), [mean_class_a, mean_class_b])
cs = ax.contour(X, Y, Z, levels=0)
plt.axis('equal')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_class_c, y_class_c, "b.")
ax.plot(x_class_d, y_class_d, 'r.')
ax.plot(x_class_e, y_class_e, 'g.')
x = np.linspace(-10, 30, 50)
y = np.linspace(-10, 25, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros((50, 50))
for i in np.arange(len(x)):
    for j in np.arange(len(y)):
        Z[i][j] = boundary((x[i], y[j]), [mean_class_c, mean_class_d, mean_class_e])
cs = ax.contour(X, Y, Z, levels=0)
plt.axis('equal')
plt.show()