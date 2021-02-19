

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.spatial import distance
from random import randint 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels


def GED_classifier(mean_a, mean_b, cov_a, cov_b, X, Y):
    dist_matrix = list()
    row = X.shape[0]
    col = X.shape[1]
    mean_a = np.array(mean_a)
    mean_b = np.array(mean_b)
    inv_cov_a = np.linalg.inv(np.array(cov_a))
    inv_cov_b = np.linalg.inv(np.array(cov_b))
    points = np.concatenate((X.reshape(row*col,1), Y.reshape(row*col,1)), axis = 1)
    
    x_mean_a = np.subtract(points,mean_a)
    x_mean_b = np.subtract(points,mean_b)
    for p in range(x_mean_a.shape[0]):
        dist_a = np.sqrt(np.dot(np.dot(x_mean_a[p].T, np.linalg.inv(inv_cov_a)),x_mean_a[p]))
        dist_b = np.sqrt(np.dot(np.dot(x_mean_b[p].T, np.linalg.inv(inv_cov_b)),x_mean_b[p]))
        dist_matrix.append(dist_a-dist_b)
    return np.array(dist_matrix).reshape(X.shape)
    # return dist_matrix
    """return dist matrix"""

def KNN_classifier(x_class_a, x_class_b, X_mesh, Y_mesh, k = 1):
    '''
    binary classifier
    x_class_a: the feature vector for class A, shape = (2,N)
    x_class_b: the feature vector for class B, shape = (2,N)
    k: k nearest neighbor
    '''
    # the list that holds the top k nearest points in x_class_a or x_class_b
    # generate a 2D array to hold the distance
    row = X_mesh.shape[0]
    col = X_mesh.shape[1]
    dist_matrix = list()

    nn_list = []
    mesh_points = np.concatenate((X_mesh.reshape(row*col,1), Y_mesh.reshape(row*col,1)), axis = 1)
    x_class_a = x_class_a.T 
    x_class_b = x_class_b.T

    for vec in mesh_points:
        nn_list_a = list()
        for vec_a in x_class_a:
            #get distance from the point in the mesh grid to the feature vector in a
            dist = distance.euclidean(vec, vec_a)
            nn_list_a.append(dist)
        #compute the prototype by taking the mean of the nn list
        nn_list_a.sort()
        dist_a = sum(nn_list_a[0:k])/k

        nn_list_b = list()
        for vec_b in x_class_b:
            #get distance from the point in the mesh grid to the feature vector in a
            dist = distance.euclidean(vec, vec_b)
            nn_list_b.append(dist)
        #compute the prototype by taking the mean of the nn list
        nn_list_b.sort()
        dist_b = sum(nn_list_b[0:k])/k
        dist_matrix.append(dist_a-dist_b)

    return np.array(dist_matrix).reshape(X_mesh.shape)
    """return dist matrix"""

def MAP_classifier(mean_a, mean_b, cov_a, cov_b, X, Y, n_a, n_b):
    dist_matrix_a = list()
    row = X.shape[0]
    col = X.shape[1]
    mean_a = np.array(mean_a)
    mean_b = np.array(mean_b)
    det_cov_a = np.linalg.det(np.array(cov_a))
    det_cov_b = np.linalg.det(np.array(cov_b))
    inv_cov_a = np.linalg.inv(np.array(cov_a))
    inv_cov_b = np.linalg.inv(np.array(cov_b))

    Q0 = np.subtract(inv_cov_a, inv_cov_b)
    Q1 = 2 * (np.dot(mean_b,inv_cov_b)- np.dot(mean_a, inv_cov_a))
    Q2 = np.dot(np.dot(mean_a, inv_cov_a), mean_a.T) - np.dot(np.dot(mean_b, inv_cov_b), mean_b.T)
    Q3 = np.log((n_b/n_a))
    Q4 = np.log(det_cov_a/det_cov_b)
    
    points = np.concatenate((X.reshape(row*col,1), Y.reshape(row*col,1)), axis = 1)
    
    # x_mean_a = np.subtract(points,mean_a)
    # x_mean_b = np.subtract(points,mean_b)
    for p in range(points.shape[0]):
        dist = np.dot(np.dot(points[p],Q0), points[p].T) + np.dot(Q1,points[p]) + Q2 + 2 * Q3 + Q4
        dist_matrix_a.append(dist)
    return np.array(dist_matrix_a).reshape(X.shape)
    """return dist matrix"""

def MED_classifier(mean_a, mean_b, X, Y):
    dist_matrix = list()
    row = X.shape[0]
    col = X.shape[1]
    mean_a = np.array(mean_a)
    mean_b = np.array(mean_b)
    
    points = np.concatenate((X.reshape(row*col,1), Y.reshape(row*col,1)), axis = 1)
    x_mean_a = np.subtract(points,mean_a)
    x_mean_b = np.subtract(points,mean_b)
    for p in range(points.shape[0]):
        dist_a = np.sqrt(np.dot(x_mean_a[p].T, x_mean_a[p]))
        dist_b = np.sqrt(np.dot(x_mean_b[p].T, x_mean_b[p]))
        dist_matrix.append(dist_a - dist_b)
    return np.array(dist_matrix).reshape(X.shape)
    """return dist matrix"""

def plot_confusion_matrix(y_true, y_pred, cm, classes, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap)
    ax_cm.figure.colorbar(im, ax=ax_cm)
    # We want to show all ticks...
    ax_cm.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig_cm.tight_layout()
    return ax_cm

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    :param n_std:
    :param facecolor:
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    plt.plot(mean_x, mean_y, 'go')
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def binary_class_matrix(dist_matrix):
    """
    return a binary class matrix on the mesh grid
    e.g. 
    [[1,1,1],
     [1,1,2],
     [2,2,2]]
    """
    class_matrix = np.empty(dist_matrix.shape, dtype=int)
    for i in np.arange(class_matrix.shape[0]):
        for j in np.arange(class_matrix.shape[1]):
            if dist_matrix[i][j] < 0:
                class_matrix[i][j] = int(1)
            else:
                class_matrix[i][j] = int(2)
    return class_matrix

def three_class_matrix(dist_ab, dist_bc, dist_ac):
    """
    return a multi class matrix on the mesh grid
    e.g. 
    [[1,1,1,1,1],
     [1,1,2,2,3],
     [2,2,2,3,3]]
    """
    class_matrix = np.zeros(dist_ab.shape)
    for i in np.arange(class_matrix.shape[0]):
        for j in np.arange(class_matrix.shape[1]):
            if   dist_ab[i][j] <= 0 and dist_ac[i][j] <= 0 :
                class_matrix[i][j] = 1
            elif dist_ab[i][j] >= 0 and dist_bc[i][j] <= 0 :
                class_matrix[i][j] = 2
            elif dist_ac[i][j] >= 0 and dist_bc[i][j] >= 0 :
                class_matrix[i][j] = 3
            else:
                class_matrix[i][j] = randint(1, 3)  
    return class_matrix

def error_analysis(true_class, prediction_class):
    """
    use ski-kit learn to find error (1 - accuracy)
    generate confusion matrix
    """
    con_matrix = confusion_matrix(true_class[0], prediction_class[0])
    accuracy = 1- accuracy_score(true_class[0], prediction_class[0])
    return accuracy, con_matrix

def main():
    #generate predictable random numbers
    np.random.seed(0)

    # random dataset generation for class a,b,c,d,e
    mean_class_a = np.array([5, 10])
    cov_class_a = np.array([[8, 0], [0, 4]])
    n_a = 200

    mean_class_b = np.array([10, 15])
    cov_class_b = np.array([[8, 0], [0, 4]])
    n_b = 200

    mean_class_c = np.array([5, 10])
    cov_class_c = np.array([[8, 4], [4, 40]])
    n_c = 100

    mean_class_d = np.array([15, 10])
    cov_class_d = np.array([[8, 0], [0, 8]])
    n_d = 200

    mean_class_e = np.array([10, 5])
    cov_class_e = np.array([[10, -5], [-5, 20]])
    n_e = 150

    x_class_a = np.random.multivariate_normal(mean_class_a, cov_class_a, n_a).T
    x_class_b = np.random.multivariate_normal(mean_class_b, cov_class_b, n_b).T
    x_class_c = np.random.multivariate_normal(mean_class_c, cov_class_c, n_c).T
    x_class_d = np.random.multivariate_normal(mean_class_d, cov_class_d, n_d).T
    x_class_e = np.random.multivariate_normal(mean_class_e, cov_class_e, n_e).T
    
    # Mesh grid generation 
    delta = 0.2
    X_1 = np.concatenate((x_class_a,x_class_b), axis = 1)
    X_2 = np.concatenate((x_class_c,x_class_d,x_class_e), axis = 1)

    x_min_1, x_max_1 = X_1[0].min() - 2, X_1[0].max() + 2
    y_min_1, y_max_1 = X_1[1].min() - 2, X_1[1].max() + 2
    x_min_2, x_max_2 = X_2[0].min() - 2, X_2[0].max() + 2
    y_min_2, y_max_2 = X_2[1].min() - 2, X_2[1].max() + 2

    steps = 100

    x_range_1 = np.linspace(x_min_1, x_max_1, steps)
    y_range_1 = np.linspace(y_min_1, y_max_1, steps)
    x_range_2 = np.linspace(x_min_2, x_max_2, steps)
    y_range_2 = np.linspace(y_min_2, y_max_2, steps)

    X_1_mesh, Y_1_mesh = np.meshgrid(x_range_1, y_range_1)
    X_2_mesh, Y_2_mesh = np.meshgrid(x_range_2, y_range_2)

    #Unit contour plot for class a and b
    fig1, ax1 = plt.subplots()
    a, = ax1.plot(x_class_a[0], x_class_a[1], "b.")
    confidence_ellipse(x_class_a[0], x_class_a[1], ax1, n_std=1, edgecolor='black')
    b, = ax1.plot(x_class_b[0], x_class_b[1], 'r.')
    confidence_ellipse(x_class_b[0], x_class_b[1], ax1, n_std=1, edgecolor='black')
    ax1.legend((a, b), ('Class A', 'Class B'), loc='upper left', fontsize='x-large')
    plt.axis('equal')
    plt.show()

    #Unit contour plot for class c, d and e
    fig2, ax2 = plt.subplots()
    c, = ax2.plot(x_class_c[0], x_class_c[1], "b.")
    confidence_ellipse(x_class_c[0], x_class_c[1], ax2, n_std=1, edgecolor='black')
    d, = ax2.plot(x_class_d[0], x_class_d[1], 'r.')
    confidence_ellipse(x_class_d[0], x_class_d[1], ax2, n_std=1, edgecolor='black')
    e, = ax2.plot(x_class_e[0], x_class_e[1], 'y.')
    confidence_ellipse(x_class_e[0], x_class_e[1], ax2, n_std=1, edgecolor='black')
    ax2.legend((c, d, e), ('Class C', 'Class D', 'Class E'), loc='best', fontsize='x-large')
    plt.axis('equal')
    plt.show()

    
    #distance & class matrices generation for group 1 (A vs. B)
    distance_matrix_nn_1 = KNN_classifier(x_class_a, x_class_b, X_1_mesh, Y_1_mesh)
    class_matrix_nn_1 = binary_class_matrix(distance_matrix_nn_1)

    distance_matrix_knn_5_1 = KNN_classifier(x_class_a, x_class_b, X_1_mesh, Y_1_mesh,5)
    class_matrix_knn_5_1 = binary_class_matrix(distance_matrix_knn_5_1)

    distance_matrix_map_1 = MAP_classifier(mean_class_a, mean_class_b, cov_class_a, cov_class_b, X_1_mesh, Y_1_mesh, n_a, n_b)
    class_matrix_map_1 = binary_class_matrix(distance_matrix_map_1)

    distance_matrix_ged_1 = GED_classifier(mean_class_a, mean_class_b, cov_class_a, cov_class_b, X_1_mesh, Y_1_mesh)
    class_matrix_ged_1 = binary_class_matrix(distance_matrix_ged_1)

    distance_matrix_med_1 = MED_classifier(mean_class_a, mean_class_b, X_1_mesh, Y_1_mesh)
    class_matrix_med_1 = binary_class_matrix(distance_matrix_med_1)
    

    #distance & class matrices generation for group 2 (C vs. D vs. E)
    distance_matrix_med_cd_2 = MED_classifier(mean_class_c, mean_class_d, X_2_mesh, Y_2_mesh)
    distance_matrix_med_de_2 = MED_classifier(mean_class_d, mean_class_e, X_2_mesh, Y_2_mesh)
    distance_matrix_med_ce_2 = MED_classifier(mean_class_c, mean_class_e, X_2_mesh, Y_2_mesh)
    class_matrix_med_2 = three_class_matrix(distance_matrix_med_cd_2, distance_matrix_med_de_2, distance_matrix_med_ce_2)

    distance_matrix_map_cd_2 = MAP_classifier(mean_class_c, mean_class_d, cov_class_c, cov_class_d, X_2_mesh, Y_2_mesh, n_c, n_d)
    distance_matrix_map_de_2 = MAP_classifier(mean_class_d, mean_class_e, cov_class_d, cov_class_e, X_2_mesh, Y_2_mesh, n_d, n_e)
    distance_matrix_map_ce_2 = MAP_classifier(mean_class_c, mean_class_e, cov_class_c, cov_class_e, X_2_mesh, Y_2_mesh, n_c, n_e)
    class_matrix_map_2 = three_class_matrix(distance_matrix_map_cd_2, distance_matrix_map_de_2, distance_matrix_map_ce_2)

    distance_matrix_ged_cd_2 = GED_classifier(mean_class_c, mean_class_d, cov_class_c, cov_class_d, X_2_mesh, Y_2_mesh)
    distance_matrix_ged_de_2 = GED_classifier(mean_class_d, mean_class_e, cov_class_d, cov_class_e, X_2_mesh, Y_2_mesh)
    distance_matrix_ged_ce_2 = GED_classifier(mean_class_c, mean_class_e, cov_class_c, cov_class_e, X_2_mesh, Y_2_mesh)
    class_matrix_ged_2 = three_class_matrix(distance_matrix_ged_cd_2, distance_matrix_ged_de_2, distance_matrix_ged_ce_2)

    distance_matrix_nn_cd_2 = KNN_classifier(x_class_c, x_class_d, X_2_mesh, Y_2_mesh)
    distance_matrix_nn_de_2 = KNN_classifier(x_class_d, x_class_e, X_2_mesh, Y_2_mesh)
    distance_matrix_nn_ce_2 = KNN_classifier(x_class_c, x_class_e, X_2_mesh, Y_2_mesh)
    class_matrix_nn_2 = three_class_matrix(distance_matrix_nn_cd_2, distance_matrix_nn_de_2, distance_matrix_nn_ce_2)

    distance_matrix_knn5_cd_2 = KNN_classifier(x_class_c, x_class_d, X_2_mesh, Y_2_mesh, 5)
    distance_matrix_knn5_de_2 = KNN_classifier(x_class_d, x_class_e, X_2_mesh, Y_2_mesh, 5)
    distance_matrix_knn5_ce_2 = KNN_classifier(x_class_c, x_class_e, X_2_mesh, Y_2_mesh, 5)
    class_matrix_knn5_2 = three_class_matrix(distance_matrix_knn5_cd_2, distance_matrix_knn5_de_2, distance_matrix_knn5_ce_2)
    

    #Plot Decision Boundary for 5NN (A vs. B)
    fig_5nn1, ax_5nn1 = plt.subplots()
    ax_5nn1.plot(x_class_a[0], x_class_a[1], "b.")
    confidence_ellipse(x_class_a[0], x_class_a[1], ax_5nn1, n_std=1, edgecolor='black')
    ax_5nn1.plot(x_class_b[0], x_class_b[1], 'r.')
    confidence_ellipse(x_class_b[0], x_class_b[1], ax_5nn1, n_std=1, edgecolor='black')
    plt.title('5NN Decision Boundary')
    knn1 = plt.contour(X_1_mesh, Y_1_mesh, class_matrix_knn_5_1, linewidths=0.5, colors='orange')
    ax_5nn1.legend((a, b, knn1.collections[0]), ('Class A', 'Class B', '5NN Boundary'), loc='upper right', fontsize='x-small')
    plt.show()


    #Plot Decision Boundary for NN (A vs. B)
    fig_nn1, ax_nn1 = plt.subplots()
    ax_nn1.plot(x_class_a[0], x_class_a[1], "b.")
    confidence_ellipse(x_class_a[0], x_class_a[1], ax_nn1, n_std=1, edgecolor='black')
    ax_nn1.plot(x_class_b[0], x_class_b[1], 'r.')
    confidence_ellipse(x_class_b[0], x_class_b[1], ax_nn1, n_std=1, edgecolor='black')
    plt.title('NN Decision Boundary')
    nn1 = plt.contour(X_1_mesh, Y_1_mesh, class_matrix_nn_1, linewidths=0.5, colors='orange')
    ax_nn1.legend((a, b, nn1.collections[0]), ('Class A', 'Class B', '5NN Boundary'), loc='upper right', fontsize='x-small')
    plt.show()


    #Plot Decision Boundary for MED,MAP,GED (A vs. B)
    fig_comb1, ax_comb1 = plt.subplots()
    a, = ax_comb1.plot(x_class_a[0], x_class_a[1], "b.", alpha=0.5)
    confidence_ellipse(x_class_a[0], x_class_a[1], ax_comb1, n_std=1, edgecolor='black')
    b, = ax_comb1.plot(x_class_b[0], x_class_b[1], 'r.', alpha=0.5)
    confidence_ellipse(x_class_b[0], x_class_b[1], ax_comb1, n_std=1, edgecolor='black')
    plt.title('MED,MAP,GED Decision Boundary')
    med1 = plt.contour(X_1_mesh, Y_1_mesh, class_matrix_med_1, colors='orange', linestyles='solid', linewidths=0.5)
    map1 = plt.contour(X_1_mesh, Y_1_mesh, class_matrix_map_1, colors='green', linestyles='solid', linewidths=0.5)
    ged1 = plt.contour(X_1_mesh, Y_1_mesh, class_matrix_ged_1, colors='black', linestyles='solid', linewidths=0.5)
    ax_comb1.legend((a, b, med1.collections[0], map1.collections[0], ged1.collections[0]), ('Class A', 'Class B', 'MED Boundary', 'MAP Boundary', 'GED Boundary'), loc='upper right', fontsize='x-small')
    plt.show()
    

    #Plot Decision Boundary for 5NN (C vs. D vs. E)
    fig_comb2, ax_comb2 = plt.subplots()
    c, = ax_comb2.plot(x_class_c[0], x_class_c[1], "b.", alpha=0.5)
    confidence_ellipse(x_class_c[0], x_class_c[1], ax_comb2, n_std=1, edgecolor='black')
    d, = ax_comb2.plot(x_class_d[0], x_class_d[1], 'r.', alpha=0.5)
    confidence_ellipse(x_class_d[0], x_class_d[1], ax_comb2, n_std=1, edgecolor='black')
    e, = ax_comb2.plot(x_class_e[0], x_class_e[1], 'y.', alpha=0.5)
    confidence_ellipse(x_class_e[0], x_class_e[1], ax_comb2, n_std=1, edgecolor='black')
    plt.title('MED,MAP,GED Decision Boundary')
    med2 = plt.contour(X_2_mesh, Y_2_mesh, class_matrix_med_2, colors='orange', linestyles='solid', linewidths=0.5)
    map2 = plt.contour(X_2_mesh, Y_2_mesh, class_matrix_map_2, colors='green', linestyles='solid', linewidths=0.5)
    ged2 = plt.contour(X_2_mesh, Y_2_mesh, class_matrix_ged_2, colors='black', linestyles='solid', linewidths=0.5)
    ax_comb2.legend((c, d, e, med2.collections[0], map2.collections[0], ged2.collections[0]), ('Class C', 'Class D', 'Class E', 'MED Boundary', 'MAP Boundary', 'GED Boundary'), loc='upper right', fontsize='x-small')
    plt.show()


    #Plot Decision Boundary for NN (C vs. D vs. E)
    fig_knn_5, ax_knn_5 = plt.subplots()
    ax_knn_5.plot(x_class_c[0], x_class_c[1], "b.", alpha=0.5)
    confidence_ellipse(x_class_c[0], x_class_c[1], ax_knn_5, n_std=1, edgecolor='black')
    ax_knn_5.plot(x_class_d[0], x_class_d[1], 'r.', alpha=0.5)
    confidence_ellipse(x_class_d[0], x_class_d[1], ax_knn_5, n_std=1, edgecolor='black')
    ax_knn_5.plot(x_class_e[0], x_class_e[1], 'y.', alpha=0.5)
    confidence_ellipse(x_class_e[0], x_class_e[1], ax_knn_5, n_std=1, edgecolor='black')
    plt.title('5NN Decision Boundary')
    knn2 = plt.contour(X_2_mesh, Y_2_mesh, class_matrix_knn5_2, colors='orange', linewidths=0.5)
    ax_knn_5.legend((c, d, e, knn2.collections[0]), ('Class C', 'Class D', 'Class E', '5NN Boundary'), loc='upper right', fontsize='x-small')
    plt.show()

    #Plot Decision Boundary for MED,MAP,GED (C vs. D vs. E)
    fig_nn, ax_nn = plt.subplots()
    ax_nn.plot(x_class_c[0], x_class_c[1], "b.", alpha=0.5)
    confidence_ellipse(x_class_c[0], x_class_c[1], ax_nn, n_std=1, edgecolor='black')
    ax_nn.plot(x_class_d[0], x_class_d[1], 'r.', alpha=0.5)
    confidence_ellipse(x_class_d[0], x_class_d[1], ax_nn, n_std=1, edgecolor='black')
    ax_nn.plot(x_class_e[0], x_class_e[1], 'y.', alpha=0.5)
    confidence_ellipse(x_class_e[0], x_class_e[1], ax_nn, n_std=1, edgecolor='black')
    plt.title('NN Decision Boundary')
    nn2 = plt.contour(X_2_mesh, Y_2_mesh, class_matrix_nn_2, colors='orange', linewidths=0.5)
    ax_nn.legend((c, d, e, nn2.collections[0]), ('Class C', 'Class D', 'Class E', 'NN Boundary'), loc='upper right', fontsize='x-small')
    plt.show()
    

    #testing dataset preparation
    #generate test dataset for class a - e
    classes_1 = ["A", "B"]
    classes_2 = ["C", "D", "E"]
    test_class_a = np.random.multivariate_normal(mean_class_a, cov_class_a, n_a).T
    test_class_b = np.random.multivariate_normal(mean_class_b, cov_class_b, n_b).T
    test_class_c = np.random.multivariate_normal(mean_class_c, cov_class_c, n_c).T
    test_class_d = np.random.multivariate_normal(mean_class_d, cov_class_d, n_d).T
    test_class_e = np.random.multivariate_normal(mean_class_e, cov_class_e, n_e).T

    true_label_1 = np.concatenate((np.full((1, test_class_a.shape[1]),1, dtype=int), np.full((1, test_class_b.shape[1]), 2, dtype=int)), axis = 1)
    true_label_2 = np.concatenate((np.full((1, test_class_c.shape[1]),1, dtype=int), np.full((1, test_class_d.shape[1]), 2, dtype=int), np.full((1, test_class_e.shape[1]), 3, dtype=int)), axis = 1)

    test_1 = np.concatenate((test_class_a,test_class_b), axis = 1)
    test_2 = np.concatenate((test_class_c,test_class_d,test_class_e), axis = 1)
    test_x_1 = test_1[0].reshape(test_1.shape[1],1)
    test_y_1 = test_1[1].reshape(test_1.shape[1],1)
    test_x_2 = test_2[0].reshape(test_2.shape[1],1)
    test_y_2 = test_2[1].reshape(test_2.shape[1],1)


    # error analysis for GED classifiers
    test_dis_matrix_ged_1 = GED_classifier(mean_class_a, mean_class_b, cov_class_a, cov_class_b, test_x_1, test_y_1)
    test_class_matrix_ged_1 = binary_class_matrix(test_dis_matrix_ged_1).T

    test_dis_matrix_ged_cd = GED_classifier(mean_class_c, mean_class_d, cov_class_c, cov_class_d, test_x_2, test_y_2)
    test_dis_matrix_ged_de = GED_classifier(mean_class_d, mean_class_e, cov_class_d, cov_class_d, test_x_2, test_y_2)
    test_dis_matrix_ged_ce = GED_classifier(mean_class_c, mean_class_e, cov_class_c, cov_class_e, test_x_2, test_y_2)
    test_class_matrix_ged_2 = three_class_matrix(test_dis_matrix_ged_cd, test_dis_matrix_ged_de, test_dis_matrix_ged_ce).T

    ged_error_1, ged_confusion_1 = error_analysis(true_label_1, test_class_matrix_ged_1)
    ged_error_2, ged_confusion_2 = error_analysis(true_label_2, test_class_matrix_ged_2)

    print("(A vs. B, GED) Error is: %s"%(ged_error_1))
    print("(C vs. D vs. E, GED) Error is: %s"%(ged_error_2))

    plot_confusion_matrix(true_label_1, test_class_matrix_ged_1, ged_confusion_1, classes=np.array(classes_1), title='ged Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plot_confusion_matrix(true_label_2, test_class_matrix_ged_2, ged_confusion_2, classes=np.array(classes_2), title='ged Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plt.show()

    # error analysis for MAP classifiers
    test_dis_matrix_map_1 = MAP_classifier(mean_class_a, mean_class_b, cov_class_a, cov_class_b, test_x_1, test_y_1, n_a, n_b)
    test_class_matrix_map_1 = binary_class_matrix(test_dis_matrix_map_1).T

    test_dis_matrix_map_cd = MAP_classifier(mean_class_c, mean_class_d, cov_class_c, cov_class_d, test_x_2, test_y_2, n_c, n_d)
    test_dis_matrix_map_de = MAP_classifier(mean_class_d, mean_class_e, cov_class_d, cov_class_d, test_x_2, test_y_2, n_d, n_e)
    test_dis_matrix_map_ce = MAP_classifier(mean_class_c, mean_class_e, cov_class_c, cov_class_e, test_x_2, test_y_2, n_c, n_e)
    test_class_matrix_map_2 = three_class_matrix(test_dis_matrix_map_cd, test_dis_matrix_map_de, test_dis_matrix_map_ce).T

    map_error_1, map_confusion_1 = error_analysis(true_label_1, test_class_matrix_map_1)
    map_error_2, map_confusion_2 = error_analysis(true_label_2, test_class_matrix_map_2)

    print("(A vs. B, MAP) Error is: %s"%(map_error_1))
    print("(C vs. D vs. E, MAP) Error is: %s"%(map_error_2))

    plot_confusion_matrix(true_label_1, test_class_matrix_map_1, map_confusion_1, classes=np.array(classes_1), title='MAP Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plot_confusion_matrix(true_label_2, test_class_matrix_map_2, map_confusion_2, classes=np.array(classes_2), title='MAP Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plt.show()

    # error analysis for MED classifiers
    test_dis_matrix_med_1 = MED_classifier(mean_class_a, mean_class_b, test_x_1, test_y_1)
    test_class_matrix_med_1 = binary_class_matrix(test_dis_matrix_med_1).T

    test_dis_matrix_med_cd = MED_classifier(mean_class_c, mean_class_d, test_x_2, test_y_2)
    test_dis_matrix_med_de = MED_classifier(mean_class_d, mean_class_e, test_x_2, test_y_2)
    test_dis_matrix_med_ce = MED_classifier(mean_class_c, mean_class_e, test_x_2, test_y_2)
    test_class_matrix_med_2 = three_class_matrix(test_dis_matrix_med_cd, test_dis_matrix_med_de, test_dis_matrix_med_ce).T

    med_error_1, med_confusion_1 = error_analysis(true_label_1, test_class_matrix_med_1)
    med_error_2, med_confusion_2 = error_analysis(true_label_2, test_class_matrix_med_2)

    print("(A vs. B, MED) Error is: %s"%(med_error_1))
    print("(C vs. D vs. E, MED) Error is: %s"%(med_error_2))

    plot_confusion_matrix(true_label_1, test_class_matrix_med_1, med_confusion_1, classes=np.array(classes_1), title='MED Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plot_confusion_matrix(true_label_2, test_class_matrix_med_2, med_confusion_2, classes=np.array(classes_2), title='MED Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plt.show()

    # error analysis for NN classifiers (k=1)
    test_dis_matrix_nn_1 = KNN_classifier(x_class_a, x_class_b, test_x_1, test_y_1)
    test_class_matrix_nn_1 = binary_class_matrix(test_dis_matrix_nn_1).T

    test_dis_matrix_nn_cd = KNN_classifier(x_class_c, x_class_d, test_x_2, test_y_2)
    test_dis_matrix_nn_de = KNN_classifier(x_class_d, x_class_e, test_x_2, test_y_2)
    test_dis_matrix_nn_ce = KNN_classifier(x_class_c, x_class_e, test_x_2, test_y_2)
    test_class_matrix_nn_2 = three_class_matrix(test_dis_matrix_nn_cd, test_dis_matrix_nn_de, test_dis_matrix_nn_ce).T

    nn_error_1, nn_confusion_1 = error_analysis(true_label_1, test_class_matrix_nn_1)
    nn_error_2, nn_confusion_2 = error_analysis(true_label_2, test_class_matrix_nn_2)

    print("(A vs. B, NN) Error is: %s"%(nn_error_1))
    print("(C vs. D vs. E, NN) Error is: %s"%(nn_error_2))

    plot_confusion_matrix(true_label_1, test_class_matrix_nn_1, nn_confusion_1, classes=np.array(classes_1), title='NN Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plot_confusion_matrix(true_label_2, test_class_matrix_nn_2, nn_confusion_2, classes=np.array(classes_2), title='NN Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plt.show()

    # error analysis for kNN classifiers (k=5)
    test_dis_matrix_knn5_1 = KNN_classifier(x_class_a, x_class_b, test_x_1, test_y_1, 5)
    test_class_matrix_knn5_1 = binary_class_matrix(test_dis_matrix_knn5_1).T

    test_dis_matrix_knn5_cd = KNN_classifier(x_class_c, x_class_d, test_x_2, test_y_2, 5)
    test_dis_matrix_knn5_de = KNN_classifier(x_class_d, x_class_e, test_x_2, test_y_2, 5)
    test_dis_matrix_knn5_ce = KNN_classifier(x_class_c, x_class_e, test_x_2, test_y_2, 5)
    test_class_matrix_knn5_2 = three_class_matrix(test_dis_matrix_knn5_cd, test_dis_matrix_knn5_de, test_dis_matrix_knn5_ce).T

    knn5_error_1, knn5_confusion_1 = error_analysis(true_label_1, test_class_matrix_knn5_1)
    knn5_error_2, knn5_confusion_2 = error_analysis(true_label_2, test_class_matrix_knn5_2)

    print("(A vs. B, 5NN) Error is: %s"%(knn5_error_1))
    print("(C vs. D vs. E, 5NN) Error is: %s"%(knn5_error_2))

    plot_confusion_matrix(true_label_1, test_class_matrix_knn5_1, knn5_confusion_1, classes=np.array(classes_1), title='5-NN Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plot_confusion_matrix(true_label_2, test_class_matrix_knn5_2, knn5_confusion_2, classes=np.array(classes_2), title='5-NN Confusion Matrix')
    plt.autoscale(enable=True, axis='y')
    plt.show()

if __name__ == "__main__":
    main()