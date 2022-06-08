# Andrew Frisby COMP527 CA2

# load libaries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(66)
np.random.seed(66)
rseed = 208

# load, label, and combine data
countries_fpath = 'countries'
veggies_fpath = 'veggies'
fruit_fpath = 'fruits'
animals_fpath = 'animals'

countries_data = pd.read_csv(countries_fpath, header=None, delimiter=' ').values
countries_data = np.insert(countries_data, 1, 'country', axis=1)

veggies_data = pd.read_csv(veggies_fpath, header=None, delimiter=' ').values
veggies_data = np.insert(veggies_data, 1, 'veggie', axis=1)

fruit_data = pd.read_csv(fruit_fpath, header=None, delimiter=' ').values
fruit_data = np.insert(fruit_data, 1, 'fruit', axis=1)

animals_data = pd.read_csv(animals_fpath, header=None, delimiter=' ').values
animals_data = np.insert(animals_data, 1, 'animal', axis=1)

combined_data = np.concatenate((countries_data, veggies_data, fruit_data, animals_data))

# generate normalized data
norm_data = combined_data.copy()

norms = np.linalg.norm(norm_data[:,2:].astype(float), axis=0)
norm_data[:,2:] = norm_data[:,2:]/norms

"""
Helper Functions
"""

def dist_calc(x1, x2):
    """
    Function to calculate Euclidean distance given two points.

    Parameters:
        x1: np.array - data object/row
        x2: np.array - data object/row

    Return:
        distance: float - Euclidean distance between x1 and x2
    """

    return np.linalg.norm(x1 - x2)

def man_dist_calc(x1, x2):
    """
    Function to calculate Manhattan distance given two points.

    Parameters:
        x1: np.array - data object/row
        x2: np.array - data object/row

    Return:
        distance: float - Manhattan distance between x1 and x2
    """

    return sum(abs(ai - bi) for ai, bi in zip(x1, x2))

def assign(d, centroids, man_dist=False):
    """
    Function to assign data points to nearest centroid. Can use either Euclidean or Manhattan
    distance depending on Boolean value of man_dist parameter.

    Parameters:
        d: np.array - data array
        centroids: np.array - array of centroids 
        man_dist: Boolean - what distance function to use. True corresponds to Manhattan distance,
        False corresponds to Euclidean distance (default value)

    Return:
        assigned_centroids: list - list of assigned centroids/clusters, values are integers 
        from 0 to k-1, index matches the index of the d data array
    """
    # create empty to populate with assigned clusters/centroids
    assigned_centroids = []
    # loop through each row in data
    for row in d:
        # create empty list to populate with distances from the point to each centroid
        cent_dists = []
        # loop through each centroid
        for cent in centroids:
            # calculate distance (either manhattan or Euclidean) and append to cent_dists
            if man_dist:
                cent_dists.append(man_dist_calc(row, cent))
            else:
                cent_dists.append(dist_calc(row, cent))

        # find centroid that is closest to point
        assigned_cent = np.argmin(cent_dists)

        # append centroid/cluster label to assigned_centroids
        assigned_centroids.append(assigned_cent)

    return assigned_centroids

def obj_function(d, cents, man_dist=False):
    """
    Function to calculate the objective function, i.e., the sum of all the distances from the objects
    in the d data array to their assigned centroids.

    Parameters:
        d: np.array - data array
        cents: np.array - array of centroids
        man_dist: Boolean - what distance function to use. True corresponds to Manhattan distance,
        False corresponds to Euclidean distance (default value)

    Return:
        obj: float - total sum of the distances between data objects and their centroids
    """

    obj = 0
    for row in d:

        # cluster label (0, 1, 2, ..., k-1) for row
        rcluster = int(row[0])

        # calculate distance and add to obj
        if man_dist:
            obj += man_dist_calc(row[1:], cents[rcluster])
        else:
            obj += dist_calc(row[1:], cents[rcluster])

    return obj


def get_accuracy(data):
    """
    Function to calculate the accuracy scores (precision, recall, F1 score) of the clustering 
    algorithm.

    Parameters:
        data: np.array - data array that includes the true labels and cluster assignments in the
        first two indices respectively of the data objects

    Return:
        acc_list: list - list of lists that contains the 3 accuracy scores for the data
    """
    fin_data = data.copy()

    # create empty lists for each accuracy measure
    prec_list = []
    recall_list = []
    fscore_list = []

    # loop through each row and calculate its precision, recall, and F1 score
    # append these values to lists that were created above
    for row in data:
        rlabel = row[0]
        rcluster = row[1]

        label_size = len(fin_data[fin_data[:,0] == rlabel])
        cluster_size = len(fin_data[fin_data[:,1] == rcluster])
        
        criteria = (fin_data[:,0] == rlabel) & (fin_data[:,1] == rcluster)
        data_criteria = fin_data[criteria]
        total_label = len(data_criteria)
        
        # precision calculation
        prec = total_label/cluster_size
        prec_list.append(prec)

        # recall calculation
        recall = total_label/label_size
        recall_list.append(recall)

        # F1 score calculation
        fscore = (2 * prec * recall) / (prec + recall)
        fscore_list.append(fscore)

    # calculate the mean of each list to get overall accuracy scores
    prec_mean = np.mean(prec_list)
    recall_mean = np.mean(recall_list)
    f1_mean = np.mean(fscore_list)

    # compile scores to one list 
    acc_list = [prec_mean, recall_mean, f1_mean]

    return acc_list

def get_acc_dict(data, clusterting_alg='kmeans', iterations=20):
    """
    Function to generate dictionary of accuracy scores (precision, recall, F1) for different values
    of k (hard-coded as 1-9 for this assignment).

    Parameters:
        data: np.array - data array that includes the true labels in the first index of the data objects
        clustering_alg: str - which clustering algorithm to use. Must either be 'kmedians' or 
        'kmeans' (default)
        iterations: int - the number of iterations to pass to the clustering algorithms, default=20

    Return:
        acc_dict: dict - dictionary where the keys are accuracy score names and the values are lists
        of the accuracy scores for each run of k, 1 to 9
    """

    # generate accuracy dictionary where keys are accuracy names and values will be lists of 
    # the accuracy scores over all k, 1-9
    keys = ['precision_values', 'recall_values', 'fscore_values']
    acc_dict = {key:[] for key in keys}
    
    # loop through each value of k and run the appropriate clustering algorithm, get
    # its accuracies, and append them to the relevant acc_dict key value
    for k in range(1,10):

        if clusterting_alg == 'kmeans':
            loop_data = kmeans(data[:,1:], k, iterations)
        
        else:
            loop_data = kmedians(data[:,1:], k, iterations)
        
        k_accs = get_accuracy(loop_data)

        acc_dict['precision_values'].append(k_accs[0])
        acc_dict['recall_values'].append(k_accs[1])
        acc_dict['fscore_values'].append(k_accs[2])
    
    return acc_dict


def visualize(score_dict, title='Graph', savefig=False):
    """
    Function to visualize the results of the clustering from get_acc_dict()

    Parameters:
        score_dict: dict - dictionary where the keys are accuracy score names and the values are lists
        of the accuracy scores for each run of k, 1 to 9
        title: str - title of the graph
        savefig: Boolean - whether to save the visualization to the current directory. Will use title
        as file name. Default is False

    Return:
        None - generates visualization, will save to directory if savefig set to True
    """
    
    # plotting accuracy scores
    plt.plot(np.arange(start=1, stop=10), score_dict['precision_values'], label='Precision')
    plt.plot(np.arange(start=1, stop=10), score_dict['recall_values'], label='Recall')
    plt.plot(np.arange(start=1, stop=10), score_dict['fscore_values'], label='F1')
    
    # creating and generating table of values
    row_headers = [f'K={i}' for i in range(1,10)]
    column_headers = ['Precision', 'Recall', 'F1 Score']
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    cell_text = [list(np.around(np.array(score_dict[key]),5)) for key in score_dict.keys()]
    cell_text_trans = list(map(list, zip(*cell_text)))
    
    plt.table(cellText=cell_text_trans,
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='right',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='right',
                      bbox=[1.1, 0.2, 0.5, 0.8])

    plt.ylabel('Score')
    plt.xlabel('K Value')
    plt.title(title)
    plt.legend()

    if savefig:
        plt.savefig(f'{title}.png', bbox_inches='tight')
    
    plt.show()


"""
Question 1
"""

def kmeans(d, k, iterations):
    """
    Function to cluster data using the K-means methodology

    Parameters:
        d: np.array - data array that includes the true labels in the first index of the data objects
        k: int - number of clusters
        iterations: int - number of maximum iterations to loop through. Will break early if local
        optimum is reached

    Return:
        return_data: np.array - same array as d array, but with the final cluster assignments inserted
        into the second index of each object
    """

    random.seed(rseed)
    np.random.seed(rseed)
    data = d[:,1:]

    # initialize centroids from randomly chosen points in data
    init_cents_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[init_cents_idx, :]

    # make initial assignments using initial centroids
    cluster_assignments = assign(data, centroids)
    insert_data = np.insert(data, 0, cluster_assignments, axis=1)

    # calculate initial objective function
    best_obj = obj_function(insert_data, centroids)


    counter = 0
    for _ in range(iterations):
        counter += 1

        # calculate the mean of each cluster and update its centroid value to that mean
        new_centroids = []
        # loop through each cluster
        for c_label in range(k):
            # get all rows that belong to that cluster
            cluster_rows = insert_data[insert_data[:,0] == c_label][:,1:]
            # find the mean of those rows
            cluster_mean = np.mean(cluster_rows, axis=0)
            # append the new mean centroid to list
            new_centroids.append(cluster_mean)

        # convert to np.array
        new_centroids = np.asarray(new_centroids)

        # create temporary cluster assignments with new, mean centroid
        temp_cluster_assignments = assign(data, new_centroids)

        # create temporary data set with temporary cluster assignment labels for each row
        temp_insert_data = np.insert(data, 0, temp_cluster_assignments, axis=1)

        # calculate objective function score for temporary centroids
        temp_obj = obj_function(temp_insert_data, new_centroids)


        # if the temporary obj score is less than the previous best obj score, then update values
        if temp_obj < best_obj:
            centroids = new_centroids
            best_obj = temp_obj
            cluster_assignments = temp_cluster_assignments
            insert_data = temp_insert_data
        # else, clustering has reached local optimum , break the loop
        else:
            print(f'breaking at {counter} iterations')
            break
        
    # create data with final cluster assignments
    return_data = np.insert(d, 1, cluster_assignments, axis=1)

    # print final obj score 
    print('final obj score: ', best_obj)
    return return_data

"""
Question 2
"""

def kmedians(d, k, iterations=20):
    """
    Function to cluster data using the K-medians methodology

    Parameters:
        d: np.array - data array that includes the true labels in the first index of the data objects
        k: int - number of clusters
        iterations: int - number of maximum iterations to loop through. Will break early if local
        optimum is reached

    Return:
        return_data: np.array - same array as d array, but with the final cluster assignments inserted
        into the second index of each object
    """

    random.seed(rseed)
    np.random.seed(rseed)
    data = d[:,1:]

    # initialize centroids
    init_cents_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[init_cents_idx, :]

    # make initial assignments using Manhattan distance
    cluster_assignments = assign(data, centroids, man_dist=True)
    insert_data = np.insert(data, 0, cluster_assignments, axis=1)

    # calculate initial objective function score with Manhattan distance
    best_obj = obj_function(insert_data, centroids, man_dist=True)

    counter = 0
    for _ in range(iterations):
        counter += 1

        # same as Kmeans algorithm, but use np.median instead of np.mean
        new_centroids = []
        for c_label in range(k):
            cluster_rows = insert_data[insert_data[:,0] == c_label][:,1:]
            cluster_med = np.median(cluster_rows, axis=0)
            new_centroids.append(cluster_med)
        
        new_centroids = np.asarray(new_centroids)

        # create temporary assignments with median clusters and Manhattan distance
        temp_cluster_assignments = assign(data, new_centroids, man_dist=True)
        temp_insert_data = np.insert(data, 0, temp_cluster_assignments, axis=1)
        temp_obj = obj_function(temp_insert_data, new_centroids, man_dist=True)
        
        if temp_obj < best_obj:
            centroids = new_centroids
            best_obj = temp_obj
            cluster_assignments = temp_cluster_assignments
            insert_data = temp_insert_data
        
        else:
            print(f'breaking at {counter} iterations')
            break
    
    print('final obj score: ', best_obj)
    return_data = np.insert(d, 1, cluster_assignments, axis=1)
    
    return return_data

"""
Code below will produce the visualizations for questions 3-6. May have to adjust 'right' slider
with the 'Configure Subplots' button on the bottom of the visualization to see table. 
"""

"""
Question 3
"""

# Q3
q3_dict = get_acc_dict(combined_data, iterations=25)
visualize(q3_dict, title='K-Means Scores')

"""
Question 4
"""

# Q4
q4_dict = get_acc_dict(norm_data)
visualize(q4_dict, title='Normalized K-Means Scores')

"""
Question 5
"""

# Q5
q5_dict = get_acc_dict(combined_data, clusterting_alg='kmedians')
visualize(q5_dict, title='K-Medians Scores')

"""
Question 6
"""

# Q6
q6_dict = get_acc_dict(norm_data, clusterting_alg='kmedians')
visualize(q6_dict, title='Normalized K-Medians Scores')

