# K-Means clustering implementation
# Major Resource Credit :
# https://medium.com/analytics-vidhya/from-pseudocode-to-python-code-k-means-clustering-from-scratch-2e32aa469bef
# Challenge: Opening CSV file 'dataBoth' and being able to 'differentiate' (2008) / (1953) - most of this program was
# derived from the resource. Credits to E Bauscher

# https://youtu.be/EItlUEPCIzM -> To improve my understanding of K-means clustering

"""My approach to this task was to initially go through all the documents provided for this task so I could determine
where best to start.

Initially, I make reference to the Task 22 Shell.py which had a clearer structure in terms of the specifications of
this task, however I found it overbaringly challenging to have dived into ML and more complex algorithms without more
examples - so I dove into the net, and various sites just to gain a better understanding.

I had to take a step back, refresh my mind and re-approach this task from a different perspective, this is where I
segmented the specifications of the task to determine what was required, and what would be the most appropriate order of
retrieving the required info and working on it.

                                            Pseudo_Code
  a) We obtain the file input from the user to determine which dataset will be used for plotting
  b) Convert the dataset to a workable list for sampling
  c) Get the number of clusters and iteration from the user
  d) Set random centroids for each cluster
  e) Present the raw data to the user indicating the starting centroids and datapoints from file
  f) - Use the k-means algorathm to determine appropriate centroid position and nearest datapoints:
    f)1 - Calculate the distance between each datapoint and centroid
    f)2 - Recalculate the new means for appropriate plotting of centroid
    f)3 - List country in dictionary for each country with its co-ordinates
  g) Count the number of countries, the average age, lifespan and total distance for each cluster
  h) Plot each datapoint in color of for each centroid according to cluster for each iteration -
  as specified by the user

"""

from math import sqrt  # For returning the square root of a number or calculation
import random  # For the choosing of our random centroids to begin with
import numpy as np  # For various data wrangling tasks
import matplotlib.pyplot as plt  # For plots to be generated
import seaborn as sns  # For plots, their styling and colouring.
import pandas as pd  # For reading of the data and drawing inferences from it

pd.options.display.max_rows = 4000  # To display all the rows of the data frame

# Obtain the file_name from the user
data_file = input('''Please enter the file name you want to use: 
                    data1953.csv
                    data2008.csv
                    dataBoth.csv \n> ''')


# Function to read file
def read_csv_pd(data_file):
    """This function reads a csv file with pandas, prints the dataframe and returns
    the two columns in numpy ndarray for processing as well as the country names in
    numpy array needed for cluster matched results"""
    data1 = pd.read_csv(data_file, delimiter=',')
    # Display the tabled data to the user
    print(data1)
    country_names = data1[data1.columns[0]].values
    list_array = data1[[data1.columns[1], data1.columns[2]]].values
    return list_array, country_names


# Function to calculate distance from datapoint to centroid
def distance_between(cent, data_points):
    """This function takes in the number of calculates the euclidean distance between each data point and each centroid.
    It appends all the values to a list and returns this list."""
    distances_arr = []  # create a list to which all distance calculations could be appended.
    for centroid in cent:
        for datapoint in data_points:
            distances_arr.append(sqrt((datapoint[0] - centroid[0]) ** 2 + (datapoint[1] - centroid[1]) ** 2))
    return distances_arr


# Assign the function for reading the csv to a variable as this will provide us with a Numpy array with all the values
# in the two columns as well as the country names in a separate array. We can access them via slicing.
file_name = read_csv_pd(data_file)

# convert the ndarray to a list for sampling
x_list = np.ndarray.tolist(file_name[0][0:, :])

# Get the input from the user for the number of clusters to be specified as well as the number of iterations that the
# algorithm must run
k = int(input("Please enter the number of clusters you want: "))
iterations = int(input("Please enter the number of iterations that the algorithm must run: "))

# Set the random number of centroids based on the user input value of "k".
centroids = random.sample(x_list, k)
print('Random Centroids are: ' + str(centroids))


# Function to assign datapoint to centroid and determine accurate centroid coordinates
def assign_to_cluster_mean_centroid(x_in=file_name, centroids_in=centroids, n_user=k):
    """This function calls the distance_between() function. The returned list values will be used to allocate each
    data-point to it's nearest centroid. It also rewrites the centroids with a newly calculated means.
    This function returns the list with cluster allocations that are in line with the order of the countries.
    It also returns the clusters in a dictionary."""
    distances_arr_re = np.reshape(distance_between(centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
    datapoint_cen = []
    distances_min = []  # Done if needed
    for value in zip(*distances_arr_re):
        distances_min.append(min(value))
        datapoint_cen.append(np.argmin(value) + 1)
    # Create clusters dictionary and add number of clusters according to user input
    clusters = {}
    for no_user in range(0, n_user):
        clusters[no_user + 1] = []
    # Allocate each data point to it's closest cluster
    for d_point, cent in zip(x_in[0], datapoint_cen):
        clusters[cent].append(d_point)
    # Run a for loop and rewrite the centroids with the newly calculated means
    for i, cluster in enumerate(clusters):
        reshaped = np.reshape(clusters[cluster], (len(clusters[cluster]), 2))
        centroids[i][0] = sum(reshaped[0:, 0]) / len(reshaped[0:, 0])
        centroids[i][1] = sum(reshaped[0:, 1]) / len(reshaped[0:, 1])
    print('Centroids for this iteration are:' + str(centroids))
    return datapoint_cen, clusters


# Present the initial plotted figure
# Create a scatter plot of the data without clustering
plt.scatter(file_name[0][0:, 0], file_name[0][0:, 1])
plt.xlabel('Birthrate')
plt.ylabel('Life Expectancy')
plt.title('Data Points with random centroids\nNo data point allocation')
cv = np.reshape(centroids, (k, 2))
plt.plot(cv[0:, 0], cv[0:, 1],
         c='#000000', marker="*", markersize=15, linestyle=None, linewidth=0)
plt.show()

# set the font size of the labels on matplotlib
plt.rc('font', size=10)

# =========
# MAIN LOOP - runs each clustering iteration
# =========
for iteration in range(0, iterations):
    # Print the iteration number
    print("ITERATION: " + str(iteration + 1) + "\n")
    # assign the function to a variable as it has more than one return value
    assigning = assign_to_cluster_mean_centroid()
    # Create the dataframe for visualization
    cluster_data = pd.DataFrame({'Birth Rate': file_name[0][0:, 0],
                                 'Life Expectancy': file_name[0][0:, 1],
                                 'label': assigning[0],
                                 'Country': file_name[1]})

    # Create the dataframe and grouping, then print out inferences
    group_by_cluster = cluster_data[['Country', 'Birth Rate', 'Life Expectancy', 'label']].groupby('label')
    count_clusters = group_by_cluster.count()
    # Inference 1
    print("COUNTRIES PER CLUSTER: \n" + str(count_clusters))
    print()
    # Inference 2
    print("LIST OF COUNTRIES PER CLUSTER: \n",
          list(group_by_cluster))
    # Inference 3
    print()
    print("AVERAGES: \n", str(cluster_data.groupby(['label']).mean()))
    print()
    # Set the variable mean that holds the clusters dict
    mean = assigning[1]

    # create a dict that will hold the distances of between each data point in a particular cluster and it's mean.
    # The loop here will create the amount of clusters based on user input.
    means = {}
    for clst in range(0, k):
        means[clst + 1] = []

    # Create a for loop to calculate the squared distances between each
    # data point and its cluster mean
    for index, data in enumerate(mean):
        array = np.array(mean[data])
        array = np.reshape(array, (len(array), 2))
        # Set two variables, one for each variable in the data set that holds the calculation of the cluster mean of
        # each variable
        birth_rate = sum(array[0:, 0]) / len(array[0:, 0])
        life_exp = sum(array[0:, 1]) / len(array[0:, 1])

        # within this for loop, create another for loop that appends to the means dict the squared distance of between
        # each data point in its cluster and the cluster mean.
        for data_point in array:
            distance = sqrt(
                (birth_rate - data_point[0]) ** 2 + (life_exp - data_point[1]) ** 2)
            means[index + 1].append(distance)

    # create a list that will hold all the sums of the means in each of the clusters.
    total_distance = []
    for ind, summed in enumerate(means):
        total_distance.append(sum(means[ind + 1]))

    # print the summed distance
    print("Summed distance of all clusters: " + str(sum(total_distance)))
    # plot data with seaborn. This plot will show the clusters in colour with the centroids
    facet = sns.lmplot(data=cluster_data, x='Birth Rate', y='Life Expectancy', hue='label', fit_reg=False,
                       legend=False, legend_out=False)
    plt.legend(loc='upper right')
    centr = np.reshape(centroids, (k, 2))

    # centroids plot - without line joining centroids
    plt.plot(centr[0:, 0], centr[0:, 1], c='#000000', marker="*", markersize=10, linestyle=None, linewidth=0)
    plt.title('Iteration: ' + str(iteration + 1) + "\nSummed distance of all clusters: " +
              str(round(sum(total_distance), 0)))

    # print the sns plot
    plt.show()
