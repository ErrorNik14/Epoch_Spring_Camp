import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Extracting and preprocessing the data (removing the character artifacts in the longitudes and latitudes)
'''
Errors I noticed in the data-
(i) Some kind of special character at the end of the data (" E" or similar)
(ii) "Nan" type missing data
(iii) Duplicate data
(iv) Points lying outside India or the designated states

I was initially very uncertain about the data because of the sheer number of outliers like (iv), but I tried dealing with them 
to the extent I could...
'''
# Reading the .csv file
df = pd.read_csv("clustering_data.csv") 

# Extracting only the decimal parts, using regex
df[["Longitude", "Latitude"]] = df[["Longitude", "Latitude"]].apply(lambda col: col.str.extract(r'(\d*\.?\d+)')[0]) 


# Basic cleaning up of data
data = df.dropna().drop_duplicates(subset=['Latitude','Longitude'])  # Dropping all NaN values and duplicates
data = data[(data['StateName']=='TAMIL NADU')&(data['CircleName']=='Tamilnadu Circle')] # Selecting home state - Tamil Nadu
data = data.astype({'Longitude':float,'Latitude':float}) # Converting float


# Surface-level removal of points outside india (using latitude and longitude extent of the country borders)
la1,la2=8.4,37.6
lo1,lo2=68.7,97.25
data = data[(data['Latitude'].between(la1,la2))&(data['Longitude'].between(lo1,lo2))]


# Using IQR to try and remove more outliers that are very far away from dense regions
def remove_outlier_iqr(df:pd.DataFrame,col):
    L = df[col].quantile(0.25)
    U = df[col].quantile(0.75)
    IQR = U-L
    A = L - 1.5 * IQR
    B = U + 1.5 * IQR
    return df[df[col].between(A,B)]

data = remove_outlier_iqr(data,'Latitude')
data = remove_outlier_iqr(data,'Longitude')

'''
I thought of using the number of regions in the state for deciding the number of clusters
since I assumed that the distances between offices must follow some kind of region trend 
'''

# Having ensured all the values in X are string-floats and properly cleaned up, it is time to get started with the datasets and 
# K-Means clustering dataset
X = np.array(data[['Latitude','Longitude']],dtype=float)
y = np.array(data['Pincode'],dtype=int)

X_train = (X-np.mean(X,axis=0))/np.std(X,axis=0) # Normalising the data for better clustering


# K-Means Clustering class, with the attributes and methods
class kMeansCluster():
    def __init__(self, X:np.array, iter=100, thr=0.01, k=3):
        self.k = k
        self.iter = iter
        self.thr = thr
        self.X = X
        self.y = np.zeros(shape=(len(X),),dtype=int)
        self.SSE = []
    
    def distance(self, x1,x2):
        return np.sqrt(np.sum(np.square(x1-x2),axis=1))

    def closest_centre(self,x,centroids:np.array):
        distances = self.distance(centroids,x)
        label = np.argmin(distances)
        return label


    def classify(self):
        centroids = self.X[np.random.choice(self.X.shape[0],self.k,replace=False)]
        prev_centroids = np.zeros_like(centroids,dtype=float)
        #for iter in np.arange(self.iter):
        iters = 0
        # Adding a functionality that keeps running the k-means algo, until either the threshold is reached for all centres...
        # Or, the maximum no. of iterations is reached
        while ((self.distance(prev_centroids,centroids)>self.thr).any() and iters<=self.iter):
            iters+=1
            for i in np.arange(len(self.X)):
                self.y[i] = self.closest_centre(self.X[i],centroids)
            prev_centroids = np.copy(centroids)

            sse = 0 
            for j in np.arange(self.k):
                sse+=np.sum(np.square(self.X[self.y==j]-centroids[j])) # SSE calculation for each iteration of k-means
                centroids[j] = np.mean(self.X[self.y==j],axis=0)[0]
            self.SSE.append(sse)

        #print(iters)
        return self.y


# Follow segment is for finding the optimal "k" value 

SSE = []

for i in np.arange(1, 8):
    model = kMeansCluster(X_train,k=i,iter=200, thr=0.0001)
    model.classify()
    SSE.append(model.SSE[-1])

plt.plot(np.arange(1,8),SSE)
plt.show()


# We can conclude that k=4  is one of the optimal values, which we can use for our actual clustering visualisation.

model = kMeansCluster(X_train, k=4, iter=200, thr=0.00001) # Normalised training data, no. of clusters, and no. of iterations the algorithm runs
y_pred = model.classify()

plt.scatter(X_train[:,1],X_train[:,0],c=y_pred, linewidths=.2, edgecolors="black") # Our training data, with the predicted clusters


# For analysis sake, I am also plotting the coordinates of some of the major cities
X_major = np.array([[10.8155,78.6965],[13.0674,80.2376],[11.0045,76.9616],[9.9252,78.1198]])

X_major_z = (X_major-np.mean(X,axis=0))/np.std(X,axis=0)

plt.scatter(X_major_z[:,1],X_major_z[:,0],c="black", linewidths=1, edgecolors="green")
plt.xlabel("Normalised Longitude")
plt.ylabel("Normalised Latitude")
plt.title("K-Means Clustering for Tamil Nadu Post Offices")
plt.show()
