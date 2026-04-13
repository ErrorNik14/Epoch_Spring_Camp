import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana'],
    [118, 6.2, 0, 'Banana'],
    [160, 7.3, 1, 'Apple'],
    [185, 7.7, 2, 'Orange']
],dtype=object) # I thought of combining the testing data here and then implementing the testing-training data split...

# Encoding datasets
X = data[:,0:3].astype(float)
slabels = np.unique(data[:,3]) # set of string labels
y = np.column_stack((data[:,3]=='Apple',data[:,3]=='Banana',data[:,3]=='Orange')).astype(int) #one-hot encoding

# Performing training-testing split of 80-20
X_test = X[int(len(X)*0.8):,]
y_test = y[int(len(y)*0.8):,]
X_final = X[0:int(len(X)*0.8),]
y_final = y[0:int(len(y)*0.8),]


# Data pre-processing (Z-normalisation)
X_train = (X_final-np.mean(X_final,axis=0))/np.std(X_final,axis=0)


# Making a function that converts one-hot representation to string label
def label(y):
    return slabels[np.arange(len(slabels))[np.where(y==1)]][0]


 # Distance calculating function
def distance(p1,p2,mode=1):
    if mode==1:
        return np.sqrt(np.sum(np.square(p1-p2))) # Returns Euclidean distance b/w p1 and p2
    elif mode==2:
        return np.sum((np.fabs(p1-p2))) # Returns Manhattan distance b/w p1 and p2
    elif mode==3:
        p = 3   # Minkowski parameter
        return np.power(np.sum(np.power(p1-p2,p)),1/p) # Returns Minkowski distance b/w p1 and p2

 # KNN Model class
class KNN:
    def __init__(self,k=3,weighted=False,distanceMode=1):
        self.k = k
        self.weighted= weighted
        self.distanceMode = distanceMode
        self.X = np.empty((0,0))
        self.y = np.empty((0,0))

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict_one(self,x):
        distances = [distance(x,p,mode=self.distanceMode) for p in self.X]
        dist_ind = np.argsort(distances)
        nearby_labels = self.y[dist_ind]
        k_nearest = nearby_labels[0:self.k,]
                
        # Implementing Weighted KNN option. I am using the Gaussian kernel for the weights
        if self.weighted:
            sigma = 0.5 # Power used for Gaussian kernel function
            weights = np.exp(-np.square(distances)/2/np.square(sigma))
            contr = np.array([np.sum(weights[np.where((self.y==l).all(axis=1))[0]]) for l in np.unique(k_nearest,axis=0)])

        else:
            label, contr = np.unique(k_nearest,return_counts=True,axis=0)
        # contr is the array holding the contribution of each label
        pred = k_nearest[np.argmax(contr)] # Pulling the label with the greatest contribution/vote in contr
        return pred

    def predict(self,X_test):
        y = np.empty((0,3))
        for x in X_test:
            y = np.vstack([y, self.predict_one(x)])
        return y





# Testing the KNN Model
X_test_zn = (X_test.astype(float)-np.mean(X_final,axis=0))/np.std(X_final,axis=0) #Z-Normalisation of testing data with reference to training data

model = KNN(k=7, weighted=False, distanceMode=1) # weighted=False ---> normal KNN ; weighted=True ---> weighted KNN
                                                # distanceMode : 1 - Euclid ; 2 - Manhattan ; 3 - Minkowski
model.fit(X_train,y_final)

y_pred = model.predict(X_test_zn)

print("KNN Classifier!")

for i in range(len(X_test_zn)):
    print(X_test[i],'--->',label(y_pred[i]), f"({label(y_test[i])})")

print(f"Model accuracy = {round(np.mean(y_pred==y_test)*100,2)}%") # Model accuracy checker



# Plotting the training and predicted data points
color=[]
for l in y_final:
    c = 'red' if l[0]==1 else ('yellow' if l[1]==1 else 'orange')
    color.append(c)

plt.scatter(X_final[:,0], X_final[:,1], c=color, edgecolors='black', linewidths=0.5) # Plotting training data


color=[]
for l in y_pred:
    c = 'red' if l[0]==1 else ('yellow' if l[1]==1 else 'orange')
    color.append(c)

plt.scatter(X_test[:,0], X_test[:,1], c=color, edgecolors='black', linewidths=0.5, marker='D') # Plotting predictions for reference

plt.xlabel("Weight(g)")
plt.ylabel("Size(cm)")
plt.title("KNN Classifier")
plt.show()
