import numpy as np

data = np.array([
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
], dtype=object)

# Encoding the dataset
X = data[:,0:3]
y = data[:,3]
n_labels = len(np.unique(y))

# Converting string labels to numeric equivalent
y[y=='Wine']=0
y[y=='Whiskey']=1
y[y=='Beer']=2

y = np.array(y,dtype=int)

convert = {0:'Wine',1:'Whiskey',2:'Beer'}

headings = {0:'Alcohol Content(%)',1:"Sugar(g/L)",2:"Color"}

#print(X,y) #debug

# Gini Impurity function
def gini_impurity(X,y,fi,thr): #fi ---> feature index, thr ---> threshold of a node
    quan = np.zeros((2,n_labels), dtype=int) #storing no. of each type of feature for calculating probabilities #len(np.unique(y))
    for i in range(len(X)):
        if X[i,fi]<=thr:
            quan[0,y[i]]+=1
        else:
            quan[1,y[i]]+=1
    #print(quan) #debug

    #first, calculating the gini impurity for the node itself
    gini=1 - np.sum(np.square(np.sum(quan,axis=0)))/(np.sum(quan))**2
    #print(gini) #debug

    #second, calculating the weighted gini impurity of left (<=threshold) and right (>threshold) sides
    wg=0
    wg+= 0 if np.sum(quan,axis=1)[0]==0 else np.sum(quan,axis=1)[0] / np.sum(quan) * (1 - (np.sum(np.square(quan),axis=1)[0]/(np.sum(quan,axis=1)[0])**2))
    wg+= 0 if np.sum(quan,axis=1)[1]==0 else np.sum(quan,axis=1)[1] / np.sum(quan) * (1 - (np.sum(np.square(quan),axis=1)[1]/(np.sum(quan,axis=1)[1])**2))
    #print(wg) #debug

    #third, calculating the gini impurity gain
    gain = gini-wg

    if gain<1e-9:
        dummy=np.arange(3, dtype=int)
        label = dummy[np.where(np.sum(quan,axis=0)!=0)][0] #getting the label number when gain = 0 (leaf node scenario)
        return gain,label

    return gain,None #using None as a dummy of sorts
    #print(gain) #debug

# Entropy function
def entropy(X,y,fi,thr): #fi ---> feature index, thr ---> threshold of a node
    quan = np.zeros((2,3), dtype=int) #storing no. of each type of feature for calculating probabilities #len(np.unique(y))
    for i in range(len(X)):
        if X[i,fi]<=thr:
            quan[0,y[i]]+=1
        else:
            quan[1,y[i]]+=1
    #print(quan) #debug

    #first, calculating the entropy impurity for the node itself
    ent = - (np.log(np.sum(quan,axis=0)[np.where(np.sum(quan,axis=0)!=0)]/np.sum(quan))@(np.sum(quan,axis=0)[np.where(np.sum(quan,axis=0)!=0)].T/np.sum(quan)))
    #print("ent-->",ent) #debug

    #second, calculating the weighted entropy of left (<=threshold) and right (>threshold) sides
    we=0
    #print(quan) #debug
    #print(quan[0][np.where(quan[0]!=0)]) #debug
    #print(quan[1][np.where(quan[1]!=0)]) #debug
    we+= 0 if np.sum(quan,axis=1)[0]==0 else np.sum(quan,axis=1)[0] / np.sum(quan) * (- (np.log(quan[0][np.where(quan[0]!=0)]/np.sum(quan,axis=1)[0])@(quan[0][np.where(quan[0]!=0)]).T//np.sum(quan,axis=1)[0]))
    #print(we) #debug
    we+= 0 if np.sum(quan,axis=1)[1]==0 else np.sum(quan,axis=1)[1] / np.sum(quan) * (- (np.log(quan[1][np.where(quan[1]!=0)]/np.sum(quan,axis=1)[1])@(quan[1][np.where(quan[1]!=0)]).T//np.sum(quan,axis=1)[1]))
    #print("we--->",we) #debug

    #third, calculating the entropy gain
    gain = ent-we

    if gain<1e-9:
        dummy=np.arange(3, dtype=int)
        label = dummy[np.where(np.sum(quan,axis=0)!=0)][0] #getting the label number when gain = 0 (leaf node scenario)
        return gain,label
    #print(gain) #debug
    return gain,None #using None as a dummy of sorts
    


# Best split determining function 
def best_split(X,y, mode=False):
    g = 0
    FI=THR=0
    z=[]
    for fi in range(0,3):  
        thr = np.min(X[:,fi])
        incr = 0.1 if type(thr)==float else 1
        while(thr <= np.max(X[:,fi])):
            g_calc,label = gini_impurity(X,y,fi,thr) if not mode else entropy(X,y,fi,thr) #switching b/w entropy and gini impurity
            #print("Test--->",fi,thr,g_calc,label)
            if g_calc >= g:
                g=g_calc
                FI=fi
                THR=thr
            if g_calc<1e-9:
                z.append(label)
            thr+=incr
        #if fi==2:
            #print("PING!!! g=",g) #debug
    if g<1e-9:
        return z[0],-1
        #print("z check",z) #debug
        #return 0,-1  #debug
    #print(f"Parameters in the end: max_gain={g}  FI={FI}  THR={THR}") #debug
    return FI,THR

# Defining the Node class
class Node():
    def __init__(self,depth,max_depth,mode=False):
        self.feature_index=None
        self.threshold=None
        self.left=None
        self.right=None
        self.value=None
        self.depth=depth
        self.max_depth = max_depth
        if depth==0:    
            print("CART Decision Tree!\n" if not mode else "ID3 Decision Tree!\n")
        self.mode = mode # mode:False makes use of Gini Impurity for a CART decision tree
                         # mode:True makes use of Entropy for an ID3 decision tree

    def tree_building(self,X,y):
        res = best_split(X,y, mode=self.mode)
        if res[1]==-1:
            #statements that make it a leaf node
            self.value = res[0]
            #print("Hey leaf here at depth", self.depth,"with label number",self.value) #debug
        elif self.depth==self.max_depth: #implementing max_depth parameter
            self.value = np.bincount(y).argmax()
        elif len(X)<2: #implementing lower limit on no. of samples
            self.value = np.bincount(y).argmax()
        else:
            self.feature_index,self.threshold = res
            self.left = Node(self.depth+1, self.max_depth)
            self.right = Node(self.depth+1, self.max_depth)
            condition1 = X[:,self.feature_index]<=self.threshold
            condition2 = X[:,self.feature_index]>self.threshold
            self.left.tree_building(X[condition1],y[condition1])
            self.right.tree_building(X[condition2],y[condition2])

    def tree_predict(self,x):
        if self.value is not None:
            return self.value
        else:
            if(x[self.feature_index]<=self.threshold):
                return self.left.tree_predict(x)
            else:
                return self.right.tree_predict(x)

    def tree_display(self): #Pretty-printing the tree
        if self.value is None:
            text1=text2=""
            if self.feature_index==2:
                text1=f"If {headings[2]}=={self.threshold} then"
                text2=f"If {headings[2]}=={int(not self.threshold)} then"
            else:
                text1=f"If {headings[self.feature_index]}<={round(self.threshold,2)} then"
                text2=f"If {headings[self.feature_index]}>{round(self.threshold,2)} then"
            print('     '*self.depth + "|"+'---'*self.depth+'>',text1)
            self.left.tree_display()
            print('     '*self.depth + "|"+'---'*self.depth+'>',text2)
            self.right.tree_display()
        else:
            print('     '*self.depth + "|"+'---'*self.depth+'>',convert[self.value])



Root_Node = Node(0,1,mode=True)

Root_Node.tree_building(X,y)

#Pretty printing the tree
Root_Node.tree_display()

print("\nPredictions for the given testing data:")

X_test = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])


for x in X_test:
    res = Root_Node.tree_predict(x)
    print(x,"--->",convert[res])


#gini_impurity(X,y,0,10) #debug
#best_split(X,y) #debug