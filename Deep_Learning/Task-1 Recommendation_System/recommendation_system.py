import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Function to generate negative samples for each item
def sample_negatives(df, no_neg=1):
    all_items = set(df['item_id'].unique())
    interactions = df.groupby('user_id')['item_id'].apply(set).to_dict()
    l=[]
    for ui,iis in interactions.items():
        for ii in iis:
            l.append((ui,ii,1))
        
        neg_items = np.random.choice(list(all_items - iis), size=min(no_neg*len(iis), len(all_items - iis)), replace=False)
        for nii in neg_items:
            l.append((ui,nii,0))
    l_df = pd.DataFrame(l,columns=["user","item","label"])
    return l_df


# Defining Dataset for later data loading and batching
class InteractionDataset(Dataset):
    def __init__(self,df):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.users = torch.tensor(df['user']).to(device)
        self.items = torch.tensor(df['item']).to(device)
        self.labels = torch.tensor(df['label']).float().to(device)

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (self.users[idx],self.items[idx],self.labels[idx])
    

# Matrix Factorisation model
class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, emb_dim)
        self.item_embed = nn.Embedding(num_items, emb_dim)

    def forward(self, user, item):
        emb_user = self.user_embed(user)
        emb_item = self.item_embed(item)

        dot = torch.sum((emb_user*emb_item),axis=1)
        return torch.sigmoid(dot)



# Neural Collaborative Filtering model (MLP based)
class MLP(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        # Embedding layers
        self.user_embed = nn.Embedding(num_users, emb_dim)
        self.item_embed = nn.Embedding(num_items, emb_dim)
        # MLP layers
        self.mlp_seq = nn.Sequential(
                        nn.Linear(emb_dim*2, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128,32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32,8),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(8,1),
                        nn.Sigmoid()
                        )
        print("The NN is done initialising!")
    def forward(self, user, item):
        emb_user = self.user_embed(user)
        emb_item = self.item_embed(item)
        inp = torch.cat((emb_user,emb_item),dim=-1)
        res = self.mlp_seq(inp)
        return torch.flatten(res)


# MLP-MF hybrid model - Neural Matrix Factorisation model
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_emb_dim, mlp_emb_dim):
        super().__init__()
        # Embedding layers
        self.user_embed_mf  = nn.Embedding(num_users, mf_emb_dim) # different embedding dimensions exist to
        self.item_embed_mf  = nn.Embedding(num_items, mf_emb_dim) # force more weightages to make the models equal
        self.user_embed_mlp = nn.Embedding(num_users, mlp_emb_dim)
        self.item_embed_mlp = nn.Embedding(num_items, mlp_emb_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5)) # defining a parameter that decides the weightage
                                                     # of the MF vs MLP parts when concatenating

        # MLP layers
        self.mlp_seq = nn.Sequential(
                        nn.Linear(mlp_emb_dim*2, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128,32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32,8),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        )
        self.neumf_seq = nn.Sequential(
                        nn.Linear(8 + 1,1),   # the +1 is present due to us concatenating the dot product
                        nn.Sigmoid()
                        )
        print("The NN is done initialising!")
    def forward(self, user, item):
        mf_emb_user = self.user_embed_mf(user)
        mf_emb_item = self.item_embed_mf(item)
        dot = torch.sum((mf_emb_user*mf_emb_item),axis=1).unsqueeze(1)

        mlp_emb_user = self.user_embed_mlp(user)
        mlp_emb_item = self.item_embed_mlp(item)
        inp = torch.cat((mlp_emb_user,mlp_emb_item),dim=-1)
        res = self.mlp_seq(inp)

        inp2 = torch.cat((self.alpha*dot,(1-self.alpha)*res),dim=-1)
        res2 = self.neumf_seq(inp2)
        return torch.flatten(res2)


# Defining training function
def train(model, dataloader, epochs, l_rate): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Moving to cuda accelerator if possible
    model.to(device)
    model.train()
    loss_fn = nn.BCELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=l_rate) # Stochastic Gradient Descent
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=1e-4) # Adaptive Moment Estimation
    losses = []
    epoch=0
    while epoch<epochs: # Gradient descent over epochs
        total_loss = 0
        for user, item, label in dataloader: # Retrieving features for training
            pred = model.forward(user, item)
            loss = loss_fn(pred,label)
            total_loss += loss.item()

            loss.backward() # beginning backprop
            optimizer.step()
            optimizer.zero_grad()
        
        if losses!=[] and total_loss>losses[-1]:
            break
        losses.append(total_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        epoch+=1
    return losses


# Evaluation parameters - Hit@k and Accuracy
def hit_at_k(model, test_df, full_df, K=10, num_neg=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    hits = 0
    acc_hits = 0
    total = len(test_df) # We evaluate every positive interaction in the test set

    # Map interactions for quick lookup
    interacted_items = full_df.groupby('user_id')['item_id'].apply(set).to_dict()
    all_items = full_df['item_id'].unique()

    with torch.no_grad(): # Disable gradient calculation for speed/memory
        for _, row in test_df.iterrows():
            u = int(row['user_id'])
            pos_item = int(row['item_id'])

            # 1. Sample Negatives
            negatives = []
            while len(negatives) < num_neg:
                neg_item = np.random.choice(all_items)
                if neg_item not in interacted_items.get(u, set()):
                    negatives.append(neg_item)

            # 2. Prepare Tensors
            # We need a list of the 1 positive + 100 negatives
            item_list = [pos_item] + negatives
            user_tensor = torch.tensor([u] * (num_neg + 1)).to(device)
            item_tensor = torch.tensor(item_list).to(device)

            # 3. Get Scores
            scores = model(user_tensor, item_tensor)

            # 4. Rank and Check Hit
            # We want to see if the item at index 0 (the positive) is in the top K
            # torch.topk returns values and indices of the highest scores
            _, top_indices = torch.topk(scores, K)

            top_indices = top_indices.cpu().numpy()
            if 0 in top_indices:
                hits += 1
            if top_indices[0]==0:
                acc_hits += 1

    return hits/total, acc_hits/total




# Retrieving data from the csv file
df = pd.read_csv('interactions.csv',)
user_counts = df['user_id'].value_counts() # 942 unique users
item_counts = df['item_id'].value_counts() # 1447 unique items

# Preparing training and testing data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # train-test data splitting @ 80-20
train_data = sample_negatives(train_df, no_neg=25) # sampling negatives
train_loader = DataLoader(InteractionDataset(train_data), batch_size=256, shuffle=True)


# Initialising, training, and evaluating the models!
# mfm = MF(len(user_counts), len(item_counts), emb_dim=100) # MF model
# train(mfm, train_loader, epochs=20, l_rate=5e-4)
# hitatk,acc = hit_at_k(mfm, test_df, df, K=10, num_neg=100)

# mlp = MLP(len(user_counts), len(item_counts), emb_dim=100) # MLP model
# train(mlp, train_loader, epochs=20, l_rate=5e-4)
# hitatk,acc = hit_at_k(mlp, test_df, df, K=10, num_neg=100)

nmf = NeuMF(len(user_counts), len(item_counts), mf_emb_dim=100, mlp_emb_dim=50) # NeuMF model
train(nmf, train_loader, epochs=20, l_rate=5e-4)
hitatk,acc = hit_at_k(nmf, test_df, df, K=10, num_neg=100)

print("Hit@K=",hitatk)
print("Accuracy=",round(acc*100,3),"%")
