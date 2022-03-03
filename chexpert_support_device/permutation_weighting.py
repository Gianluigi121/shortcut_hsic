"""Chexpert permutation weighting methods."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

"""
1. Permute the dataset
2. Stack the dataset
3. Build a classifier
4. Get the permutation weight
5. Repeat the above process to get the average weighting
"""

def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print(f'Error: {100 * error}%')
    return error

def get_clf(train_x, train_y, test_x, test_y):
    clf = LogisticRegression()
    train_y = train_y.reshape(train_y.shape[0],)
    clf.fit(train_x, train_y)
    # check the accuracy
    pred = clf.predict(test_x).ravel()
    test_y = test_y.ravel()
    # print(f"pred is :{pred}")
    # print(f"test y is: {test_y}")
    evaluate(pred, test_y)
    return clf

# This is the case for y1 and y2 are independent
def get_pw_y1y2(clf):
    weights = {}

    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                  [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    prob = clf.predict_proba(x)  # Shape: 8*2
    name_list = ["000", "001", "010", "011", "100", "101", "110", "111"]
    for i in range(8):
        name = name_list[i]
        weights[name] = prob[i][1]/prob[i][0]
    return weights

# This is the case for y1 and y2 are not independent(we only care about y0 and y1)
def get_pw_y1(clf):
    weights = {}

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    prob = clf.predict_proba(x)     # Shape: 4*2
    name_list = ["00", "01", "10", "11"]
    for i in range(4):
        name = name_list[i]
        weights[name] = prob[i][1]/prob[i][0]
    return weights

def pw(data, permute_pairwise=True):
    # Data columns: file_name, y0, y1, y2
    old_data = data.copy()
    old_data['C'] = 0       # Assign 0 to old dataset
    per_data = data.copy()
    per_data['C'] = 1       # Assign 1 to permuted dataset

    # Step 1: Permute the dataset
    if permute_pairwise:    # permute v1 and v2 together as a pair
        order = list(per_data.index)
        np.random.shuffle(order)
        per_data['y1'] = per_data['y1'][order].reset_index(drop=True)
        per_data['y2'] = per_data['y2'][order].reset_index(drop=True)
        # print(per_data.head())
    else:
        order1 = list(per_data.index)
        np.random.shuffle(order1)
        per_data['y1'] = per_data['y1'][order1].reset_index(drop=True)
        order2 = list(per_data.index)
        np.random.shuffle(order2)
        per_data['y2'] = per_data['y2'][order2].reset_index(drop=True)

    # Step 2: Concatenate two dataframe and shuffle
    # combine data: file_name, y0, y1, y2, C
    combine_data = pd.concat([old_data, per_data]).reset_index(drop=True)
    combine_order = list(combine_data.index)
    np.random.shuffle(combine_order)
    combine_data = combine_data.take(combine_order).reset_index(drop=True)

    # Step 3: Build a classifier: Given y0, y1(V1), y2(V2), predict C(0 for oberserved dataset, 1 for permuted dataset)
    rng = np.random.RandomState(0)
    if permute_pairwise:
        x = combine_data[["y0", "y1"]].to_numpy()
    else:
        x = combine_data[["y0", "y1", "y2"]].to_numpy()
    y = combine_data[["C"]].to_numpy()
    train_idx = rng.choice(x.shape[0], int(x.shape[0]*0.8), replace=False)
    test_idx = list(set(range(x.shape[0]))-set(train_idx))
    train_x = x[train_idx]
    train_y = y[train_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]

    clf = get_clf(train_x, train_y, test_x, test_y)
    if permute_pairwise:
        weights = get_pw_y1(clf)
    else:
        weights = get_pw_y1y2(clf)
    
    return weights

def avg_pw(data, permute_pairwise):
    weights = {}
    if permute_pairwise:
        weights = {"00":0, "01":0, "10":0, "11":0}
    else:
        weights = {"000": 0, "001":0, "010":0, "011":0, "100":0, "101":0, "110":0, "111":0}
    epoch = 1000
    for i in range(epoch):
        new_data = data.sample(frac=1, replace=True, random_state=1)
        new_weights = pw(new_data, permute_pairwise)
        for key in weights:
            weights[key] += new_weights[key]
    
    # avg the weights
    for key in weights:
        weights[key] /= epoch
    print(weights)
    return weights

def assign_weights(df, permute_pairwise):
    df['weights'] = 0.0
    weights = avg_pw(df, permute_pairwise)
    for i in range(len(df.index)):
        if permute_pairwise:
            w_str = str(int(df.y0[i])) + str(int(df.y1[i]))
        else:
            w_str = str(int(df.y0[i])) + str(int(df.y1[i])) + str(int(df.y2[i]))
        w = weights[w_str]
        df.weights[i] = w
    return df

"""The function to use in the data builder"""
def get_pw_weights(data, permute_pairwise):
    # --- load data
    data = data['0'].str.split(",", expand=True)
    data.columns = ['file_name', 'y0', 'y1', 'y2']

    data['y0'] = data.y0.astype(np.float32)
    data['y1'] = data.y1.astype(np.float32)
    data['y2'] = data.y2.astype(np.float32)

    data = assign_weights(data, permute_pairwise)
    data = data.file_name + \
            ',' + data.y0.astype(str) + \
            ',' + data.y1.astype(str) + \
            ',' + data.y2.astype(str) + \
            ',' + data.weights.astype(str)

    data = data.apply(lambda x: [x])
    return data