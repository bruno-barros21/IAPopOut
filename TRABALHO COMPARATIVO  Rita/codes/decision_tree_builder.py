import pandas as pd
import numpy as np
import pickle
from collections import Counter

class Node:
    def __init__(self, feature=None, children=None, label=None):
        self.feature = feature
        self.children = children if children is not None else {}
        self.label = label

def entropy(y):
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def information_gain(X, y, feature):
    values = X[feature].unique()
    weighted_entropy = 0

    for v in values:
        subset_y = y[X[feature] == v]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)

    return entropy(y) - weighted_entropy

def id3(X, y, features, depth=0, max_depth=15):
    if len(set(y)) == 1 or len(features) == 0 or depth == max_depth:
        return Node(label=y.iloc[0])
    
    if len(features) == 0:
        most_common_label = y.mode()[0]
        return Node(label=most_common_label)

    gains = [information_gain(X, y, f) for f in features]
    best_feature = features[np.argmax(gains)]

    node = Node(feature=best_feature)
    feature_values = X[best_feature].unique()

    for value in feature_values:
        subset_X = X[X[best_feature] == value]
        subset_y = y[X[best_feature] == value]

        if len(subset_y) == 0:
            most_common_label = y.mode()[0]
            child = Node(label=most_common_label)
        else:
            remaining_features = [f for f in features if f != best_feature]
            child = id3(subset_X, subset_y, remaining_features, depth+1, max_depth)

        node.children[value] = child

    return node
def predict(tree, sample):
    while tree.label is None:
        value = sample.get(tree.feature)
        if value in tree.children:
            tree = tree.children[value]
        else:
            # Valor nunca visto — retorna o valor de maior ocorrência entre os filhos
            # ou uma jogada aleatória segura
            if tree.children:
                return majority_vote(tree)
            else:
                return None
    return tree.label

def majority_vote(node):
    # Conta as labels mais comuns nos filhos
    labels = []
    def collect_labels(subnode):
        if subnode.label is not None:
            labels.append(subnode.label)
        else:
            for child in subnode.children.values():
                collect_labels(child)
    collect_labels(node)
    if labels:
        return Counter(labels).most_common(1)[0][0]
    else:
        return None



# --- Load data ---
data = pd.read_csv("mcts_dataset.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
features = X.columns.tolist()

# --- Train the tree ---
tree = id3(X, y, features)

# --- Test prediction ---
sample = X.iloc[0].to_dict()  # get first row as a dict
predicted_label = predict(tree, sample)
actual_label = y.iloc[0]

#print("Predicted:", predicted_label)
#print("Actual:   ", actual_label)
correct = 0
for i in range(len(X)):
    sample = X.iloc[i].to_dict()
    prediction = predict(tree, sample)
    if prediction == y.iloc[i]:
        correct += 1

accuracy = correct / len(X)
#print(f"Accuracy on training data: {accuracy:.2%}")

# --- Save the tree ---
with open("decision_tree.pkl", "wb") as f:
    pickle.dump(tree, f)
