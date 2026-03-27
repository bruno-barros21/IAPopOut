import pandas as pd
import pickle
from decision_tree_builder import id3, predict, Node

# --- Load and discretize the Iris dataset ---

df = pd.read_csv('iris.csv')

def discretize(col, n_bins=3):
    return pd.cut(col, bins=n_bins, labels=[f'bin{i}' for i in range(n_bins)])

for col in df.columns[:-1]:
    df[col] = discretize(df[col])

# --- Prepare features and labels ---
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
features = X.columns.tolist()

# --- Train-test split (70% train / 30% test) ---
df_shuffled = df.sample(frac=1).reset_index(drop=True)
split_idx = int(len(df_shuffled) * 0.7)
train = df_shuffled.iloc[:split_idx].reset_index(drop=True)
test = df_shuffled.iloc[split_idx:].reset_index(drop=True)

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

# --- Train the decision tree ---
tree = id3(X_train, y_train, features)

# --- Serialize the decision tree to disk with pickle ---
with open('iris_tree.pkl', 'wb') as f:
    pickle.dump(tree, f)
print("Decision tree serialized to iris_tree.pkl")

# --- Function to print the tree structure ---
def print_tree(node: Node, depth=0):
    indent = '  ' * depth
    if node.label is not None:
        print(f"{indent}→ Label: {node.label}")
    else:
        feat_name = features[node.feature] if isinstance(node.feature, int) else node.feature
        print(f"{indent}Feature: {feat_name}")
        for value, child in node.children.items():
            print(f"{indent}  If == {value}:")
            print_tree(child, depth+2)

# --- Display the decision tree structure ---
print("\nDecision Tree Structure:")
print_tree(tree)

# --- Evaluate accuracy on test set ---
correct = 0
for idx, row in X_test.iterrows():
    sample = row.to_dict()
    pred = predict(tree, sample)
    true_label = y_test.loc[idx]
    if pred == true_label:
        correct += 1
accuracy = correct / len(X_test)
print(f"\nAccuracy: {accuracy:.2%}")
