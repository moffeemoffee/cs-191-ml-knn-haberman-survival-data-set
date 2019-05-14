import os
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt

# Load dataset
df = pd.read_csv("haberman.csv")
X = df[["age", "op_year", "axil_nodes"]].values
y = df["surv_status"].values

# Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=191,
                                                    stratify=y)

k_values = range(1, 200 + 1)
cv_scores = []

for k in k_values:
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the data
    knn.fit(X_train, y_train)

    # Perform cross validation
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    # Add to cv_scores
    cv_scores.append(scores.mean())

# Find best K
best_cv_score = max(cv_scores)
best_k = k_values[cv_scores.index(best_cv_score)]
print(f"Optimal K: {best_k} with {best_cv_score}")

# Plot graph
plt.plot(k_values, cv_scores)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Cross-validation Score")
plt.savefig("graph.png", dpi=150)
plt.show()
