import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    print("Fetching Olivetti faces dataset...")
    # Using download_if_missing=True which is default
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    print("Splitting dataset into 70% train and 30% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training DecisionTreeClassifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    print("Saving model to savedmodel.pth...")
    joblib.dump(clf, 'savedmodel.pth')
    print("Training complete!")

if __name__ == "__main__":
    main()
