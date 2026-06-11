import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("Loading the dataset to get the test set...")
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target
    
    # Splitting with the exact same random_state to ensure we evaluate on the correct test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Loading the trained model from savedmodel.pth...")
    clf = joblib.load('savedmodel.pth')

    print("Predicting and calculating accuracy...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
