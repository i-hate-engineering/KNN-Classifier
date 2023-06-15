import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to calculate accuracy for a specific k value
def calculate_accuracy(X_train, X_test, y_train, y_test, k):
    # Create a KNN classifier with the specified k value
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Function to plot the graph for different values of k
def plot_graph(k_values, accuracies):
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k Value')
    plt.grid(True)
    plt.show()

# Get the file path from the user
file_path = input("Enter the path to the Iris dataset file: ")

# Load the dataset from the file
df = pd.read_csv(file_path)

# Split the dataset into features (X) and labels (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menu-driven interface
while True:
    print("\nMENU")
    print("1. Find accuracy for a specific k value")
    print("2. Plot graph for different values of k")
    print("3. Exit")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        k = int(input("Enter the value of k (number of neighbors): "))
        accuracy = calculate_accuracy(X_train, X_test, y_train, y_test, k)
        print("Accuracy for k =", k, ": {:.4f}".format(accuracy))
    elif choice == '2':
        k_values = []
        accuracies = []
        max_k = min(20, len(X_train))  # Limit the maximum k value to the number of samples in the training set
        
        for k in range(1, max_k + 1):
            accuracy = calculate_accuracy(X_train, X_test, y_train, y_test, k)
            k_values.append(k)
            accuracies.append(accuracy)
        
        plot_graph(k_values, accuracies)
    elif choice == '3':
        print("Exiting the program...")
        break
    else:
        print("Invalid choice! Please enter a valid option.")
