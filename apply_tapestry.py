import pandas as pd
from sklearn.cluster import KMeans

def tapestry_algorithm(data, n_clusters=3):
    """
    Apply Tapestry algorithm to the data.
    
    Parameters:
        data (pd.Series or np.array): The input data for clustering.
        n_clusters (int): The number of clusters to form.
        
    Returns:
        dict: A dictionary containing the cluster centers.
    """
    # Ensure the data is in the correct format
    if isinstance(data, pd.Series):
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.values
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    return {"cluster_centers": cluster_centers.tolist()}

# Load predictions
predictions = pd.read_csv('/predictions/predictions.csv')

# Apply Tapestry algorithm
tapestry_result = tapestry_algorithm(predictions['predictions'])

# Save Tapestry result to a file
pd.DataFrame([tapestry_result]).to_csv('/predictions/tapestry_result.csv', index=False)

print("Tapestry result saved to /predictions/tapestry_result.csv")
