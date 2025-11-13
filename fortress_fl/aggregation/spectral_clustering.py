"""
Spectral Clustering for Byzantine Detection in FORTRESS-FL

Partition operators into honest and Byzantine clusters based on gradient similarity.
Byzantine gradients form a cohesive subspace while honest gradients cluster around
the true gradient direction.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple


def compute_similarity_matrix(gradients: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for gradients.

    Args:
        gradients: List of n gradient vectors [g_1, g_2, ..., g_n]
                   Each g_i is numpy array of shape (d,)

    Returns:
        S: Similarity matrix of shape (n, n)
    """
    n = len(gradients)
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                S[i, j] = 1.0  # Self-similarity
            else:
                # Cosine similarity: S_ij = <g_i, g_j> / (||g_i|| * ||g_j||)
                dot_product = np.dot(gradients[i], gradients[j])
                norm_i = np.linalg.norm(gradients[i])
                norm_j = np.linalg.norm(gradients[j])

                if norm_i == 0 or norm_j == 0:
                    S[i, j] = 0.0  # Handle zero gradients
                else:
                    S[i, j] = dot_product / (norm_i * norm_j)

    return S


def spectral_clustering_byzantine_detection(gradients: List[np.ndarray],
                                          k: int = 2) -> Tuple[np.ndarray, int]:
    """
    Perform spectral clustering to identify Byzantine operators.

    Args:
        gradients: List of n gradient vectors
        k: Number of clusters (default: 2 for honest vs. Byzantine)

    Returns:
        (cluster_labels, byzantine_cluster_id):
            cluster_labels: Array of length n with cluster assignments [0, 1, ..., k-1]
            byzantine_cluster_id: ID of cluster identified as Byzantine
    """
    n = len(gradients)

    if n < k:
        raise ValueError(f"Cannot cluster {n} gradients into {k} clusters")

    # Step 1: Compute similarity matrix S
    S = compute_similarity_matrix(gradients)

    # Step 2: Convert to affinity matrix (ensure non-negative)
    # Shift similarities to [0, 2] range: A_ij = (S_ij + 1) / 2
    A = (S + 1.0) / 2.0

    # Step 3: Compute degree matrix D
    degrees = A.sum(axis=1)
    D = np.diag(degrees)

    # Step 4: Compute normalized graph Laplacian
    # L = D^(-1/2) * (D - A) * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))  # Add epsilon for stability
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Step 5: Compute k smallest eigenvectors of L_norm
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Sort by eigenvalue (ascending)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take k smallest eigenvectors as features
    U = eigenvectors[:, :k]  # Shape: (n, k)

    # Step 6: Apply k-means clustering on U
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(U)

    # Step 7: Identify Byzantine cluster
    byzantine_cluster_id = identify_byzantine_cluster(S, cluster_labels, k)

    return cluster_labels, byzantine_cluster_id


def identify_byzantine_cluster(S: np.ndarray, cluster_labels: np.ndarray, k: int) -> int:
    """
    Identify which cluster is Byzantine based on internal cohesion.

    Byzantine operators coordinate their attacks, resulting in high internal similarity
    but low similarity to honest operators.

    Args:
        S: Similarity matrix (n, n)
        cluster_labels: Cluster assignments (n,)
        k: Number of clusters

    Returns:
        byzantine_cluster_id: ID of Byzantine cluster
    """
    cluster_stats = []

    for cluster_id in range(k):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        if cluster_size < 2:
            # Too few members to compute meaningful statistics
            cluster_stats.append({
                'id': cluster_id,
                'size': cluster_size,
                'internal_cohesion': -1.0,
                'external_separation': -1.0
            })
            continue

        # Compute internal cohesion (average similarity within cluster)
        internal_similarities = []
        for i in cluster_indices:
            for j in cluster_indices:
                if i < j:
                    internal_similarities.append(S[i, j])

        internal_cohesion = np.mean(internal_similarities) if internal_similarities else 0.0

        # Compute external separation (average similarity to other clusters)
        external_similarities = []
        for i in cluster_indices:
            for j in range(len(cluster_labels)):
                if cluster_labels[j] != cluster_id:
                    external_similarities.append(S[i, j])

        external_separation = np.mean(external_similarities) if external_similarities else 0.0

        cluster_stats.append({
            'id': cluster_id,
            'size': cluster_size,
            'internal_cohesion': internal_cohesion,
            'external_separation': external_separation
        })

    # Byzantine cluster heuristics:
    # 1. Smaller cluster size (Byzantine operators are minority)
    # 2. High internal cohesion (coordinated attack)
    # 3. Low external separation (different from honest operators)

    # Compute suspicion score for each cluster
    max_size = max(stat['size'] for stat in cluster_stats)
    suspicion_scores = []

    for stat in cluster_stats:
        if stat['size'] == 0:
            suspicion_scores.append(0.0)
            continue

        # Normalize metrics
        size_penalty = 1.0 - (stat['size'] / max_size)  # Smaller clusters get higher penalty
        cohesion_bonus = max(0.0, stat['internal_cohesion'])  # Higher cohesion increases suspicion
        separation_penalty = max(0.0, -stat['external_separation'])  # More negative separation increases suspicion

        # Combined suspicion score
        suspicion = size_penalty * 0.4 + cohesion_bonus * 0.3 + separation_penalty * 0.3
        suspicion_scores.append(suspicion)

    # Cluster with highest suspicion score is Byzantine
    byzantine_cluster_id = np.argmax(suspicion_scores)

    return byzantine_cluster_id


def filter_byzantine_gradients(gradients: List[np.ndarray], cluster_labels: np.ndarray,
                              byzantine_cluster_id: int) -> Tuple[List[np.ndarray], List[int]]:
    """
    Remove gradients from Byzantine cluster.

    Args:
        gradients: List of n gradient vectors
        cluster_labels: Cluster assignments (n,)
        byzantine_cluster_id: ID of Byzantine cluster

    Returns:
        (honest_gradients, honest_indices):
            honest_gradients: List of gradients from honest cluster(s)
            honest_indices: Original indices of honest operators
    """
    honest_mask = (cluster_labels != byzantine_cluster_id)
    honest_indices = np.where(honest_mask)[0].tolist()
    honest_gradients = [gradients[i] for i in honest_indices]

    return honest_gradients, honest_indices


def analyze_clustering_quality(gradients: List[np.ndarray], cluster_labels: np.ndarray) -> dict:
    """
    Analyze the quality of clustering results.

    Args:
        gradients: List of gradient vectors
        cluster_labels: Cluster assignments

    Returns:
        analysis: Dict with clustering quality metrics
    """
    n = len(gradients)
    k = len(np.unique(cluster_labels))

    if n < 2 or k < 2:
        return {
            'silhouette_score': -1.0,
            'cluster_sizes': np.bincount(cluster_labels).tolist(),
            'inertia': float('inf')
        }

    # Compute similarity matrix for silhouette score
    S = compute_similarity_matrix(gradients)

    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1.0 - S
    np.fill_diagonal(distance_matrix, 0.0)

    # Compute silhouette score
    try:
        sil_score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
    except:
        sil_score = -1.0

    # Cluster sizes
    cluster_sizes = np.bincount(cluster_labels).tolist()

    # Compute inertia (within-cluster sum of squares)
    inertia = 0.0
    for cluster_id in range(k):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_gradients = [gradients[i] for i in range(n) if cluster_mask[i]]

        if len(cluster_gradients) > 1:
            # Compute cluster centroid
            centroid = np.mean(cluster_gradients, axis=0)

            # Sum of squared distances to centroid
            for gradient in cluster_gradients:
                inertia += np.sum((gradient - centroid) ** 2)

    analysis = {
        'silhouette_score': sil_score,
        'cluster_sizes': cluster_sizes,
        'inertia': inertia,
        'n_clusters': k
    }

    return analysis


def visualize_similarity_matrix(gradients: List[np.ndarray], cluster_labels: np.ndarray = None,
                               save_path: str = None) -> None:
    """
    Visualize the similarity matrix with optional cluster annotations.

    Args:
        gradients: List of gradient vectors
        cluster_labels: Optional cluster assignments for color coding
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    S = compute_similarity_matrix(gradients)
    n = len(gradients)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(S, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')

    if cluster_labels is not None:
        # Add cluster boundaries
        unique_labels = np.unique(cluster_labels)
        boundaries = []
        current_pos = 0

        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            boundaries.append(current_pos + count - 0.5)
            current_pos += count

        for boundary in boundaries[:-1]:
            plt.axhline(y=boundary, color='black', linewidth=2)
            plt.axvline(x=boundary, color='black', linewidth=2)

    plt.title(f'Gradient Similarity Matrix (n={n})')
    plt.xlabel('Operator Index')
    plt.ylabel('Operator Index')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Spectral Clustering for Byzantine Detection...")

    # Create test gradients
    n_honest = 4
    n_byzantine = 2
    gradient_dim = 10

    # Honest gradients (similar direction)
    honest_base = np.random.randn(gradient_dim)
    honest_gradients = [honest_base + 0.1 * np.random.randn(gradient_dim) for _ in range(n_honest)]

    # Byzantine gradients (coordinated attack - sign flip)
    byzantine_gradients = [-grad for grad in honest_gradients[:n_byzantine]]

    # Combine all gradients
    all_gradients = honest_gradients + byzantine_gradients
    n_total = len(all_gradients)

    print(f"Created {n_honest} honest and {n_byzantine} Byzantine gradients")

    # Perform spectral clustering
    cluster_labels, byzantine_cluster_id = spectral_clustering_byzantine_detection(all_gradients, k=2)

    print(f"Cluster labels: {cluster_labels}")
    print(f"Byzantine cluster ID: {byzantine_cluster_id}")

    # Filter Byzantine gradients
    honest_gradients_filtered, honest_indices = filter_byzantine_gradients(
        all_gradients, cluster_labels, byzantine_cluster_id
    )

    print(f"Honest operators: {honest_indices}")
    print(f"Byzantine operators: {[i for i in range(n_total) if i not in honest_indices]}")

    # Analyze clustering quality
    quality = analyze_clustering_quality(all_gradients, cluster_labels)
    print(f"Clustering quality: {quality}")

    # Check if detection was successful
    expected_byzantine = list(range(n_honest, n_total))
    detected_byzantine = [i for i in range(n_total) if i not in honest_indices]

    accuracy = len(set(expected_byzantine) & set(detected_byzantine)) / len(expected_byzantine)
    print(f"Byzantine detection accuracy: {accuracy:.2%}")

    print("Spectral clustering test completed!")