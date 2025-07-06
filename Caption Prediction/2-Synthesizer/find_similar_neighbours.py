import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def load_data(csv_path, embedding_path):
    """
    Load dataset CSV and corresponding embeddings from pickle file.
    """
    df = pd.read_csv(csv_path)
    with open(embedding_path, 'rb') as f:
        embedding_dict = pickle.load(f)

    df['ImageID'] = df['ImageCLEFmedical 2025 ID'].apply(lambda x: x + '.jpg')
    df['Embedding'] = df['ImageID'].map(embedding_dict)
    return df

def build_knn_index(embeddings, n_neighbors=1):
    """
    Build and fit a KNN index using cosine similarity.
    """
    knn = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors)
    knn.fit(embeddings)
    return knn

def analyze_neighbors(df_dev, df_merged, knn_model, similarity_thresholds, max_neighbors, output_dir):
    """
    For each dev sample, find similar images from merged (train+valid) set.
    Generate CSV results and collect metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_merged = np.array(df_merged['Embedding'].to_list())
    similarity_values = []
    mean_neighbors_list = []

    for threshold in similarity_thresholds:
        neighbor_info = {}
        neighbors_count = []

        for _, row in tqdm(df_dev.iterrows(), total=len(df_dev),
                           desc=f"Similarity ≥ {threshold:.2f}"):
            img_id = row['ImageID']
            embedding = row['Embedding']
            neighbor_info[img_id] = []

            if embedding is not None:
                distances, indices = knn_model.kneighbors(embedding.reshape(1, -1))
                for i, dist in zip(indices[0], distances[0]):
                    sim = 1 - dist
                    if sim >= threshold:
                        neighbor_row = df_merged.iloc[i]
                        neighbor_info[img_id].append({
                            'Neighbor': neighbor_row['ImageID'],
                            'Caption': neighbor_row['Caption'],
                            'Similarity': sim
                        })

            neighbors_count.append(len(neighbor_info[img_id]))
            similarity_values.extend([n['Similarity'] for n in neighbor_info[img_id]])

        mean_neighbors = np.mean(neighbors_count)
        mean_neighbors_list.append(mean_neighbors)
        print(f"Threshold {threshold:.2f}: Mean neighbors = {mean_neighbors:.2f}")

        # Save to CSV
        df_output = pd.DataFrame([
            {
                'Test Image': test_img,
                'Neighbor Image': ', '.join(n['Neighbor'] for n in neighbors),
                'Neighbor Caption': ', '.join(n['Caption'] for n in neighbors),
                'Similarities': ', '.join(f"{n['Similarity']:.2f}" for n in neighbors)
            }
            for test_img, neighbors in neighbor_info.items()
        ])
        df_output.to_csv(os.path.join(output_dir, f'dev_neighbors_{threshold:.2f}_k{max_neighbors}.csv'), index=False)

    return similarity_thresholds, mean_neighbors_list, similarity_values

def plot_results(thresholds, mean_neighbors, similarity_values, output_dir, k=1):
    """
    Generate and save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Mean neighbors bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(thresholds, mean_neighbors, width=0.03, alpha=0.7)
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Mean Number of Neighbors')
    plt.title(f'Mean Neighbors vs. Similarity Threshold (k={k})')
    plt.grid(True)
    plt.xticks(thresholds)
    plt.savefig(os.path.join(output_dir, f'mean_neighbors_plot_k{k}.pdf'))
    plt.close()

    # Similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_values, bins=20, alpha=0.7)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution of Neighbors')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'similarity_distribution_k{k}.pdf'))
    plt.close()

    # Save text output
    with open(os.path.join(output_dir, f'mean_neighbors_k{k}.txt'), 'w') as f:
        for t, m in zip(thresholds, mean_neighbors):
            f.write(f"{t:.2f}: {m:.2f}\n")

def main():
    # Config
    max_neighbors = 1
    similarity_thresholds = np.arange(0.75, 1.00, 0.05)
    output_dir = '/path/to/output/directory'  # Change to your desired output directory

    # Paths
    train_csv = "/path/to/your/train/captions.csv" # Change to your train captions CSV path
    valid_csv = "/path/to/your/valid/captions.csv" # Change to your valid captions CSV path
    dev_csv   = "/path/to/your/dev/captions.csv"   # Change to your dev captions CSV path

    train_emb = "/path/to/your/train/embeddings.pkl" # Change to your train embeddings path
    valid_emb = "/path/to/your/valid/embeddings.pkl" # Change to your valid embeddings path
    dev_emb   = "/path/to/your/dev/embeddings.pkl"   # Change to your dev embeddings path

    # Load data
    df_train = load_data(train_csv, train_emb)
    df_valid = load_data(valid_csv, valid_emb)
    df_dev   = load_data(dev_csv, dev_emb)
    df_merged = pd.concat([df_train, df_valid], ignore_index=True)

    # Build KNN
    embeddings_merged = np.array(df_merged['Embedding'].to_list())
    knn = build_knn_index(embeddings_merged, n_neighbors=max_neighbors)

    # Run analysis
    thresholds, mean_neighbors, similarity_values = analyze_neighbors(
        df_dev, df_merged, knn, similarity_thresholds, max_neighbors, output_dir
    )

    # Plot results
    plot_results(thresholds, mean_neighbors, similarity_values, output_dir, k=max_neighbors)

if __name__ == "__main__":
    main()
