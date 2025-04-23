import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from tqdm import tqdm

# Optimal transport components
import ot
from ot.bregman import sinkhorn

# PyTorch for neural components
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Evaluation metrics
from sklearn.metrics import ndcg_score

# %matplotlib inline


class MovieDataset(Dataset):
    def __init__(self, df, n_users, n_items):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["movie_id"].values)
        self.ratings = torch.FloatTensor(df["implicit"].values)  # Use implicit feedback
        self.n_users = n_users
        self.n_items = n_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class BayesianReCon(nn.Module):
    """ReCon model with Bayesian probabilistic reformulation layer"""

    def __init__(self, n_users, n_items, n_factors=40, alpha=0.5, epsilon=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.alpha = alpha  # Weight for optimal transport term
        self.epsilon = epsilon  # Regularization for Sinkhorn

        # User and item embeddings
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        # Initialize embeddings
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, items):
        # Get embeddings
        u = self.user_emb(users)
        i = self.item_emb(items)

        # Dot product for scores
        scores = (u * i).sum(dim=1)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(scores)
        return probs

    def get_all_scores(self):
        """Get scores for all user-item pairs"""
        all_users = torch.arange(self.n_users).long()
        all_items = torch.arange(self.n_items).long()

        # Get all user and item embeddings
        user_embs = self.user_emb(all_users)  # [n_users, n_factors]
        item_embs = self.item_emb(all_items)  # [n_items, n_factors]

        # Matrix multiplication for all scores
        all_scores = torch.matmul(user_embs, item_embs.T)  # [n_users, n_items]

        # Apply sigmoid
        all_probs = torch.sigmoid(all_scores)
        return all_probs

    def compute_ot_loss(self, probs):
        """Compute optimal transport loss with Bayesian reformulation"""
        # Ensure probs is 2D (users × items)
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)  # Make it (1, n_items) for single user

        # Convert probabilities to numpy for OT computation
        P = probs.detach().cpu().numpy()
        n_users, n_items = P.shape

        # Bayesian reformulation: p(u|i) ∝ p(u,i)
        p_u_given_i = P / (P.sum(axis=0, keepdims=True) + 1e-10)

        # Compute item priors p(i)
        p_i = P.sum(axis=0) / (P.sum() + 1e-10)

        # Smooth the priors
        p_i_smooth = np.power(p_i, 1 - self.alpha)
        p_i_smooth = p_i_smooth / (p_i_smooth.sum() + 1e-10)

        # Uniform distribution over users in this batch
        p_u = np.ones(n_users) / n_users

        # Cost matrix: negative log probabilities
        C = -np.log(P + 1e-10)

        # Compute optimal transport plan
        F = sinkhorn(p_u, p_i_smooth, C, reg=self.epsilon)

        # Convert back to torch tensor
        F = torch.FloatTensor(F).to(probs.device)

        # OT loss: negative log likelihood of transport plan
        ot_loss = -(
            F * torch.log(probs) + (1 - F) * torch.log(1 - probs + 1e-10)
        ).mean()

        return ot_loss

    def train_model(
        self, train_loader, test_loader, epochs=10, lr=0.01, batch_size=128
    ):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for users, items, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()

                # Forward pass
                probs = self(users, items)

                # Compute recommendation loss
                rec_loss = criterion(probs, ratings)

                # Compute OT loss on full matrix (for small datasets)
                if batch_size >= self.n_users * self.n_items:
                    all_probs = self.get_all_scores()
                    ot_loss = self.compute_ot_loss(all_probs)
                else:
                    # For large datasets, compute OT loss on batch
                    batch_probs = probs.view(len(users), -1)
                    ot_loss = self.compute_ot_loss(batch_probs)

                # Combined loss
                loss = rec_loss + self.alpha * ot_loss

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average epoch loss
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for users, items, ratings in test_loader:
                    probs = self(users, items)
                    loss = criterion(probs, ratings)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            print(
                f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
            )

        return train_losses, test_losses


def load_data(ratings_path, items_path, rating_threshold=3, ratio_neg_per_user=1):
    """Load and preprocess data"""
    df_ratings = pd.read_csv(ratings_path)
    df_items = pd.read_csv(items_path)

    # Create implicit feedback (1 if rating >= threshold, else 0)
    df_ratings["implicit"] = (df_ratings["rating"] >= rating_threshold).astype(int)

    # Negative feedback sampling
    pos_df = df_ratings[df_ratings["implicit"] == 1]
    neg_df = df_ratings[df_ratings["implicit"] == 0].sample(frac=ratio_neg_per_user)
    df_implicit = pd.concat([pos_df, neg_df])

    # Encode user and item IDs
    user_ids = df_implicit["user_id"].unique()
    item_ids = df_implicit["movie_id"].unique()

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    df_implicit["user_id"] = df_implicit["user_id"].map(user_to_idx)
    df_implicit["movie_id"] = df_implicit["movie_id"].map(item_to_idx)

    n_users = len(user_ids)
    n_items = len(item_ids)

    return df_implicit, df_items, n_users, n_items, user_to_idx, item_to_idx


def evaluate_model(model, test_loader, k=10):
    """Evaluate model using NDCG, Precision@K, Recall@K"""
    model.eval()
    all_users = []
    all_items = []
    all_ratings = []
    all_preds = []

    with torch.no_grad():
        for users, items, ratings in test_loader:
            probs = model(users, items)

            all_users.extend(users.tolist())
            all_items.extend(items.tolist())
            all_ratings.extend(ratings.tolist())
            all_preds.extend(probs.tolist())

    # Create DataFrame for evaluation
    eval_df = pd.DataFrame(
        {
            "user_id": all_users,
            "movie_id": all_items,
            "rating": all_ratings,
            "pred": all_preds,
        }
    )

    # Calculate metrics
    ndcg = []
    precisions = []
    recalls = []

    for user_id in eval_df["user_id"].unique():
        user_df = eval_df[eval_df["user_id"] == user_id]

        # Skip users with fewer than 2 ratings (NDCG needs at least 2 items)
        if len(user_df) < 2:
            continue

        # Get top-k predictions
        top_k = user_df.nlargest(k, "pred")

        # NDCG
        true_relevance = user_df["rating"].values
        pred_relevance = user_df["pred"].values
        ndcg.append(ndcg_score([true_relevance], [pred_relevance], k=k))

        # Precision@K
        relevant = top_k["rating"].sum()
        precision = relevant / k
        precisions.append(precision)

        # Recall@K
        total_relevant = user_df["rating"].sum()
        recall = relevant / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)

    # Handle case where all users had <2 ratings
    if not ndcg:
        print("Warning: No users with sufficient ratings for NDCG calculation")
        avg_ndcg = 0
        avg_precision = 0
        avg_recall = 0
    else:
        avg_ndcg = np.mean(ndcg)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

    return avg_ndcg, avg_precision, avg_recall


def plot_metrics(train_losses, test_losses):
    """Plot training and test losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid()
    plt.show()


def recommend_top_k(model, user_id, item_ids, user_rated_items, k=10):
    """Generate top-k recommendations for a user, excluding already rated items"""
    model.eval()
    with torch.no_grad():
        # Get items not rated by user
        unrated_items = list(set(item_ids) - set(user_rated_items))
        if not unrated_items:
            return []

        user_tensor = torch.LongTensor([user_id])
        items_tensor = torch.LongTensor(unrated_items)

        # Get predictions for unrated items
        scores = model(user_tensor.repeat(len(unrated_items)), items_tensor)

        # Get top-k items
        top_k_idx = torch.topk(scores, min(k, len(unrated_items))).indices
        top_k_items = items_tensor[top_k_idx].tolist()
        top_k_scores = scores[top_k_idx].tolist()

    return list(zip(top_k_items, top_k_scores))


def main():
    # Load and preprocess data
    ratings_path = "data/ratings.csv"
    items_path = "data/items.csv"

    df_implicit, df_items, n_users, n_items, user_to_idx, item_to_idx = load_data(
        ratings_path, items_path, rating_threshold=3, ratio_neg_per_user=1
    )

    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")
    print(f"Number of interactions: {len(df_implicit)}")

    # Split data into train and test
    train_df = df_implicit.sample(frac=0.8, random_state=42)
    test_df = df_implicit.drop(train_df.index)

    # Create datasets and dataloaders
    train_dataset = MovieDataset(train_df, n_users, n_items)
    test_dataset = MovieDataset(test_df, n_users, n_items)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = BayesianReCon(n_users, n_items, n_factors=40, alpha=0.5, epsilon=0.1)

    # Train model
    train_losses, test_losses = model.train_model(
        train_loader, test_loader, epochs=10, lr=0.01, batch_size=batch_size
    )

    # Plot training curves
    plot_metrics(train_losses, test_losses)

    # Evaluate model
    ndcg, precision, recall = evaluate_model(model, test_loader, k=10)
    print(f"NDCG@10: {ndcg:.4f}")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")

    # Example recommendation
    user_id = 0  # First user in our dataset
    all_item_ids = list(range(n_items))

    # Example recommendation
    user_id = 0  # First user in our dataset
    all_item_ids = list(range(n_items))

    # Get items already rated by user 0
    user_rated_items = df_implicit[df_implicit["user_id"] == user_id][
        "movie_id"
    ].tolist()

    recommendations = recommend_top_k(
        model, user_id, all_item_ids, user_rated_items, k=5
    )

    # Get user 0's actual rated movies
    user_0_ratings = (
        df_implicit[df_implicit["user_id"] == user_id]
        .merge(df_items, on="movie_id")[["title", "rating", "implicit"]]
        .sort_values("rating", ascending=False)
    )

    print("\nUser 0 Details:")
    print(f"Total movies rated: {len(user_0_ratings)}")
    print(f"Positive ratings (>=3 stars): {user_0_ratings['implicit'].sum()}")

    print("\nUser 0's Top Rated Movies:")
    print(user_0_ratings.head(10).to_string(index=False))

    print("\nTop 5 Novel Recommendations for User 0:")
    for item_id, score in recommendations:
        movie_title = df_items[df_items["movie_id"] == item_id]["title"].values[0]
        print(f"{movie_title}: {score:.4f}")

    score = score_model(model, test_df, verbose=1)


if __name__ == "__main__":
    main()


def score_model(model, test_data, verbose=1):
    """
    Evaluate BayesianReCon model performance using multiple metrics

    Parameters:
    -----------
    model : BayesianReCon
        The trained model to evaluate
    test_data : DataFrame
        Test data containing user_id, movie_id, and implicit columns
    verbose : int
        Verbosity level (0: silent, 1: show metrics)

    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    model.eval()

    # Extract test components
    users = torch.LongTensor(test_data["user_id"].values)
    items = torch.LongTensor(test_data["movie_id"].values)
    ratings = torch.FloatTensor(test_data["implicit"].values)

    # Get predictions
    with torch.no_grad():
        predictions = model(users, items)

    # Initialize metrics
    metrics = {}

    # Calculate loss (BCE)
    criterion = nn.BCELoss()
    loss = criterion(predictions, ratings).item()
    metrics["loss"] = loss

    # Calculate additional metrics
    # Convert to numpy for easier calculation
    y_true = ratings.numpy()
    y_pred = predictions.numpy()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    metrics["rmse"] = rmse

    # Calculate MAE
    mae = np.mean(np.abs(y_pred - y_true))
    metrics["mae"] = mae

    # Calculate accuracy (threshold 0.5)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_true)
    metrics["accuracy"] = accuracy

    # Calculate AUC
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(y_true, y_pred)
        metrics["auc"] = auc
    except:
        metrics["auc"] = float("nan")  # In case of single class

    # Calculate F1 Score
    from sklearn.metrics import f1_score

    f1 = f1_score(y_true, y_pred_binary, average="binary")
    metrics["f1"] = f1

    # Calculate Top-K metrics
    k_values = [5, 10, 20]

    # Create user-item matrix for top-k evaluation
    user_item_df = pd.DataFrame(
        {
            "user_id": users.numpy(),
            "movie_id": items.numpy(),
            "rating": ratings.numpy(),
            "prediction": predictions.numpy(),
        }
    )

    # Calculate NDCG@k, Precision@k and Recall@k
    for k in k_values:
        ndcg_scores = []
        precision_scores = []
        recall_scores = []

        for user_id in user_item_df["user_id"].unique():
            user_data = user_item_df[user_item_df["user_id"] == user_id]

            # Skip users with fewer than 2 ratings
            if len(user_data) < 2:
                continue

            # Get top-k predictions
            top_k_items = user_data.nlargest(k, "prediction")

            # NDCG@k
            from sklearn.metrics import ndcg_score

            try:
                true_relevance = user_data["rating"].values.reshape(1, -1)
                pred_relevance = user_data["prediction"].values.reshape(1, -1)
                ndcg_val = ndcg_score(
                    true_relevance, pred_relevance, k=min(k, len(user_data))
                )
                ndcg_scores.append(ndcg_val)
            except:
                pass

            # Precision@k
            relevant_retrieved = top_k_items["rating"].sum()
            precision = relevant_retrieved / k
            precision_scores.append(precision)

            # Recall@k
            total_relevant = user_data["rating"].sum()
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            recall_scores.append(recall)

        if ndcg_scores:
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
        if precision_scores:
            metrics[f"precision@{k}"] = np.mean(precision_scores)
        if recall_scores:
            metrics[f"recall@{k}"] = np.mean(recall_scores)

    # Print results if verbose
    if verbose:
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    return metrics


# Usage in the main function would be:
