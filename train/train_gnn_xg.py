# File: train/train_gnn_xg.py
"""
Graph Neural Network approach for xG modeling
Models the spatial relationships between players and the puck
Could potentially achieve higher AUC than traditional methods
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import logging
# from datetime import datetime  # Unused import
# import os  # Unused import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShotGraphNet(nn.Module):
    """Graph Neural Network for shot prediction"""

    def __init__(self, num_features, hidden_dim=64):
        super(ShotGraphNet, self).__init__()

        # Graph convolution layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)

        # Final prediction layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv3(x, edge_index))

        # Global pooling (aggregate node features)
        x = global_mean_pool(x, batch)

        # Final prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)


class GNNDataProcessor:
    """Convert shot data into graph format"""

    def __init__(self):
        self.shots_data = None

    def load_data(self, filepath: str = "data/shots_2024.csv"):
        """Load shot data"""
        logger.info(f"Loading data from {filepath}...")
        self.shots_data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.shots_data)} shots")

    def create_shot_graph(self, shot_data):
        """
        Create a graph representation of a shot
        Nodes: shot location, net location, last event location
        Edges: spatial relationships
        """
        nodes = []
        edges = []

        # Node 0: Shot location
        shot_node = [
            shot_data["arenaAdjustedXCordABS"],
            shot_data["arenaAdjustedYCordAbs"],
            shot_data["shotDistance"],
            shot_data["shotAngleAdjusted"],
            shot_data["speedFromLastEvent"],
            shot_data["timeSinceLastEvent"],
            1.0,  # Is shot node
        ]
        nodes.append(shot_node)

        # Node 1: Net location (always at 89, 0)
        net_node = [89.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        nodes.append(net_node)

        # Node 2: Last event location (if available)
        if pd.notna(shot_data.get("lastEventxCord_adjusted", np.nan)):
            last_event_node = [
                shot_data["lastEventxCord_adjusted"],
                shot_data["lastEventyCord_adjusted"],
                shot_data["distanceFromLastEvent"],
                0.0,
                shot_data["speedFromLastEvent"],
                shot_data["timeSinceLastEvent"],
                0.0,
            ]
            nodes.append(last_event_node)

            # Add edges: shot->net, shot->last_event, last_event->shot
            edges = [[0, 1], [0, 2], [2, 0]]
        else:
            # Just shot->net edge
            edges = [[0, 1]]

        # Add game state features to shot node
        additional_features = [
            shot_data["homeSkatersOnIce"],
            shot_data["awaySkatersOnIce"],
            shot_data["period"],
            shot_data["shotRebound"],
            shot_data["shotRush"],
            shot_data["homeTeamGoals"] - shot_data["awayTeamGoals"],
        ]

        # Extend first node with game state
        nodes[0].extend(additional_features)

        # Pad other nodes to match dimensions
        max_features = len(nodes[0])
        for i in range(1, len(nodes)):
            nodes[i].extend([0.0] * (max_features - len(nodes[i])))

        return np.array(nodes), np.array(edges).T if edges else np.array([[0], [0]])

    def prepare_graph_data(self, sample_size=None):
        """Convert all shots to graph format"""
        logger.info("Converting shots to graph format...")

        if self.shots_data is None:
            raise ValueError("No shots data loaded!")
        if sample_size:
            data = self.shots_data.sample(n=sample_size, random_state=42)
        else:
            data = self.shots_data

        graph_data = []

        for idx, shot in data.iterrows():
            nodes, edges = self.create_shot_graph(shot)

            # Create PyTorch geometric data object
            graph = Data(
                x=torch.FloatTensor(nodes), edge_index=torch.LongTensor(edges), y=torch.FloatTensor([shot["goal"]])
            )

            graph_data.append(graph)

            if isinstance(idx, int) and idx % 10000 == 0:
                logger.info(f"  Processed {idx:,} shots...")

        return graph_data


class GNNTrainer:
    """Train the Graph Neural Network"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None

    def train_model(self, train_loader, val_loader, num_features, epochs=50):
        """Train the GNN model"""
        logger.info(f"Training on {self.device}")

        # Initialize model
        self.model = ShotGraphNet(num_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        best_val_auc = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)

                    val_preds.extend(out.squeeze().cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())

            val_auc = roc_auc_score(val_labels, val_preds)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), "models/gnn_best.pt")

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}")

        return best_val_auc


def main():
    """Run GNN training"""
    logger.info("🏒 GRAPH NEURAL NETWORK xG MODEL")
    logger.info("=" * 60)

    # Load and process data
    processor = GNNDataProcessor()
    processor.load_data()

    # For demonstration, use a sample (full dataset takes longer)
    logger.info("\nUsing 50,000 shots for faster training...")
    graph_data = processor.prepare_graph_data(sample_size=50000)

    # Split data
    train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Get number of features
    num_features = graph_data[0].x.shape[1]
    logger.info(f"Number of node features: {num_features}")

    # Train model
    trainer = GNNTrainer()
    best_auc = trainer.train_model(train_loader, val_loader, num_features, epochs=30)

    logger.info(f"\n✅ Training complete! Best validation AUC: {best_auc:.4f}")

    if best_auc > 0.78:
        logger.info("🎉 GNN outperformed MoneyPuck!")
    else:
        logger.info("GNN performed well but didn't beat MoneyPuck - try more epochs or features")


if __name__ == "__main__":
    main()
