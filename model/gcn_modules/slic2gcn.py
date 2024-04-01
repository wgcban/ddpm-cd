import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from skimage.segmentation import slic
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from sklearn.manifold import TSNE


def SuperpixelEncoder(x):
    image = x.squeeze(0).cpu().numpy()
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    features = []
    positions = []
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        features.append(image[mask].mean(axis=0))
        position = np.mean(np.argwhere(segments == segment_label), axis=0)
        positions.append(position)
    features = torch.tensor(features, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)

    # Combine features and positions
    features = torch.cat((features, positions), dim=1)

    # 重新映射超像素标签到从0开始的索引
    unique_labels = np.unique(segments)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    remapped_segments = np.vectorize(label_to_index.get)(segments)

    # 创建边
    edges = []
    for i in range(remapped_segments.shape[0]):
        for j in range(remapped_segments.shape[1]):
            if i > 0 and remapped_segments[i, j] != remapped_segments[i - 1, j]:
                edges.append([remapped_segments[i, j], remapped_segments[i - 1, j]])
            if j > 0 and remapped_segments[i, j] != remapped_segments[i, j - 1]:
                edges.append([remapped_segments[i, j], remapped_segments[i, j - 1]])
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return features, edges


class GraphConvModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphConvModule, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = x.relu()
        # 第二层GCN
        x = self.conv2(x, edge_index)
        return x


# 假设我们有一个输入图像张量和一个超像素数量
input_tensor = torch.randn(1, 3, 256, 256)  # (batch_size, channels, height, width)

features, edges = SuperpixelEncoder(input_tensor)

data = Data(x=features, edge_index=edges)
model = GraphConvModule(input_dim=features.shape[1], hidden_dim=16, output_dim=32)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Placeholder for labels (you would need actual labels for training)
labels = torch.randint(0, 2, (features.shape[0],), dtype=torch.long)

# Train the model
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


