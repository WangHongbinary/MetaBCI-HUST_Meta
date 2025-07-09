import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),  # 添加padding保持尺寸
            nn.Conv2d(40, 40, (12, 1), padding=(5, 0)),  # 添加padding保持尺寸
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
            nn.ELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
        # 添加层归一化
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        
        # 层归一化
        x_norm = self.norm(x)
        
        queries = rearrange(self.queries(x_norm), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.keys(x_norm), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x_norm), 'b n (h d) -> b h n d', h=self.num_heads)
        
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        scaling = self.head_dim ** 0.5
        attention = F.softmax(energy / scaling, dim=-1)
        attention = self.att_drop(attention)
        
        out = torch.einsum('bhal, bhlv -> bhav', attention, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        out = out + x  # 残差连接
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p),
        )

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size, forward_expansion, forward_drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        # 带残差连接的注意力
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)
        
        # 带残差连接的前馈网络
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # 动态计算输入维度
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # 全局平均池化
        x = self.pool(x.permute(0, 2, 1))  # (B, E, N) -> (B, E, 1)
        x = x.squeeze(-1)  # (B, E)
        return self.fc(x)

class EEGConformer(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, emb_size=40):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size=emb_size)
        self.TransformerEncoder = TransformerEncoder(depth=4, emb_size=emb_size)
        self.classification_head = ClassificationHead(emb_size=emb_size, n_classes=n_classes)

    def forward(self, x):
        # 添加通道维度: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1) if x.dim() == 3 else x
        
        x = self.patch_embedding(x)
        x = self.TransformerEncoder(x)
        x = self.classification_head(x)
        return x

    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.001):
        # 转换数据格式
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)

        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            if X_tensor.dim() == 3:
                X_tensor = X_tensor.unsqueeze(1)
            outputs = self(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
        return predicted.cpu().numpy(), probabilities.cpu().numpy()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32
    seq_len = 2100
    n_channels = 20
    n_classes = 4
    epochs = 1
    model = EEGConformer(n_channels=n_channels, n_samples=seq_len, n_classes=n_classes, emb_size=40).to(device)
    X = torch.zeros(batch_size, 20, seq_len).to(device)
    y_true = torch.randint(0, 4, (batch_size,)).to(device)

    model.fit(X.cpu().numpy(), y_true.cpu().numpy(), epochs=epochs, batch_size=batch_size)
    y_pred, y_pro = model.predict(X.cpu().numpy())

    print(f"input.shape {X.shape}")
    print(f"out.shape {y_pro.shape}")