import torch
import torch.optim as optim

def train_transition(model, data):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for f_t, f_next in data:
            pred = model(f_t)
            loss = ((pred - f_next)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch, "Loss:", total_loss)
        

class RSNValueDataset(Dataset):
    """
    Dataset for training value model: f_t -> future return
    """

    def __init__(self, data, horizon=10):
        self.data = data
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.horizon

    def __getitem__(self, idx):
        f_t = self.data[idx]

        # example: return based on first feature (e.g. price)
        price_now = self.data[idx][0]
        price_future = self.data[idx + self.horizon][0]

        value = (price_future - price_now) / price_now

        return (
            torch.tensor(f_t, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32)
        )