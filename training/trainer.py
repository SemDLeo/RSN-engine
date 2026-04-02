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