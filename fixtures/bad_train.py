"""A deliberately flawed training script used as a review fixture.

Known issues:
1. bare except that swallows errors
2. `total` computed but never used
3. zero_grad only inside the loop, missing for the accumulation path
4. learning rate is fixed and high relative to batch size (lr schedule issue)
"""

from torch import nn, optim


def train_epoch(model, loader, lr=0.1):
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    total = 0
    for batch in loader:
        x, y = batch
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    try:
        print("epoch done")
    except:
        pass
    return total
