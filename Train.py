import json
import torch.nn.functional as F
import torch.optim as optim
import utils
from DataLoad import get_loader

config = json.load(open('./config.json', 'r'))

def Train(model, epoch, batchsize, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = 0
    for file in config['train_set']:
        dataset = get_loader(file, batchsize)
        for idx, parameters in enumerate(dataset):
            parameters = utils.to_var(parameters)
            out = model(parameters)
            loss = F.nll_loss(out, parameters['Class'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            if idx > 0 and idx % 10 == 0:
                print("\r"+'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, idx * batchsize, len(dataset.dataset),
                        100. * idx / len(dataset), train_loss.item() / 10), end="", flush=True)
                train_loss = 0
    return model