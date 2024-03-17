import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler


def cos_face_loss(cos_theta, y, m, s):
    y_oh = F.one_hot(y, 1000)
    cos_theta_m = cos_theta - m * y_oh
    logits = cos_theta_m * s
    return F.cross_entropy(logits, y)


def train_epochs(model, opt, loss_fn, epochs, data_tr, data_val, s, m):
    history = []
    scheduler = lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        train_avg_loss = 0
        model.train()
        for i in data_tr:
            X_batch, Y_batch = i['image'], i['label']
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            opt.zero_grad()
            features, cos_theta = model(X_batch)
            loss = loss_fn(cos_theta, Y_batch, m, s)
            loss.backward()
            opt.step()
            train_avg_loss += loss.item() / len(data_tr)
        print('train_loss: %f' % train_avg_loss)

        model.eval()
        val_avg_loss = 0
        val_avg_acc = 0
        with torch.no_grad():
            for j in data_val:
                X_val, Y_val = j['image'], j['label']
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                features, cos_theta = model(X_val)
                val_los = loss_fn(cos_theta, Y_val, m, s)
                pred = torch.argmax(cos_theta, dim=-1)
                val_acc = (pred == Y_val).sum() / len(Y_val)
                val_avg_loss += val_los.item() / len(data_val)
                val_avg_acc += val_acc.item() / len(data_val)

        print('val_loss: %f' % val_avg_loss)
        print('val_acc: %f' % val_avg_acc)
        history.append([train_avg_loss, val_avg_loss])
        scheduler.step()

    return history
