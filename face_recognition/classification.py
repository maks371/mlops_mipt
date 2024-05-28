from collections import defaultdict

import torch


def compute_avg_embeddings(model, train_data, device):
    images_for_class = defaultdict(list)
    for sample in train_data:
        images_for_class[sample["label"]].append(sample["image"][None, :, :, :])

    avg_embeddings_list = []
    for i in range(1000):
        pictures = torch.cat(images_for_class[i], dim=0)
        embeddings, _ = model(pictures.to(device))
        embeddings = embeddings.detach()
        avg_embeddings_list.append(embeddings.mean(dim=0)[None, :])

    vector_avg_embeddings = torch.cat(avg_embeddings_list, dim=0)
    return vector_avg_embeddings


def predict(image_embeddings, avg_embeddings):
    numerator = (image_embeddings * avg_embeddings).sum(dim=-1)
    denominator = torch.sqrt(
        torch.pow(image_embeddings, 2).sum(dim=-1)
        * torch.pow(avg_embeddings, 2).sum(dim=-1)
    )
    batch_cosine_similarity = numerator / denominator
    pred = torch.argmax(batch_cosine_similarity, dim=-1)
    return pred


def get_predictions(model, test_loader, device, avg_embeddings):
    predictions = []
    for i in test_loader:
        X_batch, _ = i["image"], i["label"]
        emb, _ = model(X_batch.to(device))
        pred = predict(emb.detach()[:, None, :], avg_embeddings[None, :, :])
        predictions += pred.tolist()
    return predictions
