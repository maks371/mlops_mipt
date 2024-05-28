import fire
import torch
from torchvision import transforms

from face_recognition.classification import compute_avg_embeddings
from face_recognition.custom_dataset import CelebADataset
from face_recognition.model import CosModel
from face_recognition.train_model import cos_face_loss, train_epochs


def train_and_save(data_folder, save_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CosModel().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_data = CelebADataset(data_folder, "train", transform)
    val_data = CelebADataset(data_folder, "val", transform)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    loss = cos_face_loss
    epochs_number = 10
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    history = train_epochs(
        model, opt, loss, epochs_number, train_loader, val_loader, m=0.35, s=16
    )
    torch.save(model.state_dict(), f"{save_folder}/model_weights.pth")

    avg_embeddings = compute_avg_embeddings(model, train_data, device)
    torch.save(avg_embeddings, f"{save_folder}/avg_embeddings.pth")
    print("Train done!")


def main():
    fire.Fire(train_and_save)


if __name__ == "__main__":
    main()
