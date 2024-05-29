import hydra
import torch
from omegaconf import DictConfig
from torchvision import transforms

from face_recognition.classification import compute_avg_embeddings
from face_recognition.custom_dataset import CelebADataset
from face_recognition.model import CosModel
from face_recognition.train_model import train_model


def train_and_save(cfg: DictConfig):
    transform = transforms.Compose(
        [
            transforms.Resize(cfg["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
        ]
    )

    train_data = CelebADataset(cfg["data_folder"], "train", transform)
    val_data = CelebADataset(cfg["data_folder"], "val", transform)

    batch_size = cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = train_model(cfg, train_loader, val_loader, device)
    torch.save(model.state_dict(), f"{cfg['save_folder']}/model_weights.pth")

    avg_embeddings = compute_avg_embeddings(model, train_data, device, cfg["n_classes"])
    torch.save(avg_embeddings, f"{cfg['save_folder']}/avg_embeddings.pth")
    print("Train done!")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train_and_save(cfg["train_params"])


if __name__ == "__main__":
    main()
