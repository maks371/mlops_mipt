import hydra
import torch
from omegaconf import DictConfig
from torchvision import transforms

from face_recognition.classification import get_predictions
from face_recognition.custom_dataset import CelebADataset
from face_recognition.model import CosModel


def predict_and_save(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(cfg["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
        ]
    )

    test_data = CelebADataset(cfg["data_folder"], "test", transform)

    batch_size = cfg["batch_size"]
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    model = CosModel(cfg["n_classes"], cfg["output_dim"]).to(device)
    model.load_state_dict(torch.load(f"{cfg['model_folder']}/model_weights.pth"))

    avg_embeddings = torch.load(f"{cfg['model_folder']}/avg_embeddings.pth")

    predictions = get_predictions(model, test_loader, device, avg_embeddings)

    with open(cfg["save_file"], "w") as fp:
        for label in predictions:
            # write each item on a new line
            fp.write("%s\n" % label)

    print("Inference done!")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    predict_and_save(cfg["infer_params"])


if __name__ == "__main__":
    main()
