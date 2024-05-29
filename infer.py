import fire
import torch
from torchvision import transforms

from face_recognition.classification import get_predictions
from face_recognition.custom_dataset import CelebADataset
from face_recognition.model import CosModel


def predict_and_save(data_folder, model_folder, save_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_data = CelebADataset(data_folder, "test", transform)

    batch_size = 128
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    model = CosModel().to(device)
    model.load_state_dict(torch.load(f"{model_folder}/model_weights.pth"))

    avg_embeddings = torch.load(f"{model_folder}/avg_embeddings.pth")

    predictions = get_predictions(model, test_loader, device, avg_embeddings)

    with open(save_file, "w") as fp:
        for label in predictions:
            # write each item on a new line
            fp.write("%s\n" % label)

    print("Inference done!")


def main():
    fire.Fire(predict_and_save)


if __name__ == "__main__":
    main()
