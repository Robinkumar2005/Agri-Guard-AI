import torch
import torch.nn as nn
import logging
import json
from torchvision import transforms
from app.models.config import settings

logger = logging.getLogger(__name__)

model = None
class_names = None

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def get_model():
    global model, class_names
    if model is None:
        try:
            model_path = settings.MODEL_PATH
            class_path = settings.CLASS_PATH
            with open(class_path, "r", encoding="utf-8") as f:
                loaded_classes = json.load(f)

            if len(loaded_classes) != 38:
                raise ValueError(
                    "Expected 38 class names."
                )

            loaded_model = CustomCNN(num_classes=38)

            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )

            loaded_model.load_state_dict(state_dict)
            loaded_model.to(device)
            loaded_model.eval()

            model = loaded_model
            class_names = loaded_classes

            logger.info(
                f"Model loaded successfully on {device}."
            )

        except Exception as e:
            model = None
            class_names = None
            logger.exception(
                f"Error loading model: {str(e)}"
            )

            raise RuntimeError(
                "Failed to load model."
            ) from e

    return model


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def single_predict(image):
    try:
        loaded_model = get_model()
        image = image.convert("RGB")
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = loaded_model(image_tensor)
            probabilities = torch.softmax(
                outputs, dim=1
            )
            confidence, predicted = torch.max(
                probabilities, dim=1
            )
        predicted_index = predicted.item()
        return {
            "disease": class_names[predicted_index],
            "confidence": round(
                confidence.item(), 4
            )
        }

    except Exception as e:
        logger.exception(
            f"Error during single prediction: {str(e)}"
        )
        raise RuntimeError(
            "Single prediction failed."
        ) from e


def batch_predict(images):
    predictions = []
    try:
        loaded_model = get_model()
        with torch.inference_mode():
            for image in images:
                try:
                    image = image.convert("RGB")
                    image_tensor = preprocess(image)
                    image_tensor = image_tensor.unsqueeze(0).to(device)
                    outputs = loaded_model(image_tensor)
                    probabilities = torch.softmax(
                        outputs, dim=1
                    )
                    confidence, predicted = torch.max(
                        probabilities, dim=1
                    )
                    predicted_index = predicted.item()
                    predictions.append({
                        "disease": class_names[predicted_index],
                        "confidence": round(
                            confidence.item(), 4
                        )
                    })

                except Exception as e:
                    logger.exception(
                        f"Error processing batch image: {str(e)}"
                    )
                    predictions.append({

                        "error": "Prediction failed."
                    })
        return predictions

    except Exception as e:
        logger.exception(
            f"Batch prediction failed: {str(e)}"
        )
        raise RuntimeError(
            "Batch prediction failed."
        ) from e
