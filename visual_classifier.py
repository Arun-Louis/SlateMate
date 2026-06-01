from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("distilbert_binary_outputs") / "best_model"


class VisualClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT model folder not found: {self.model_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "NO_VISUAL",
            1: "VISUAL"
        }

    def predict(self, text: str):
        if not text or not text.strip():
            return {
                "label": "NO_VISUAL",
                "confidence": 0.0
            }

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted_id = torch.max(probabilities, dim=-1)

        predicted_id = int(predicted_id.item())
        confidence = float(confidence.item())

        return {
            "label": self.id2label[predicted_id],
            "confidence": confidence
        }


_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = VisualClassifier()
    return _classifier


def predict_visual_need(text: str):
    classifier = get_classifier()
    return classifier.predict(text)


if __name__ == "__main__":
    samples = [
        "The equation y equals mx plus b represents a straight line with slope m and intercept b.",
        "Today we will discuss the homework deadline and next week's schedule."
    ]

    for text in samples:
        print(text)
        print(predict_visual_need(text))
        print()