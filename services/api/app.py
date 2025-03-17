from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI, LitServer
from config import config
import numpy as np
from loguru import logger

class BERTLitAPI(LitAPI):
    def setup(self, device):
        """
        Load the tokenizer and model, and move the model to the specified device.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.pre_trained_bert_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(config.hf_model_registry)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error during model setup: {e}")
            raise
    
    def decode_request(self, request):
        """
        Preprocess the request data (tokenize)
        """
        try:
            if "title" not in request:
                raise ValueError("Missing 'title' field in request.")
            inputs = self.tokenizer(request["title"], return_tensors="pt")
            return inputs
        except Exception as e:
            logger.error(f"Error decoding request: {e}")
            raise

    def predict(self, inputs):
        """
        Perform the inference
        """
        try:
            with torch.no_grad():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
            return outputs.logits
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def encode_response(self, logits):
        """
        Process the model output into a response dictionary
        """
        try:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = probabilities.argmax().item()
            predicted_class = config.id2label.get(predicted_class_id, "Unknown")
            probability = np.round(probabilities[0][predicted_class_id].item(), 4)
            return {"predicted_class": predicted_class, "probability": probability}
        except Exception as e:
            logger.error(f"Error encoding response: {e}")
            raise

if __name__ == "__main__":
    try:
        api = BERTLitAPI()
        server = LitServer(api)
        server.run(port=8000)
    except Exception as e:
        logger.error(f"Error starting the server: {e}")