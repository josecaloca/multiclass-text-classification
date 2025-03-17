import numpy as np
import torch
from config import config
from litserve import LitAPI, LitServer
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BERTLitAPI(LitAPI):
    def setup(self, device):
        """
        Loads the pre-trained BERT tokenizer and model, moves the model to the specified device,
        and sets the model to evaluation mode.

        Args:
            device (str): The device to which the model should be moved (e.g., 'cuda' or 'cpu').
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.pre_trained_bert_model
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.hf_model_registry
            )
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            logger.error(f'Error during model setup: {e}')
            raise

    def decode_request(self, request):
        """
        Tokenizes the input text from the request.

        Args:
            request (dict): The input request containing a 'title' field with text to be classified.

        Returns:
            dict: A dictionary containing tokenized inputs in PyTorch tensor format.
        
        Raises:
            ValueError: If the 'title' field is missing from the request.
        """
        try:
            if 'title' not in request:
                raise ValueError("Missing 'title' field in request.")
            return self.tokenizer(request['title'], return_tensors='pt')
        except Exception as e:
            logger.error(f'Error decoding request: {e}')
            raise

    def predict(self, inputs):
        """
        Performs inference on the tokenized inputs using the loaded BERT model.

        Args:
            inputs (dict): A dictionary of tokenized inputs suitable for the model.

        Returns:
            torch.Tensor: The raw logits output from the model.
        """
        try:
            with torch.no_grad():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
            return outputs.logits
        except Exception as e:
            logger.error(f'Error during prediction: {e}')
            raise

    def encode_response(self, logits):
        """
        Converts the model output logits into a human-readable response with predicted class
        and confidence score.

        Args:
            logits (torch.Tensor): The output logits from the model.

        Returns:
            dict: A dictionary containing the predicted class and its probability.
        """
        try:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = probabilities.argmax().item()
            predicted_class = config.id2label.get(predicted_class_id, 'Unknown')
            probability = np.round(probabilities[0][predicted_class_id].item(), 4)
            return {'predicted_class': predicted_class, 'probability': probability}
        except Exception as e:
            logger.error(f'Error encoding response: {e}')
            raise


if __name__ == '__main__':
    try:
        api = BERTLitAPI()
        server = LitServer(api, api_path="/predict", healthcheck_path ="/health")
        server.run(port=8000)
    except Exception as e:
        logger.error(f'Error starting the server: {e}')
