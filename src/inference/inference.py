from tqdm import tqdm
import torch

class Inference():
    def __init__(self,model,tokenizer,max_len, device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def inference(self,texts):  
        ''' Same preprocessing steps should be done for both trainnig and inferencing phases before inferencing. 
            Include all the preprocessing steps in training here for inference.
            input : List of texts 
            output: List of integers coresponding to each text input. 
        '''
        results = []

        for text in tqdm(texts, desc="Inferencing"):
            text = " ".join(text.split())

            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )

            ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)
            token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(ids, mask, token_type_ids)

            _, predicted_class = torch.max(output, dim=1)
            results.append(predicted_class.item())

        return results
