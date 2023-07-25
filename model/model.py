from typing import Dict

from model.llama_generation import Llama

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._generator = None

    def load(self):
        ckpt_dir = "/llama/llama-7b-chat"
        tokenizer_path = "/llama/tokenizer.model"
        
        self._generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=512,
            max_batch_size=4,
        )

    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                dialogs = [
                    {"role": "user", "content": f"{request.pop('prompt')}"}
                ]
                results = self._generator.chat_completion(
                    dialogs,
                    max_gen_len=None,
                    temperature=0.6,
                    top_p=0.9,
                )
                return {"data": f"{result['generation']['content']}"}
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
