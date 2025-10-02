import logging
from typing import List, Optional, Union
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class OpenWebTextLoader:
    """
    Loads and streams data from the OpenWebText dataset.
    """
    def __init__(self, dataset_name: str = "openwebtext", split: str = "train", streaming: bool = True):
        self.dataset_name = dataset_name
        self.split = split
        self.streaming = streaming
        self.dataset: Optional[Union[DatasetDict, Dataset]] = None

    def load(self):
        """Loads the dataset."""
        logger.info(f"Loading dataset: {self.dataset_name}, split: {self.split}, streaming: {self.streaming}")
        try:
            self.dataset = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
            logger.info(f"Dataset {self.dataset_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def stream_texts(self, num_samples: Optional[int] = None) -> List[str]:
        """
        Streams texts from the loaded dataset.
        
        Args:
            num_samples: Maximum number of samples to stream.
        
        Returns:
            A list of text strings.
        """
        if self.dataset is None:
            self.load()

        texts = []
        for i, example in enumerate(tqdm(self.dataset, desc=f"Streaming {self.dataset_name} texts")):
            if num_samples and i >= num_samples:
                break
            if "text" in example:
                texts.append(example["text"])
            else:
                logger.warning(f"Example {i} in dataset does not contain 'text' key.")
        logger.info(f"Streamed {len(texts)} texts from {self.dataset_name}.")
        return texts

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage:
    loader = OpenWebTextLoader(num_samples=100) # Limit to 100 samples for quick test
    texts = loader.stream_texts()
    
    if texts:
        print(f"First 3 texts:")
        for i, text in enumerate(texts[:3]):
            print(f"--- Text {i+1} ---\n{text[:200]}...") # Print first 200 chars
    else:
        print("No texts streamed.")
