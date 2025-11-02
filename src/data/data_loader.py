#load dataset here 
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name:str="Dahoas/rm-static"):
        self.dataset_name = dataset_name
        self.dataset = self.load_data()
    
    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def train_dataset(self):
        "Returns train dataset"
        return self.dataset['train']
    
    def test_dataset(self):
        "Returns test dataset"
        return self.dataset['test']
    
    def __repr__(self):
        return f"DataLoader for {self.dataset_name}"


#Test 
# if __name__ == "__main__":
#     data_loader = DataLoader("Dahoas/rm-static")
#     dataset = data_loader.load_data()
#     print(data_loader.test_dataset())
#     print(data_loader.train_dataset())
