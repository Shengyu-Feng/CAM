from torch.utils.data import Dataset, DataLoader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
    
class BatchDataset(Dataset):
    def __init__(self, batched_data):
        """
        Args:
            batched_data (list of tuples): Each tuple is a batch, typically (inputs, targets)
        """
        self.data = batched_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return the pre-batched data
        return self.data[index]
    
class BatchBuffer:
    def __init__(self):
        self.batch_list = []

    def add(self, batched_data):
        self.batch_list.extend(batched_data)

    def get_dataset(self):
        dataset = BatchDataset(self.batch_list)
        return dataset