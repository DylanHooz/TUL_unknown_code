import csv
import datetime
import os
import pickle
import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence

def read_processed_tra(traj_path):
    try:
        with open(traj_path, "rb") as f:
            loaded_data = pickle.load(f)
            return loaded_data
    except FileNotFoundError:
        print(f"Error: File {traj_path} not found.")
    except Exception as e:
        print(f"Error: An unknown error occurred: {e}")

class PandasDataset(Dataset):
    def __init__(self, list_data):
        self.list_data = list_data

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        row = self.list_data[idx]
        return [
            torch.tensor([each[1] for each in row[0]], dtype=torch.long), # traj: list of poi_ids
            torch.tensor([each[3] + each[4] + each[5] for each in row[0]], dtype=torch.float), # POI typeName, time, geo encodings
            torch.tensor(row[0][0][0], dtype=torch.long), # user
            torch.tensor(len(row[1]), dtype=torch.long), # length
            torch.tensor(row[1], dtype=torch.long) # mask (variable length)
        ]


def collate_fn(batch):
    """
    Handle variable-length data padding to uniform length
    """
    # Unpack batch
    trajs, traj_embed, users, lengths, masks = zip(*batch)

    # Pad traj and mask
    padded_trajs_embed = pad_sequence(traj_embed, batch_first=True)  # Shape: (B, max_len, ...)
    padded_masks = pad_sequence(masks, batch_first=True)  # Shape: (B, max_len)

    return (
        padded_trajs_embed,
        torch.stack(users),  # user is scalar, can be directly stacked
        torch.stack(lengths),  # length is scalar
        padded_masks
    )

class EarlyStopping:
    """
    Early stopping for fast convergence
    """

    def __init__(self, patience=10, model_name='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_name = model_name

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.model_name)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.model_name)
            self.counter = 0
        return self.early_stop

def save_training_info_to_csv(file_name, train_or_test, seconds,epoch, lr, loss, acc1, acc5, macro_f, macro_p, macro_r):
    # Get current date
    current_date = datetime.datetime.now().strftime("%Y%m%d %H")
    # Define CSV column headers
    headers = ['train_or_test','seconds','epoch', 'lr', 'total_sum_loss', 'logistic_acc1', 'logistic_acc5', 'logistic_Macro-F1', 'logistic_Precision',
               'logistic_Recall']

    # Check if file exists
    file_exists = os.path.exists('./log/' + file_name)

    # Open CSV file in append mode
    with open('./log/' + file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write headers if file is new
        if not file_exists:
            writer.writeheader()

        # Write information to CSV file
        writer.writerow({
            'train_or_test': train_or_test,
            'seconds': seconds,
            'epoch': epoch,
            'lr': lr,
            'total_sum_loss': f"{loss:.4f}",
            'logistic_acc1': f"{acc1:.4f}",
            'logistic_acc5': f"{acc5:.4f}",
            'logistic_Macro-F1': f"{macro_f:.4f}",
            'logistic_Precision': f"{macro_p:.4f}",
            'logistic_Recall': f"{macro_r:.4f}"
        })

    print(f"write row into csv file {file_name} finished")

