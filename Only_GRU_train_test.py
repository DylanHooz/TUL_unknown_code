#《You should ignore the id for tul task》
import argparse
import datetime
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from Only_GRU_model import Only_GRU,Only_GRU_TUL,LinearLogits
from torch.utils.data import DataLoader, Dataset
import pickle
from metrics import accuracy_at_k, calculate_macro_metrics
from Only_GRU_MergeEmbeddingNetwork_model import Only_GRU_MERGE
from save_csv import save_training_info_to_csv

# Training script for VAE encoding
SEED = 42
np.random.seed(SEED)

torch.set_default_tensor_type(torch.FloatTensor)
parser = argparse.ArgumentParser(description='pytorch version of new_idea for MAE_TUL')

parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--b_sz', type=int, default=256)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False, help='use mps')
parser.add_argument('--num_layers', type=int, default=1, help='number of gru layers')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden_channels OF GRU')
parser.add_argument('--embed_size', type=int, default=128, help='Embedding Size OF NODEs') #this is the default
parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--file_name', type=str, default="AAAA_newGeoEmbed_newGPT2_nyk_400user_foursquare_6hour_schedule20decay_80epoch_adaml2_lr1e-3.csv", help='log file name')
parser.add_argument('--data_set_test', type=str, default=None, help='data set path')
parser.add_argument('--data_set_train', type=str, default=None, help='data set path')
parser.add_argument('--output_user_size',type=int, default=401, help='output user size + 1')
parser.add_argument('--checkpoint_save_path', type=str, default='./foursquare_best_checkpoint.pt', help='checkpoint save path')
parser.add_argument('--dynamic_balance', type=bool, default=True, help='whether to add dynamic balance')
args = parser.parse_args()

if torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have a mps device, so you should probably run with --mps")
    else:
        device = torch.device("mps")
        print(f"\nCurrent MPS Device: {device}")
elif torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        device = torch.device("cuda")
        print('using device cuda', device_id, torch.cuda.get_device_name(device_id))
else:
    print('CPU mode')
    device = torch.device("cpu")

print('DEVICE:', device)

# Get current date
current_date = datetime.datetime.now().strftime("%Y%m%d")

def main():
    starttime = datetime.datetime.now()
    seed = args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    z_dim = 32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load original dataset meta information
    output_user_size = args.output_user_size

    # Load datasets - separate train and test to apply random masking only to training set
    train_data = read_processed_tra(args.data_set_train)
    test_data = read_processed_tra(args.data_set_test)

    # Main model: encoder, decoder, classifier
    encoder = Only_GRU(embed_size=args.embed_size, hidden_size=args.hidden_size, output_size=args.hidden_size, dropout_prob=args.drop_out, num_layers=args.num_layers,
                           device=device)
    logistic = LinearLogits(input_dim=args.hidden_size, hidden_dim=args.hidden_size, dropout_prob=args.drop_out, num_classes=output_user_size, device=device)
    model = Only_GRU_TUL(encoder=encoder,logistic=logistic, device=device, hidden_size=args.hidden_size,
                    dropout_prob=args.drop_out, h_dim=args.hidden_size, z_dim=z_dim).to(device)

    # Single optimizer approach
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Learning rate decay every 20 epochs
    # Early stopping
    stopper = EarlyStopping(patience=5, model_name=args.checkpoint_save_path)

    # DataLoader splits dataset into batches, shuffle=True removes order features
    train_data_loader = DataLoader(dataset=PandasDataset(train_data), batch_size=args.b_sz, shuffle=True, collate_fn=collate_fn)
    test_data_loader = DataLoader(dataset=PandasDataset(test_data), batch_size=args.b_sz, shuffle=True, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        model.train()

        # Epoch accuracy and loss tracking
        correct = 0
        loss_a = 0
        correct5 = 0
        train_data_len = len(train_data)
        all_preds = []
        all_labels = []

        timeStart = datetime.datetime.now()
        for i, (traj, user_label, lengths, mask) in enumerate(train_data_loader):
            traj = traj.to(device)
            user_label = user_label.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)  # mask: 1=keep, 0=masked

            # Model forward pass
            h, h_mse, mu, log_var, logistic_output, logistic_preds, logistic_loss = model(encoder_input=traj, decoder_input=traj, user_label=user_label, lengths=lengths,
                                                              mask=mask)

            # Backpropagation and loss update
            loss = h_mse + logistic_loss

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            # Save predictions and labels for entire epoch
            all_preds.extend(logistic_preds.cpu().tolist())
            all_labels.extend(user_label.cpu().tolist())

            # Calculate top5 classification accuracy
            correct += int((logistic_preds == user_label).sum())
            # Accumulate batch average loss
            loss_a += loss

            top_k = 5
            out_np = logistic_output.cpu().detach().numpy()
            for index, o in enumerate(out_np):
                top5 = o.argsort()[::-1][:top_k]
                if int(user_label[index]) in top5:
                    correct5 = correct5 + 1

        # Learning rate adjustment after epoch
        scheduler.step()

        # Calculate top1, top5 accuracy
        acc1 = correct / train_data_len
        acc5 = correct5 / train_data_len
        # Calculate precision, recall and F1 score
        macro_f, macro_p, macro_r = calculate_macro_metrics(all_preds, all_labels)

        # Print once per epoch
        print(f"----------epoch:{epoch}------------------")
        print(f"learning_rate: {optimizer.param_groups[0]['lr']:.4f}")
        print(f"epoch_total_sum_loss_is: {loss_a/(train_data_len//args.b_sz):.4f}")
        print('---logistic_acc1', acc1, '---logistic_acc5', acc5)
        print(f"logistic_Macro-F1: {macro_f:.4f}, logistic_Precision: {macro_p:.4f}, logistic_Recall: {macro_r:.4f}")

        timeEnd = datetime.datetime.now()
        trainSeconds = (timeEnd - timeStart).seconds
        csv_file_name = f"{current_date}" + args.file_name
        save_training_info_to_csv(csv_file_name, 'train', trainSeconds, epoch, optimizer.param_groups[0]['lr'],
                                  loss_a / train_data_len, acc1, acc5, macro_f, macro_p, macro_r)

        # Evaluation once per epoch
        model.eval()
        with torch.no_grad():
            # Test set accuracy and loss tracking
            correct = 0
            loss_a = 0
            correct5 = 0
            test_data_len = len(test_data)
            all_preds = []
            all_labels = []

            timeStart = datetime.datetime.now()
            for i, (traj, user_label, lengths, mask) in enumerate(test_data_loader):
                traj = traj.to(device)
                user_label = user_label.to(device)
                lengths = lengths.to(device)
                mask = mask.to(device)
                h, h_mse, mu, log_var, logistic_output, logistic_preds, logistic_loss = model(encoder_input=traj, decoder_input=traj,user_label=user_label, lengths=lengths,
                                                             mask=mask)
                loss = logistic_loss + h_mse

                # Save predictions and labels for entire epoch
                all_preds.extend(logistic_preds.cpu().tolist())
                all_labels.extend(user_label.cpu().tolist())
                # Calculate top5 classification accuracy
                correct += int((logistic_preds == user_label).sum())
                # Accumulate epoch batch average loss
                loss_a += loss
                top_k = 5
                out_np = logistic_output.cpu().detach().numpy()
                for index, o in enumerate(out_np):
                    top5 = o.argsort()[::-1][:top_k]
                    if int(user_label[index]) in top5:
                        correct5 = correct5 + 1

            # Calculate macro metrics for entire test set
            acc1 = correct / test_data_len
            acc5 = correct5 / test_data_len
            macro_f, macro_p, macro_r = calculate_macro_metrics(all_preds, all_labels)

            print(f"----------!!epoch:{epoch}-TEST!!-----------------")
            print(f"learning_rate: {optimizer.param_groups[0]['lr']:.4f}")
            print(f"epoch_total_sum_loss_is: {loss_a/(test_data_len//args.b_sz):.4f}")
            print('---logistic_acc1', acc1, '---logistic_acc5', acc5)
            print(f"logistic_Macro-F1: {macro_f:.4f}, logistic_Precision: {macro_p:.4f}, logistic_Recall: {macro_r:.4f}")

            timeEnd = datetime.datetime.now()
            testSeconds = (timeEnd - timeStart).seconds
            # Save to CSV file
            csv_file_name = f"{current_date}"+args.file_name
            save_training_info_to_csv(csv_file_name,'test', testSeconds, epoch, optimizer.param_groups[0]['lr'], loss_a/test_data_len, acc1, acc5, macro_f, macro_p, macro_r)

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
            torch.tensor([each[1] for each in row[0]], dtype=torch.long),  # traj: POI ID list
            torch.tensor([each[3] + each[4] + each[5] for each in row[0]], dtype=torch.float),  # POI typeName, time, geo encoding
            torch.tensor(row[0][0][0], dtype=torch.long),  # user
            torch.tensor(len(row[1]), dtype=torch.long),  # length
            torch.tensor(row[1], dtype=torch.long)  # mask (variable length)
        ]

def collate_fn(batch):
    """
    Handle variable length data padding to uniform length
    """
    # Unpack batch
    trajs, traj_embed, users, lengths, masks = zip(*batch)

    # Pad traj and mask
    padded_trajs_embed = pad_sequence(traj_embed, batch_first=True)  # Shape: (B, max_len, ...)
    padded_masks = pad_sequence(masks, batch_first=True)  # Shape: (B, max_len)

    return (
        padded_trajs_embed,
        torch.stack(users),  # user is scalar, can be stacked directly
        torch.stack(lengths),  # length is scalar
        padded_masks
    )

def read_data_meta_info(meta_info_path) -> dict:
    with open(meta_info_path, 'rb') as f:
        dic = pickle.load(f)
    return dic

class EarlyStopping:
    '''
    Early stopping for quick results
    '''
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

if __name__ == '__main__':
    main()
