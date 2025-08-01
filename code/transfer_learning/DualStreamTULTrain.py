import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from DualStreamTUL import TwoStageDualStreamModel
from data_utils import *
from metrics import calculate_macro_metrics
from itertools import cycle

parser = argparse.ArgumentParser(description='pytorch version of new_idea for adapterTUL')

parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--b_sz', type=int, default=256)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mps', default=False, help='use mps')
parser.add_argument('--cuda', default=True, help='use gpus')
parser.add_argument('--num_layers', type=int, default=1, help='number of gru layers')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden_channels OF GRU')

parser.add_argument('--type_dim', type=int, default=64, help='type')
parser.add_argument('--geo_dim', type=int, default=32, help='type')
parser.add_argument('--time_dim', type=int, default=32, help='type')

parser.add_argument('--embed_size', type=int, default=128, help='Embedding Size OF NODEs') # Fixed to 128 as our embedding is 128
parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')
parser.add_argument('--z_dim', type=int, default=32, help='z_dim of GRU')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')

parser.add_argument('--num_pretrain_epochs', type=int, default=20, help='num_pretrain_epochs')
parser.add_argument('--num_finetune_epochs', type=int, default=60, help='num_finetune_epochs')

parser.add_argument('--target_csv_file_name', type=str, default=None, help='log file name')

parser.add_argument('--data_set_source_test', type=str, default=None, help='source data set test path')
parser.add_argument('--data_set_source_train', type=str, default=None, help='source data set train path')

parser.add_argument('--data_set_target_test', type=str, default=None, help='target data set test path')
parser.add_argument('--data_set_target_train', type=str, default=None, help='target data set train path')

parser.add_argument('--output_source_user_size',type=int, default=210, help='output user size')
parser.add_argument('--output_target_user_size',type=int, default=210, help='output user size')

parser.add_argument('--remove_entropy',type=bool, default=False, help='remove the entropy loss')
parser.add_argument('--big_entropy',type=int, default=None, help='add a big entropy')
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


def main():
    model = TwoStageDualStreamModel(args.type_dim, args.geo_dim, args.time_dim, args.hidden_dim,args.drop_out, args.num_layers, args.output_source_user_size, args.output_target_user_size, device)
    model.to(device)
    # 1. Prepare source city training set
    source_train_data = read_processed_tra(args.data_set_source_train)  # Training set
    # Assume target_data is a TensorDataset containing trajectories and labels
    source_train_loader = DataLoader(dataset=PandasDataset(source_train_data), batch_size=args.b_sz, shuffle=True,
                                     collate_fn=collate_fn)

    # 2. Prepare target city training and test sets
    target_train_data = read_processed_tra(args.data_set_target_train)  # Training set
    target_test_data = read_processed_tra(args.data_set_target_test)  # Test set
    # Assume target_data is a TensorDataset containing trajectories and labels
    target_train_loader = DataLoader(dataset=PandasDataset(target_train_data), batch_size=args.b_sz, shuffle=True,
                                     collate_fn=collate_fn)
    target_test_loader = DataLoader(dataset=PandasDataset(target_test_data), batch_size=args.b_sz, shuffle=False,
                                    collate_fn=collate_fn)

    # 3. Train and test every epoch, save csv file
    train_two_stage_model(model, source_train_loader, target_train_loader, target_test_loader, args.num_pretrain_epochs, args.num_finetune_epochs, device)


def get_matched_batch(target_iter, target_loader, source_size):
    """Efficiently get target batch matching source batch size"""
    collected_data = []
    collected_labels = []
    collected_lengths = []
    collected_masks = []
    current_size = 0

    while current_size < source_size:
        try:
            data, labels, lengths, masks = next(target_iter)
        except StopIteration:
            # Restart target domain data iteration
            target_iter = iter(target_loader)
            data, labels, lengths, masks = next(target_iter)

        # Calculate needed amount
        needed = min(source_size - current_size, data.size(0))

        # Add to collection lists separately
        collected_data.append(data[:needed])
        collected_labels.append(labels[:needed])
        collected_lengths.append(lengths[:needed])
        collected_masks.append(masks[:needed])

        current_size += needed

    # Concatenate separately and return
    return (
        torch.cat(collected_data, dim=0),
        torch.cat(collected_labels, dim=0),
        torch.cat(collected_lengths, dim=0),
        torch.cat(collected_masks, dim=0)
    )

def train_two_stage_model(
        model, source_loader, target_loader, target_test_loader,
        num_pretrain_epochs=20, num_finetune_epochs=60,
        device='cuda', patience=5
):
    """Two-stage training: 1. Pretrain GRU 2. Finetune TUL classifier"""
    # First stage: Pretrain GRU (using vector prediction task)
    print("===== Stage 1: Pretrain GRU =====")
    pretrain_optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(pretrain_optimizer, step_size=30, gamma=0.5) # Reduce by half every 30 epochs

    best_loss = float('inf')
    early_stop_counter = 0

    # Get current date
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    csv_file_name = f"{current_date}" + args.target_csv_file_name

    for epoch in range(num_pretrain_epochs):
        model.train()
        total_loss = 0.0

        # Synchronously iterate source and target domain data, cycle target data due to smaller size
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        # Linear growth entropy constraint weight scheduler
        base_lambda = 0.1
        max_lambda = 1.0
        progress = epoch / num_pretrain_epochs
        entropy_lambad = base_lambda + progress * (max_lambda - base_lambda)

        if(args.remove_entropy):
            entropy_lambad = 0
            print('entropy_loss is remove!')
        if(args.big_entropy != None):
            entropy_lambad = args.big_entropy
            print('big_entropy is active, entropy_lambad is {}!'.format(args.big_entropy))

        timeStart = datetime.datetime.now()
        for _ in range(len(source_loader)):

            source_batch = next(source_iter)
            target_batch = next(target_iter)

            # Force batch alignment
            if(source_batch[0].size(0) > target_batch[0].size(0)):
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
            if(source_batch[0].size(0) < target_batch[0].size(0)):
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)

            source_trajs, source_labels, source_lens, source_mask = source_batch
            target_trajs, target_labels, target_lens, target_mask = target_batch

            source_trajs, source_labels, source_lens, source_mask = source_trajs.to(device), source_labels.to(device), source_lens.to(device), source_mask.to(device)
            target_trajs, target_labels, target_lens, target_mask = target_trajs.to(device), target_labels.to(device), target_lens.to(device), target_mask.to(device)

            # Forward pass, trajectory reconstruction
            loss = model(source_trajs, source_labels, source_lens, source_mask,
                target_trajs, target_labels, target_lens, target_mask,
                         entropy_lambad=entropy_lambad ,mode='pretrain')

            # Backpropagation
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

            total_loss += loss.item()

        # Update learning rate
        avg_loss = total_loss / len(source_loader)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_pretrain_epochs}: Loss={avg_loss:.4f} "
              f"lr={pretrain_optimizer.param_groups[0]['lr']:.4f}")
        timeEnd = datetime.datetime.now()
        timeSeconds = (timeEnd - timeStart).seconds
        save_training_info_to_csv(csv_file_name,'train', timeSeconds, epoch, pretrain_optimizer.param_groups[0]['lr'], avg_loss, -1,-1,-1,-1,-1)

        # Test every epoch
        model.eval()
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            loss_a = 0
            correct5 = 0
            test_data_len = len(target_test_loader.dataset)
            all_preds = []
            all_labels = []

            timeStart = datetime.datetime.now()
            for target_trajs, target_labels, target_lens, target_mask in target_test_loader:
                target_trajs, target_labels, target_lens, target_mask = target_trajs.to(device), target_labels.to(
                    device), target_lens.to(device), target_mask.to(device)

                # Forward pass
                logistic_outputs, logistic_preds, loss = model(None, None, None, None, target_trajs, target_labels,
                                                               target_lens, target_mask,
                                                               mode='finetune')

                # Save predictions and true labels for entire epoch
                all_preds.extend(logistic_preds.cpu().tolist())
                all_labels.extend(target_labels.cpu().tolist())
                # Calculate top-1 classification accuracy
                correct += int((logistic_preds == target_labels).sum())
                # Accumulate average loss for each batch in epoch
                loss_a += loss
                top_k = 5
                out_np = logistic_outputs.cpu().detach().numpy()
                for index, o in enumerate(out_np):
                    top5 = o.argsort()[::-1][:top_k]
                    if int(target_labels[index]) in top5:
                        correct5 = correct5 + 1

            # Calculate macro metrics for entire test set
            # top1, top5 accuracy
            acc1 = correct / test_data_len
            acc5 = correct5 / test_data_len
            # Precision, recall and F1 score
            macro_f, macro_p, macro_r = calculate_macro_metrics(all_preds, all_labels)

            print(f"----------!!epoch:{epoch} pretrain TEST !!-----------------")
            print(f"learning_rate: {pretrain_optimizer.param_groups[0]['lr']:.4f}")
            print(f"epoch_total_sum_loss_is: {loss_a / (test_data_len // args.b_sz):.4f}")
            print('---logistic_acc1', acc1, '---logistic_acc5', acc5)
            print(
                f"logistic_Macro-F1: {macro_f:.4f}, logistic_Precision: {macro_p:.4f}, logistic_Recall: {macro_r:.4f}")

            timeEnd = datetime.datetime.now()
            timeSeconds = (timeEnd - timeStart).seconds
            save_training_info_to_csv(csv_file_name,'test', timeSeconds, epoch, pretrain_optimizer.param_groups[0]['lr'],
                                      loss_a / test_data_len, acc1, acc5, macro_f, macro_p, macro_r)

    # Second stage: Finetune TUL classifier
    print("===== Stage 2: Finetune TUL classifier =====")
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Then only unfreeze gru + classifier
    for param in model.encoder_gru.parameters():
        param.requires_grad = True
    for param in model.tul_classifier.parameters():
        param.requires_grad = True

    # Only train TUL classifier and encoder_gru
    finetune_optimizer = optim.Adam(model.parameters(), lr=0.001)
    finetune_scheduler = StepLR(finetune_optimizer, step_size=30, gamma=0.5)  # Learning rate drops by half every 30 epochs

    for epoch in range(num_finetune_epochs):
        # Actual epoch is num_pretrain_epochs + epoch
        epoch = num_pretrain_epochs + epoch

        # Train
        model.train()

        # Accuracy metrics calculation
        correct = 0
        loss_a = 0
        total_loss = 0
        correct5 = 0
        train_data_len = len(target_loader.dataset)
        all_preds = []
        all_labels = []

        timeStart = datetime.datetime.now()

        for target_trajs, target_labels, target_lens, target_mask in target_loader:
            target_trajs, target_labels, target_lens, target_mask = target_trajs.to(device), target_labels.to(device), target_lens.to(device), target_mask.to(device)

            # Forward pass
            logistic_outputs, logistic_preds, loss = model(None, None, None, None,
                target_trajs, target_labels, target_lens, target_mask, mode='finetune')

            # Backpropagation
            finetune_optimizer.zero_grad()
            loss.backward()
            finetune_optimizer.step()
            total_loss += loss.item()

            # Save predictions and true labels for entire epoch
            all_preds.extend(logistic_preds.cpu().tolist())
            all_labels.extend(target_labels.cpu().tolist())
            # Calculate top-1 classification accuracy
            correct += int((logistic_preds == target_labels).sum())
            # Accumulate average loss for each batch in epoch
            loss_a += loss
            top_k = 5
            out_np = logistic_outputs.cpu().detach().numpy()
            for index, o in enumerate(out_np):
                top5 = o.argsort()[::-1][:top_k]
                if int(target_labels[index]) in top5:
                    correct5 = correct5 + 1

        avg_loss = total_loss / len(target_loader)
        finetune_scheduler.step()

        # Calculate macro metrics for entire test set
        # top1, top5 accuracy
        acc1 = correct / train_data_len
        acc5 = correct5 / train_data_len
        # Precision, recall and F1 score
        macro_f, macro_p, macro_r = calculate_macro_metrics(all_preds, all_labels)

        print(f"----------!!epoch:{epoch}!!-----------------")
        print(f"learning_rate: {finetune_optimizer.param_groups[0]['lr']:.4f}")
        print(f"epoch_total_sum_loss_is: {loss_a / (train_data_len // args.b_sz):.4f}")
        print('---logistic_acc1', acc1, '---logistic_acc5', acc5)
        print(f"logistic_Macro-F1: {macro_f:.4f}, logistic_Precision: {macro_p:.4f}, logistic_Recall: {macro_r:.4f}")
        timeEnd = datetime.datetime.now()
        timeSeconds = (timeEnd - timeStart).seconds
        save_training_info_to_csv(csv_file_name, 'train', timeSeconds, epoch, pretrain_optimizer.param_groups[0]['lr'],
                                  loss_a / test_data_len, acc1, acc5, macro_f, macro_p, macro_r)

        # Test every epoch
        model.eval()
        timeStart = datetime.datetime.now()
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            loss_a = 0
            correct5 = 0
            test_data_len = len(target_test_loader.dataset)
            all_preds = []
            all_labels = []

            for target_trajs, target_labels, target_lens, target_mask in target_test_loader:
                target_trajs, target_labels, target_lens, target_mask = target_trajs.to(device), target_labels.to(
                    device), target_lens.to(device), target_mask.to(device)

                # Forward pass
                logistic_outputs, logistic_preds, loss = model(None, None, None, None,target_trajs, target_labels, target_lens, target_mask,
                                       mode='finetune')

                # Save predictions and true labels for entire epoch
                all_preds.extend(logistic_preds.cpu().tolist())
                all_labels.extend(target_labels.cpu().tolist())
                # Calculate top-1 classification accuracy
                correct += int((logistic_preds == target_labels).sum())
                # Accumulate average loss for each batch in epoch
                loss_a += loss
                top_k = 5
                out_np = logistic_outputs.cpu().detach().numpy()
                for index, o in enumerate(out_np):
                    top5 = o.argsort()[::-1][:top_k]
                    if int(target_labels[index]) in top5:
                        correct5 = correct5 + 1

            # Calculate macro metrics for entire test set
            # top1, top5 accuracy
            acc1 = correct / test_data_len
            acc5 = correct5 / test_data_len
            # Precision, recall and F1 score
            macro_f, macro_p, macro_r = calculate_macro_metrics(all_preds, all_labels)

            print(f"----------!!epoch:{epoch} finetune TEST !!-----------------")
            print(f"learning_rate: {finetune_optimizer.param_groups[0]['lr']:.4f}")
            print(f"epoch_total_sum_loss_is: {loss_a / (test_data_len // args.b_sz):.4f}")
            print('---logistic_acc1', acc1, '---logistic_acc5', acc5)
            print(f"logistic_Macro-F1: {macro_f:.4f}, logistic_Precision: {macro_p:.4f}, logistic_Recall: {macro_r:.4f}")
            timeEnd = datetime.datetime.now()
            timeSeconds = (timeEnd - timeStart).seconds
            save_training_info_to_csv(csv_file_name,'test', timeSeconds, epoch, finetune_optimizer.param_groups[0]['lr'],
                                      loss_a / test_data_len, acc1, acc5, macro_f, macro_p, macro_r)

if __name__ == '__main__':
    main()