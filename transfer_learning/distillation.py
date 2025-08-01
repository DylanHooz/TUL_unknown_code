import torch.nn.functional as F
import torch
import torch.optim as optim
from data_utils import save_training_info_to_csv
from new_idea.metrics import calculate_macro_metrics

# Knowledge distillation, transfer and training functions

# Feature-level distillation loss (core improvement: ignore classification layer differences, only distill GRU features)
def feature_distillation_loss(target_features, source_features, temperature=1.0):
    # Normalize features for enhanced stability
    target_features = F.normalize(target_features, p=2, dim=1)
    source_features = F.normalize(source_features, p=2, dim=1)

    # Use cosine similarity loss instead of MSE, better for capturing feature distributions
    cosine_loss = 1 - F.cosine_similarity(target_features, source_features).mean()
    return cosine_loss


# Adaptive regularization (maintain proximity to source model parameters)
def adaptive_regularization(target_model, source_model, lambda_reg=0.1):
    reg_loss = 0.0
    for (name_t, param_t), (name_s, param_s) in zip(
            target_model.named_parameters(), source_model.named_parameters()
    ):
        if 'logistic' not in name_t:  # Only regularize GRU layers
            reg_loss += torch.norm(param_t - param_s, p=2)
    return lambda_reg * reg_loss


# Progressive unfreezing strategy (improvement: gradually unfreeze GRU layers during training)
def progressive_unfreeze(target_model, epoch, total_epochs, freeze_ratio=0.7):
    """
    Freeze all GRU layers for the first freeze_ratio of training epochs, then gradually unfreeze
    """
    if freeze_ratio < 0: # Directly unfreeze all
        for param in target_model.encoder.parameters():
            param.requires_grad = True
        print("unfreeze ALL encoder layer !!!")
        return

    if epoch < total_epochs * freeze_ratio:
        for param in target_model.encoder.parameters():
            param.requires_grad = False
    else:
        # Calculate unfreezing ratio
        thaw_ratio = (epoch - total_epochs * freeze_ratio) / (total_epochs * (1 - freeze_ratio))
        # Gradually unfreeze GRU layers (assuming two GRU layers)
        num_layers = target_model.encoder.num_layers
        layers_to_unfreeze = int(num_layers * thaw_ratio)
        # Print total layers and unfrozen layers
        print("now epoch:{} -------- total encoder layers：{}, now unfreeze encoder layers: {}， thaw_ratio:{}".format(epoch, num_layers, layers_to_unfreeze, thaw_ratio))
        # Unfreezing strategy: start from the last layer
        for i in range(num_layers - layers_to_unfreeze, num_layers):
            for name, param in target_model.encoder.named_parameters():
                if f'weight_hh_l{i}' in name or f'weight_ih_l{i}' in name or f'bias_hh_l{i}' in name or f'bias_ih_l{i}' in name:
                    param.requires_grad = True
                    print("unfreeze encoder layer:{}".format(name))


# Train target city model (combining distillation and progressive fine-tuning)
def train_with_distillation(
        source_model, target_model, target_train_loader, target_test_loader,csv_file_name,
        batch_size,
        num_epochs=50, learning_rate=0.001,
        lambda_distill=0.5, lambda_reg=0.2,
        freeze_ratio=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
):
    source_model = source_model.to(device)
    target_model = target_model.to(device)

    # Optimizer (initially only train classifier, adjust later based on unfreezing strategy)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, target_model.parameters()),
        lr=learning_rate
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        # Progressive unfreezing
        progressive_unfreeze(target_model, epoch, num_epochs, freeze_ratio)

        # Update optimizer parameters (reset after unfreezing)
        for param_group in optimizer.param_groups:
            param_group['params'] = list(filter(lambda p: p.requires_grad, target_model.parameters()))

        target_model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_distill_loss = 0.0
        total_reg_loss = 0.0

        # Dynamically adjust regularization strength (weaken constraints as training progresses)
        dynamic_lambda_reg = lambda_reg * (1 - epoch / num_epochs)

        for batch_idx, (traj, user_label, lengths, mask) in enumerate(target_train_loader):
            traj = traj.to(device)
            user_label = user_label.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)  # mask data keeps 1 (true), masked data is 0 (false)
            # predict is POI sequence reconstruction prediction, logistic_preds is TUL user localization prediction
            h, h_mse, mu, log_var, logistic_output, logistic_preds, logistic_loss = target_model(encoder_input=traj,
                                                                                       decoder_input=traj,
                                                                                       user_label=user_label,
                                                                                       lengths=lengths,
                                                                                       mask=mask)

            # 1. Target model forward pass, get logistic classifier output and encoder output
            target_logits = logistic_output
            target_features = h

            # 2. Source model forward pass (no gradients)
            with torch.no_grad():
                source_features, _, _, _, _, _, _ = source_model(encoder_input=traj,
                                                     decoder_input=traj,
                                                     user_label=user_label,
                                                     lengths=lengths,
                                                     mask=mask)

            # 3. Calculate losses
            # 3.1 Standard classification loss
            ce_loss = F.cross_entropy(target_logits, user_label)

            loss = ce_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()

        # Learning rate scheduling
        scheduler.step()

        # Print training information
        avg_loss = total_loss / len(target_train_loader)
        print(f"----------!!epoch:{epoch}!!-----------------")
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f}, CE Loss: {total_ce_loss / len(target_train_loader):.4f}")
        print(
            f"  Distill Loss: {total_distill_loss / len(target_train_loader):.4f}, Reg Loss: {total_reg_loss / len(target_train_loader):.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(
            f"  Frozen Layers: {sum(1 for param in target_model.encoder.parameters() if not param.requires_grad)}/{len(list(target_model.encoder.parameters()))}")

        # Test set evaluation
        # Evaluate once per epoch
        # 6. Evaluate target model
        target_model.eval()
        # Overall test set accuracy and loss
        correct = 0
        loss_a = 0
        correct5 = 0
        test_data_len = len(target_test_loader.dataset)
        all_preds = []
        all_labels = []

        for i, (traj, user_label, lengths, mask) in enumerate(target_test_loader):
            traj = traj.to(device)
            user_label = user_label.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            h, h_mse, mu, log_var, logistic_output, logistic_preds, logistic_loss = target_model(encoder_input=traj,
                                                                                       decoder_input=traj,
                                                                                       user_label=user_label,
                                                                                       lengths=lengths,
                                                                                       mask=mask)
            # loss
            loss = logistic_loss + h_mse

            # Save predictions and true labels for entire epoch
            all_preds.extend(logistic_preds.cpu().tolist())
            all_labels.extend(user_label.cpu().tolist())
            # Calculate top-1 classification accuracy
            correct += int((logistic_preds == user_label).sum())
            # Accumulate average loss for each batch in epoch
            loss_a += loss
            top_k = 5
            out_np = logistic_output.cpu().detach().numpy()
            for index, o in enumerate(out_np):
                top5 = o.argsort()[::-1][:top_k]
                if int(user_label[index]) in top5:
                    correct5 = correct5 + 1

        # Calculate macro metrics for entire test set
        # top1, top5 accuracy
        acc1 = correct / test_data_len
        acc5 = correct5 / test_data_len
        # Precision, recall and F1 score
        macro_f, macro_p, macro_r = calculate_macro_metrics(all_preds, all_labels)

        print(f"----------!!epoch:{epoch}-TEST!!-----------------")
        print(f"learning_rate: {optimizer.param_groups[0]['lr']:.4f}")
        print(f"epoch_total_sum_loss_is: {loss_a / (test_data_len // batch_size):.4f}")
        print('---logistic_acc1', acc1, '---logistic_acc5', acc5)
        print(f"logistic_Macro-F1: {macro_f:.4f}, logistic_Precision: {macro_p:.4f}, logistic_Recall: {macro_r:.4f}")

        save_training_info_to_csv(csv_file_name, epoch, optimizer.param_groups[0]['lr'], loss_a / test_data_len, acc1,
                                  acc5, macro_f, macro_p, macro_r)

        # 7. Save target model
        torch.save(target_model.state_dict(), 'city_B_model_with_distillation.pth')

    return target_model