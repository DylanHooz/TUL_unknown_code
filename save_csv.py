import csv
import datetime
import os

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

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write training information to CSV
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

