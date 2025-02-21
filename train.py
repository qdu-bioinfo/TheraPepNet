import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, \
    matthews_corrcoef, precision_recall_curve, auc, average_precision_score
import numpy as np
import matplotlib.style as style

def train(model, num_epochs, train_loader, val_loader, device, save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_accuracy = 0.0 
    total_Accuracy = 0
    val_accuracies = []  
    val_losses = [] 
    
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0  
        
        for sequences, labels, batch_indices in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            combined, outputs = model(sequences, batch_indices)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
        
        model.eval()  
        total_predictions = []
        total_labels = []
        total_Accuracy = 0
        correct = 0
        total = 0
        val_loss = 0 

        all_labels = []
        all_probs = []


        with torch.no_grad():  
            for sequences, labels, batch_indices in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                combined, outputs = model(sequences, batch_indices)
                loss = criterion(outputs, labels)
                val_loss += loss.item()  
                _, predicted = torch.max(outputs, 1)

                total_predictions.extend(predicted.cpu().numpy())  
                total_labels.extend(labels.cpu().numpy())  

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        cm = confusion_matrix(total_labels, total_predictions)  
        accuracy_scores = []

        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP  
            FN = np.sum(cm[i, :]) - TP  
            TN = np.sum(cm) - TP - FP - FN  

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            accuracy_scores.append(accuracy)

        average_accuracy = np.mean(accuracy_scores)

        accuracy = correct / total 
        val_accuracies.append(average_accuracy)  
        val_losses.append(total_loss / len(train_loader)) 
        total_Accuracy += accuracy

        precisions = []

       
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP 
            precision = TP / (TP + FP) 
            precisions.append(precision)

        average_precision = np.mean(precisions)

        recalls = []
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            recall = TP / (TP + FN)  
            recalls.append(recall)

        average_recall = np.mean(recalls)

        f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall)

        specificities = []
        for i in range(cm.shape[0]):
            TN = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            FP = np.sum(cm[:, i]) - cm[i, i]
            specificity = TN / (TN + FP)
            specificities.append(specificity)
            
        mcc_scores = []

        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP  
            FN = np.sum(cm[i, :]) - TP  
            TN = np.sum(cm) - TP - FP - FN 

            numerator = TP * TN - FP * FN
            denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

            if denominator != 0:
                mcc = numerator / denominator
            else:
                mcc = 0  

            mcc_scores.append(mcc)

        average_mcc = np.mean(mcc_scores)


        avg_specificity = sum(specificities)/len(specificities)
        print(f' Accuracy: {average_accuracy:.4f}')
        print(f' Precision: {average_precision:.4f}')
        print(f' Recall: {average_recall:.4f}')
        print(f' F1-score: {f1_score:.4f}')
        print(f' avg_Specificity: {avg_specificity:.4f}')
        print(f' Matthews Correlation Coefficient (MCC): {average_mcc:.4f}')
        
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'best_model.pth')
            print(f"Best model saved with accuracy: {average_accuracy:.4f}")
    
    print(f'Average accuracy: {(total_Accuracy / num_epochs):.4f}')
    print(f'Best accuracy: {best_accuracy:.4f}')

    n_classes = 3  
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        y_true = (all_labels == i).astype(int) 
        y_score = all_probs[:, i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)  

        plt.plot(recall, precision, label=f'Class {i} (AP={auc_score:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for each class')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

