from Dataset import get_data
from train import train
from peptideClassifier import peptideClassifier
from torch.utils.data import DataLoader, TensorDataset,  Subset
from sklearn.model_selection import train_test_split
from CustomDataset import CustomDataset
import torch

amino_acid_to_index = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

random_seed = 43
torch.manual_seed(random_seed)

batch_size = 16
num_epochs = 10
d_model = 128
nhead = 8
num_layers = 2
num_output = 3
vocab_size = 20
dropout = 0.2


def encode_sequence(sequence):
    return [amino_acid_to_index[aa] for aa in sequence]


def main():
    peptide_seq_dict, label = get_data()
    peptide_seq_list = list(peptide_seq_dict.values())
    
    numerical_vector = [encode_sequence(seq) for seq in peptide_seq_list]

    max_length = max(len(seq) for seq in numerical_vector)
    numerical_vector_padding = [vector + [0] * (max_length - len(vector)) for vector in numerical_vector]

    Input = torch.tensor(numerical_vector_padding)
    label = torch.tensor(label)
    label = label - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(Input, label)
    targets = label.numpy()
    print(targets.min(), targets.max())  
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2, stratify=targets,
        random_state=4  
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training data size: {len(train_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}")

    model = peptideClassifier(vocab_size, d_model, max_length, num_output, dropout, nhead, num_layers)
    model = model.to(device)

    train(model, num_epochs, train_loader, val_loader, device)


if __name__ == '__main__':
    main()
