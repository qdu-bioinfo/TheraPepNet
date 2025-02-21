import torch
import torch.nn as nn
from LocalNet import TemporalConvNet
from AminoNet import AminoNet
from GlobalNet import GlobalNet
from CFFT import CFFT
from Local import Local
from ProtBERT import ProtBERT

class peptideClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, timesteps, num_output, dropout, nhead, num_layers):
        super(peptideClassifier, self).__init__()
        self.encoder1 = GlobalNet1(128, nhead, num_layers)
        self.encoder2 = GlobalNet2(528, nhead, num_layers)
        self.encoder3 = GlobalNet3(d_model, nhead, num_layers)
        self.aaindex = AminoNet(timesteps)
        self.CFFT = CFFT()
        self.tcn = TemporalConvNet(128, num_channels=[8, 16, 32, 64, 128], kernel_size=3, dropout=0.2)
        self.protBERT = ProtBERT()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_output)
        self.local_fc = nn.Linear(self.Local.shape[1],d_model)
        self.gate_tcn1 = nn.Linear(128, 128) 
        self.gate_tcn2 = nn.Linear(128, 128)
        self.gate_tcn3 = nn.Linear(128, 128)
        self.gate_trans1 = nn.Linear(d_model, d_model) 
        self.gate_trans2 = nn.Linear(d_model, d_model) 
        self.gate_trans3 = nn.Linear(d_model, d_model) 

        self.feature_fusion_mlp = nn.Sequential(
            nn.Linear(128 * 6, 256),  
            nn.ReLU(),
            nn.Linear(256, 128)  
        )

    def forward(self, input, batch_indices):
        device = input.device
        AAindex_vector = self.aaindex(input)  
        CFFT_vector = self.CFFT
        CFFT_batch = CFFT_vector[batch_indices]
        CFFT_vector = CFFT_batch.clone().detach().to(device).float() 
        Prot_BERT_vector = self.protBERT

        AAindex_vector_TCN = AAindex_vector.permute(0, 2, 1) 
        CFFT_vector_TCN = CFFT_vector.permute(0, 2, 1) 
        Prot_BERT_vector_TCN = Prot_BERT_vector.permute(0, 2, 1) 

        output_tcn1 = self.tcn1(Prot_BERT_vector_TCN)  
        output_tcn2 = self.tcn2(AAindex_vector_TCN) 
        output_tcn3 = self.tcn3(CFFT_vector_TCN)  

        output_tcn1 = output_tcn1[:, :, -1] 
        output_tcn2 = output_tcn2[:, :, -1]  
        output_tcn3 = output_tcn3[:, :, -1]  

        output_trans1 = self.encoder1(Prot_BERT_vector)  
        output_trans2 = self.encoder2(AAindex_vector) 
        output_trans3 = self.encoder3(CFFT_vector) 

        gate_tcn1 = torch.sigmoid(self.gate_tcn1(output_tcn1)) 
        gate_tcn2 = torch.sigmoid(self.gate_tcn2(output_tcn2)) 
        gate_tcn3 = torch.sigmoid(self.gate_tcn3(output_tcn3))  
        gate_trans1 = torch.sigmoid(self.gate_trans1(output_trans1))  
        gate_trans2 = torch.sigmoid(self.gate_trans2(output_trans2)) 
        gate_trans3 = torch.sigmoid(self.gate_trans3(output_trans3)) 

        output_tcn1 = gate_tcn1 * output_tcn1
        output_tcn2 = gate_tcn2 * output_tcn2
        output_tcn3 = gate_tcn3 * output_tcn3
        output_trans1 = gate_trans1 * output_trans1
        output_trans2 = gate_trans2 * output_trans2
        output_trans3 = gate_trans3 * output_trans3

        combined = torch.cat((output_tcn1, output_tcn2, output_tcn3, output_trans1,output_trans2, output_trans3), dim=1)
        fused_features = self.feature_fusion_mlp(combined)
        output = self.fc(combined)
        return combined, output
