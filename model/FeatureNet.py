import torch
import torch.nn as nn

'''
The EEG Feature Extractor Network
'''

class FeatureNet(nn.Module):
    def __init__(self, channels=10, time_second=30, freq=100):
        super(FeatureNet, self).__init__()
        activation = nn.ReLU
        padding = 'same'
        self.feature_small = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=50, stride=6, padding=0),
            nn.BatchNorm1d(32),
            activation(),
            nn.MaxPool1d(kernel_size=16, stride=16),
            nn.Dropout(0.5),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=padding),
            nn.BatchNorm1d(64),
            activation(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=padding),
            nn.BatchNorm1d(64),
            activation(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=padding),
            nn.BatchNorm1d(64),
            activation(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Flatten()
        )
        self.feature_large = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=0),
            nn.BatchNorm1d(64),
            activation(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, kernel_size=6, stride=1, padding=padding),
            nn.BatchNorm1d(64),
            activation(),
            nn.Conv1d(64, 64, kernel_size=6, stride=1, padding=padding),
            nn.BatchNorm1d(64),
            activation(),
            nn.Conv1d(64, 64, kernel_size=6, stride=1, padding=padding),
            nn.BatchNorm1d(64),
            activation(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channels*256, 64),
            nn.Linear(64, 5)
        )
            
    def get_feature(self, x):
        # the feature before the classifier
        B, C, T = x.shape
        x_ch = x.reshape(B * C, 1, T)
        x_s = self.feature_small(x_ch)
        x_l = self.feature_large(x_ch)
        x_f = torch.cat((x_s, x_l), dim=1)
        x_f = x_f.reshape(B, C, -1)
        return x_f
    
    def forward(self, x):
        B, C, T = x.shape
        x_ch = x.reshape(B * C, 1, T)
        x_s = self.feature_small(x_ch)
        x_l = self.feature_large(x_ch)
        x_f = torch.cat((x_s, x_l), dim=1)
        x_f = x_f.reshape(B, -1)
        y_hat = self.classifier(x_f)
        return y_hat

