import torch.nn as nn
import torch.nn.functional as F
import torch


class PatientClassificationNet(nn.Module):
    def __init__(self, input_dim, hidden_size, activation=torch.relu):
        super(PatientClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class PatientGroupNet(nn.Module):
    def __init__(self, patient_class_net, class_boundary):
        super(PatientGroupNet, self).__init__()
        self.patient_class_net = patient_class_net
        self.patient_class_net.eval()
        self.class_boundary = class_boundary
        self.num_class = self.class_boundary.size(0)
        self.eval()

    def forward(self, x):
        x = self.patient_class_net(x)
        comp = x < self.class_boundary
        return self.num_class - comp.sum(-1) + 1
