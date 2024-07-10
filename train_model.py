import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Carregar dados do CSV (sem as colunas de Concurso e Data)
df = pd.read_csv('resultado.csv', usecols=lambda col: col.startswith('bola '))

# Converter para tipo de dado adequado (exemplo: float32)
data = df.values.astype(np.float32)

# Exemplo de modelo simples em PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(15, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 15)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Definir targets com base nos dados do próximo concurso
# Suponha que o próximo sorteio seja o último registro no seu CSV
next_draw = df.iloc[0].values.astype(np.float32)  # Pega os números do próximo sorteio e converte para float32
targets = torch.tensor(next_draw, dtype=torch.float32)

# Verificar dimensões dos dados
print(f"Dimensões de inputs: {data.shape}")
print(f"Dimensões de targets: {targets.shape}")

# Treinamento com os dados do CSV
for epoch in range(1000):
    inputs = torch.tensor(data, dtype=torch.float32)

    optimizer.zero_grad()
    outputs = model(inputs)

    # Verificar dimensões dos outputs
    print(f"Dimensões de outputs: {outputs.shape}")

    loss = criterion(outputs, targets)  # Utilizando os targets definidos
    loss.backward()
    optimizer.step()

# Salvar o modelo treinado
torch.save(model.state_dict(), "model.pth")
