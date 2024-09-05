import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam#, SGD
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from math import sqrt, ceil


#from lightning.pytorch.callbacks import EarlyStopping



class LSTM(L.LightningModule):

    def __init__(self, 
                 dataset,
                 input_size=6,
                 hidden_size=50,
                 output_size=1, 
                 num_layers=2, 
                 initial_batch_size=64, 
                 dropout_prob=0.3, 
                 learning_rate=0.001, 
                 step_size=16, 
                 max_batch_size=160, 
                 increase_batch_every_n_epochs=4):
        super().__init__()
        #input-size: number of features; hidden-size: num outputs (> 1 if fed into another network with multiple inputs)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

        self.initial_batch_size = initial_batch_size
        self.batch_size = initial_batch_size
        self.batch_size_by_input = 0
        self.save_hyperparameters()
        self.dataset = dataset
        self.lr = learning_rate
        self.step_size = step_size
        self.max_batch_size = max_batch_size
        self.increase_batch_every_n_epochs = increase_batch_every_n_epochs

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        if(len(lstm_out.shape) == 3):
            prediction = self.fc(lstm_out[:, -1, :])  # Verwende die letzte Ausgabe der LSTM-Schicht
        else:
            prediction = self.fc(lstm_out[:, :])
        return prediction
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        #SGD(momentum=)
        # Beispiel für einen Lernraten-Scheduler, der die Lernrate reduziert, wenn der Validierungsverlust nicht besser wird.
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6, verbose=True)

        # Scheduler und Optimizer müssen als Liste zurückgegeben werden
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # der Wert, den der Scheduler überwacht
                'interval': 'epoch',  # die Frequenz der Anpassung (z.B. pro Epoche)
                'frequency': 1  # wie oft der Scheduler aufgerufen wird
            }
        }
    

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        self.loss = F.mse_loss(output_i, label_i)

        #for i in range(label_i.size(0)):
        #    label_i_val = label_i[i, 0] 
        #    output_i_val = output_i[i, 0]
        #    self.log('val_actuals', label_i_val)
        #    self.log('val_predictions', output_i_val)
        if (batch_idx == 1):
            batch_size_by_input = input_i.size(0)
            self.batch_size_by_input = batch_size_by_input
            self.log('batch_size_byInput_size', batch_size_by_input, prog_bar=True)
        used_memory = torch.cuda.memory_allocated()
        self.log('gpu_memory_usage', used_memory / 1024 ** 2, prog_bar=True)  # In MB
        if(batch_idx % 1000 == 0):
            self.log('val_loss', self.loss)  

        return self.loss
    

    def on_train_epoch_end(self):
        # Durchschnittlichen Verlust über alle Schritte in der Epoche berechnen
        #avg_loss = torch.stack([x['loss'] for x in self.outputs]).mean()
        #self.log('epoch_train_loss', avg_loss, prog_bar=True)
        
        # Loggen der Lernrate des Optimizers
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]
        self.log('learning_rate', current_lr, prog_bar=True)


    def train_dataloader(self):
        # Anpassung der batch_size während des Trainings
        epoch = self.current_epoch
        #batch_size = self.initial_batch_size * ceil((sqrt(epoch/2) + 1))  # Einfaches Beispiel: batch_size wird mit jeder Epoche verdoppelt

        self.batch_size = min(self.initial_batch_size + self.step_size * (epoch // self.increase_batch_every_n_epochs), self.max_batch_size)

        #self.batch_size = batch_size
        #print(self.batch_size)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=31)
        return dataloader

    #def configure_callbacks(self):
     #   return [EarlyStopping(monitor='val_loss', patience=10)]





### tuning parameters:

    # early stopping (overfitting),
    # learning rate (scheduler), 
    # batch value, 
    # extra layer (drop out - avoid overfitting), 
    # (increasing the layer - observe how bahaviour changes) 
    # saving model, while loss goes down (best model saved), 
    ## (another gradient function), 
    ## saving graphs (epoch vs loss, actual vs predicted), 
    ## changing scaler (min, max)
