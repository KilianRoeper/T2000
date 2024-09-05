import torch
import lightning as L
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import LSTM
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from math import sqrt, ceil
from lightning.pytorch.tuner import Tuner
from veh_spd_LSTM.vsm_data_preparation import preprocess_data, load_config


#from lightning.pytorch.loggers import TensorBoardLogger
#from pytorch_lightning.loggers import CSVLogger

#TUNING_PARAMETERS
initial_batch_size = 32
n_layers = 2
output_size = 1
hidden_size = 50
n_features = 6
n_epochs = 300
log_n_steps = 2
stopping_patience = 3
dropout_prob = 0.02
initial_learning_rate = 0.00005
step_size_batch_increment = 16
increase_batch_every_n_epochs = 1
max_batch_size = 160



torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Lade die CSV-Datei
config = load_config('veh_spd_LSTM/lib/vm_config.yaml')
data = pd.read_csv('veh_spd_LSTM/prepared_data.csv')
X, y = preprocess_data(data, config)



# standardise Features
scaler = StandardScaler()

#reshape to 2D, scale and reshape back to 3D
X_reshaped = X.reshape(-1, X.shape[-1]) 
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# convert to PyTorch-tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # Zielvariable in Form von (N, 1)
#print(X_tensor.shape)
#print(y_tensor.shape)

# DataLoader
dataset = TensorDataset(X_tensor, y_tensor)

#not needed -> lightning: train_dataloader()
#dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=31)    


testX = torch.tensor(X_test, dtype=torch.float32)
testY = torch.tensor(y_test, dtype=torch.float32)

test_dataset = TensorDataset(testX, testY)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



#learning_rate_finder = FineTuneLearningRateFinder(milestones=(1, 2, 5, 10))


# initialise the model and move it to the GPU
model = LSTM(dataset, 
             input_size=n_features, 
             hidden_size=hidden_size, 
             output_size=output_size, 
             num_layers=n_layers, 
             initial_batch_size=initial_batch_size, 
             dropout_prob=dropout_prob, 
             learning_rate=initial_learning_rate, 
             step_size=step_size_batch_increment,
             increase_batch_every_n_epochs=increase_batch_every_n_epochs,
             max_batch_size=max_batch_size)
model = model.to(device)  # move model to GPU

early_stop_callback = EarlyStopping(
    monitor='val_loss', 
    patience=stopping_patience,        
    verbose=True
)
# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  
    filename='best-checkpoint',  
    save_top_k=1,  # only save best model
    mode='min'  # only save while loss is minimized
)
#tb_logger = TensorBoardLogger('logs', name='Vehicle_Speed_Predictions')
#csv_logger = CSVLogger('logs', name='vehicle_speed_predictions')

# train the model
trainer = L.Trainer(
    max_epochs=n_epochs, 
    #log_every_n_steps=log_n_steps, 
    callbacks=[early_stop_callback, checkpoint_callback],
    reload_dataloaders_every_n_epochs=1
    #logger=csv_logger
    )

#tuner = Tuner(trainer)
#lr_finder = tuner.lr_find(model)
#new_lr = lr_finder.suggestion()
#model.hparams.lr = new_lr

trainer.fit(model) #, train_dataloaders=dataloader)

#trainer.save_checkpoint("model_checkpoint.ckpt")
#torch.save(model.state_dict(), "model.pth")


##TESTING

# Konvertiere den Testdatensatz in PyTorch-Tensoren
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)  # Zielvariable in Form von (N, 1)

# Erstelle DataLoader für den Testdatensatz
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Shuffle=False für den Testdatensatz


writer = SummaryWriter('runs/vehicle_speed_prediction')

#Initialisiere Variablen für die Berechnung des Testverlusts und die Speicherung der Vorhersagen
total_loss = 0.0
num_batches = 0
all_predictions = []
all_actuals = []
n_pred = 10

# Deaktiviere das Gradient-Tracking für die Evaluation
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        #inputs, labels = inputs.to(device), labels.to(device)
        
        # Berechne die Modellvorhersagen
        outputs = model(inputs)

        
        # Berechne den Verlust
        loss = F.mse_loss(outputs, labels)
        total_loss += loss.item()
        num_batches += 1
        
        # Speichere die Vorhersagen und die tatsächlichen Werte
        all_predictions.append(outputs.cpu())
        all_actuals.append(labels.cpu())

# Berechne den durchschnittlichen Testverlust

#predictions = []
#for _ in range (n_pred):
#    outputs = model(inputs)
#    predictions.append(outputs)
#    inputs = outputs

average_loss = total_loss / num_batches
print(f'Average Test Loss: {average_loss}')

# Konvertiere die Listen in Tensoren
all_predictions = torch.cat(all_predictions)
all_actuals = torch.cat(all_actuals)

# Plotten der tatsächlichen und vorhergesagten Werte in TensorBoard
for i in range(len(all_actuals)):
    writer.add_scalars('Vehicle Speed Prediction',
                       {'Actual': all_actuals[i].item(),
                        'Predicted': all_predictions[i].item()},
                       i)

writer.close()
