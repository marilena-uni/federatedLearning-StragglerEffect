"""progetto-flw: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from progetto_flw.task import Net, get_weights, load_data, set_weights, test, train
import time
import random
import logging

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, client_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.client_id = client_id

    def fit(self, parameters, config):
        start_time = time.time()
        set_weights(self.net, parameters)

        initial_loss, _ = test(self.net, self.valloader, self.device)


        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        final_loss, _ = test(self.net, self.valloader, self.device)


        delta_loss = initial_loss - final_loss


        
        


        #i clienti pari dormono
        if self.client_id % 2 == 0 :
            logging.warning(f"[PORTA {self.client_id}] Client lento: ritardo di 5 secondi...")
            time.sleep(5)
            logging.info(f"[FIT_DEBUG] RITARDO di 5 secondi APPLICATO.")
       

      
        elapsed = time.time() - start_time
        logging.info(f" fit completato in {elapsed:.2f} secondi")

        
        metrics= {
            "elapsed_time": elapsed,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "delta_loss": delta_loss,
        }

        return get_weights(self.net), len(self.trainloader.dataset),  metrics

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)


"""embeddedexample: A Flower / PyTorch app."""
"""
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from progetto_flw.task import Net, get_weights, load_data, set_weights, test, train
import time
import random
import logging

# Configura il livello di logging all'inizio del file
logging.basicConfig(level=logging.INFO)



# Define Flower Client
class FlowerClient(NumPyClient):
   
        
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        

    def fit(self, parameters, config):
        # Train the model with data of this client.
        start_time = time.time()

        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
       
       
        # AGGIUNGI QUESTI LOG PER IL DEBUG
        logging.info(f"[FIT_DEBUG] Client ID (approx): {self.client_port}. Verifico condizione sleep.")
        logging.info(f"[FIT_DEBUG] self.client_port: {self.client_port}, Tipo: {type(self.client_port)}, Condizione: {self.client_port == 9094 if self.client_port is not None else 'False'}")

        #dormo sarà un valore random tra 0 e 1
        dormo=random.random()
        logging.info(f"[DORMIRE] Dormire {dormo}")

        if dormo >= 0.8:
            logging.warning(f"[PORTA {self.client_port}] Client lento: ritardo di 10 secondi...")
            time.sleep(5)
            logging.info(f"[FIT_DEBUG] Ritardo di 5 secondi APPLICATO.")
        else:
            logging.info(f"[FIT_DEBUG] Condizione di ritardo NON soddisfatta per porta {self.client_port}.")
        
        elapsed = time.time() - start_time
        # Modificato da 'log(INFO, ...)' a 'logging.info(...)'
        #logging.info(f"[PORTA {self.client_port}] fit completato in {elapsed:.2f} secondi")

        return get_weights(self.net), len(self.trainloader.dataset), {"elapsed_time": elapsed}

    def evaluate(self, parameters, config):
        #Evaluate the model on the data this client has.
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {
        "accuracy": accuracy
    }

def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()

# Flower ClientApp
app = ClientApp(client_fn)
"""