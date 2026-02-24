# FederatedLearning-StragglerEffect
sviluppo di un algoritmo alternativo al FedAvg in grado di mitigare lo Straggler Effect

#Straggler Effect
ritardi dei client più lenti nel completare il training, che rallentano l’intero processo

#Obiettivo:
Progettare un algoritmo alternativo al FedAvg per risolvere il problema dello "Straggler Effect" migliorando i tempi di esecuzione, senza compromettere l’accuratezza del modello, utilizzando il framework Flower e analizzando le performance con grafici e metriche

## Usage

Per runnare lo script

```
 run_simulation (
 server_app =server ,
 client_app =client ,
 num_supernodes = NUM_CLIENTS ,
 backend_config = backend_config ,
 )
```


