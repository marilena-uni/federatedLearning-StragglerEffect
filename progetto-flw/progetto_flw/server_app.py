"""progetto-flw: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
#from progetto_flw.fedavg_SOGLIA import FedAvg
#from flwr.server.strategy import FedAvg
#from progetto_flw.fedavg_dinamico import FedAvg
#from progetto_flw.fedavg_timeout import FedAvg
from progetto_flw.fedavgTIMEclust import FedAvg
#from progetto_flw.fedavg_nosoglia import FedAvg
from progetto_flw.task import Net, get_weights

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    weighted_average_metrics = {
        " Accuracy ": sum( accuracies ) / sum( examples ) ,
    }
    # Aggregate and return custom metric (weighted average)
    # return {"accuracy": sum(accuracies) / sum(examples)}
    return weighted_average_metrics



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
