# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""

from logging import WARNING, INFO  # Ho aggiunto INFO
from typing import Callable, Optional, Union


from sklearn.cluster import KMeans
import numpy as np

#per scrivere sul file i risultati
import csv
from pathlib import Path

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
import logging

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy
import time

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class FedAvg(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.client_start_times = {}
        self.response_times_history = []
        self.delayed_updates = [] 

       

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            # FIX: Corrette le chiamate alla funzione log
            log(INFO, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
            log(INFO, "Questo è un messaggio INFO visibile")
            log(WARNING, "Questo è un messaggio WARNING")

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        print(f"[SERVER] Round {server_round} evaluation:")
        print(f"  Loss: {loss:.4f}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        now = time.time()
        log(INFO, "now ora vale %f", now)
        for client in clients:
            self.client_start_times[client.cid] = now
            log(INFO, "client.cid %s ha un tempo di start registrato di  %f", client.cid, self.client_start_times[client.cid])
            
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        now = time.time()
        

        #MAX_TIME= 15.0

        #log(INFO, f"Aggregate fit called for round {server_round}")
        log(INFO, f"Number of results: {len(results)}, failures: {len(failures)}")
        
        if not results:
            log(INFO, " No results received.")
            return None, {}
        if not self.accept_failures and failures:
            log(INFO, " Failures present and not accepted.")
            return None, {}

        current_round_times = []
        current_round_times_filtered = []
        filtered_results = []


        Path("logs_soglia").mkdir(exist_ok=True)

        #ROUND 1 e 2 accettiamo tutti i client e sommiamo il tempo che questi ci mettono
        if server_round <= 2:
            log(INFO, f"[ROUND {server_round}] Fase di inizializzazione. Nessun limite di tempo applicato.")
            
            for client, fit_res in results:
                client_id = client.cid
                client_elapsed_time = fit_res.metrics.get("elapsed_time")

                if client_elapsed_time is not None:
                    current_round_times.append(client_elapsed_time)
                    #scrivo sul file csv i tempi
                    with open("logs_soglia/response_times_soglia.csv", "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([server_round, client_id, client_elapsed_time])
                
                else:  #se non ci sono valori
                    log(WARNING, f"Nessun client ha inviato la metrica 'elapsed_time' nel round {server_round}.")
                    #return None, {}
                   
            
            filtered_results = results


        #ROUND 3 , adesso ogni client deve rientrare nel tempo TOT altrimenti non sarà considerato    
        else:
            average_time = sum(self.response_times_history) / len(self.response_times_history)
            MAX_TIME = max(2, min(average_time * 1.2, 60))  # un buffer del 20% e puo essere max 60sec e min 5sec
            log(INFO, f"[ROUND {server_round}] Limite di tempo dinamico impostato a {MAX_TIME:.2f}s (media storica).")
            
            for client, fit_res in results:
                client_id = client.cid
                client_elapsed_time = fit_res.metrics.get("elapsed_time")
                if client_elapsed_time is not None:
                    current_round_times_filtered.append(client_elapsed_time)
                    log(INFO, f"[TEMPO IMPIEGATO: ROUND {server_round}] Client {client_id} (dal client) ha impiegato {client_elapsed_time:.2f} secondi per il fit.")
                    
                    #1 se è stato accettato, 0 se non lo è stato perche ha superato il tempo max
                    accepted = 1 if client_elapsed_time <= MAX_TIME else 0
                    with open("logs_soglia/response_times_soglia.csv", "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([server_round, client_id, client_elapsed_time, accepted])
                   

                    if client_elapsed_time <= MAX_TIME:
                        filtered_results.append((client, fit_res))
                    else:
                        log(WARNING, f"[ROUND {server_round}] Client {client_id} ESCLUSO. Il tempo di {client_elapsed_time:.2f}s ha superato il limite di {MAX_TIME}s.")
                else:
                    log(INFO, f"[ ROUND {server_round}] Client {client_id} non ha inviato la metrica 'elapsed_time'.")
            
            if current_round_times_filtered:
                self.response_times_history = current_round_times_filtered


        #aggiorno l'array dei tempi, aggiungendo anche quello dell'ultimo round
        self.response_times_history.extend(current_round_times)

        
        #se tutti i clienti sono esclusi
        if not filtered_results:
            log(WARNING, f"Nessun client idoneo per l'aggregazione nel round {server_round}.")
            return None, {}
        
        filtered_results += self.delayed_updates
        self.delayed_updates = []

        if self.inplace:
            aggregated_ndarrays = aggregate_inplace(filtered_results)
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in filtered_results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in filtered_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated