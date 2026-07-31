import flwr as fl
import torch
from collections import OrderedDict
from model import VehicularGRU_IDS

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
       
        if aggregated_parameters is not None:
            print(f"Saving global weights from aggregation round {server_round}...")
            # Convert Flower parameters back to NumPy arrays
            _parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
           
            # Load weights into a temporary network instance to save state dict natively
            net = VehicularGRU_IDS()
            params_dict = zip(net.state_dict().keys(), _parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
           
            # Save the updated model weights file explicitly
            torch.save(net.state_dict(), "global_road_ids_model.pth")
            print("Successfully serialized 'global_road_ids_model.pth'")
           
        return aggregated_parameters, aggregated_metrics

if __name__ == "__main__":
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,      # We will run 2 simulated vehicle clients
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
