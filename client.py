import flwr as fl
import torch
import sys
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from model import VehicularGRU_IDS, preprocess_ciciov_data, train, test

if len(sys.argv) < 3:
    print("Execution Error. Usage format: python3 client.py <csv_path> <label_id>")
    sys.exit(1)

data_path = sys.argv[1]
target_label = int(sys.argv[2]) # 0 for benign, 1 for attack targets

# Run the target data pipeline pointing directly to the selected data path
X, y = preprocess_ciciov_data(data_path, target_label)
dataset = TensorDataset(X, y)
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

net = VehicularGRU_IDS()

class FlowerVehicleClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=2)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, trainloader)
        return float(loss), len(trainloader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerVehicleClient())