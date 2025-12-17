# [Knowledge-Based Systems]FedRDA
FedRDA: Federated Learning with Representation Decoupling and Divergence-Aware Aggregation
This is the official implementation of our paper [*FedRDA: Federated Learning with Representation Decoupling and Divergence-Aware Aggregation*](https://www.sciencedirect.com/science/article/pii/S0950705125021227) (accepted by Knowledge-Based Systems 2025). 
## Examples for **MNIST** in the ***label skew*** scenario
```bash
cd ./dataset
# Please modify train_ratio and alpha in dataset\utils\dataset_utils.py

python generate_MNIST.py iid - - # for iid and unbalanced scenario
python generate_MNIST.py iid balance - # for iid and balanced scenario
python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario
python generate_MNIST.py noniid - exdir # for Extended Dirichlet strategy 
```
## Run
- Run evaluation: 
    ```bash
    cd ./system
    python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0 # using the MNIST dataset, the FedAvg algorithm, and the 4-layer CNN model
    python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0,1,2,3 # running on multiple GPUs
    ```
