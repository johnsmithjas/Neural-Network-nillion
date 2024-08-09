# Neural Network Inference with Nada

This project demonstrates how to perform neural network inference using the Nada framework. The program leverages the Nada NumPy wrapper to create secure multi-party computations and runs inference on a neural network model.

## Prerequisites

- Python 3.x
- nada_numpy library
- config module with DIM configuration
- my_nn module with MyNN class
- nada_dsl library

## Installation

- Clone the repository:

```bash
    git clone https://github.com/your-repo/neural-net-inference.git
cd neural-net-inference
```

- Install the required libraries:

  ```bash
  pip install nada_numpy nada_dsl
  ```

- Ensure the config and my_nn modules are available in your Python path.

## Usage

The main script for running the inference is neural_net.py. The nada_main function performs the following steps:

1. Create Parties: Uses the Nada NumPy wrapper to create two parties, "Party0" and "Party1".
2. Instantiate Model: Creates an instance of the MyNN model.
3. Load Model Weights: Loads the model weights from the Nillion network using "Party0".
4. Load Input Data: Loads the input data for inference, provided by "Party1".
5. Compute Inference: Runs the inference on the input data using the model.
6. Produce Output: Produces the output for "Party1" with the variable name "my_output".

## Example

To run the program, execute the following command:

```bash
python neural_net.py
```

## Function Details

nada_main

````bash
def nada_main() -> List[Output]:
    """
    Main Nada program.

    Returns:
        List[Output]: Program outputs.
    """
    # Step 1: We use Nada NumPy wrapper to create "Party0" and "Party1"
    parties = na.parties(2)

    # Step 2: Instantiate model object
    my_model = MyNN()

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    my_model.load_state_from_network("my_nn", parties[0], na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array([3], parties[1], "my_input", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model.forward(...)`
    result = my_model(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(parties[1], "my_output")
 ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Nada framework for secure multi-party computation.
Nillion network for model weight storage and retrieval.
````
