# Dynamic Rank Adjustment Training Framework
This repository contains the S/W framework of 'Dynamic Rank Adjustment Training' method.
## Software Requirements
To install the required packages for this project, you can use the `requirements.txt` file.  Follow these steps:

* **1. Make sure you have Python and pip installed on your system.**
* **2. Clone this repository:**

   ```bash
   git clone https://github.com/HyunTakShin/DynamicRankSW
* **3. Install the required packages:**
    ```bash
   pip3 install -r requirements.txt

## Usage
### Training
   1. Set hyper-parameters in `config.py`
      * Hyper-parameter like batch_size, learning rate, rank adjustment, weight decay
      * For other dataset:
      
      ```python
      # Choose one of the following dataset:
      dataset = "cifar10"        # For SVD decomposition
      dataset = "cifar100"       # For CP decomposition  
   2. Check decomposition method in `main.py`
      * In default method is set to "SVD"
      * For other decomposition method edit the `method` variable in `main.py`:
      
      ```python
      # Choose one of the following methods:
      method = "SVD"     # For SVD decomposition
      method = "CP"      # For CP decomposition  
      method = "Tucker"  # For Tucker decomposition
      ```
   4. Run main.py using command
      ```python
      python3 main.py
      ```
### Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
 1. `loss.txt`: An output file that contains the training loss of auxiliary model for every epoch.
 2. `acc.txt`: An output file that contains the validation accuracy of auxiliary model for every epoch.
 5. `./checkpoint`: The checkpoint files generated after every epoch. This directory is created only when `checkpoint` is set to 1 in `config.py`.
## Questions / Comments
 * Hyuntak Shin (hyuntakshin@inha.edu)
