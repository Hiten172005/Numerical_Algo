# Federated Reinforcement Learning with LSTD

This project implements a federated reinforcement learning approach using Least-Squares Temporal Difference (LSTD) learning for healthcare applications. The implementation is done in phases to demonstrate the concept, enhance with real data, and optimize performance.

## Project Structure

```
├── Fed_LSTD.ipynb         # Main Jupyter notebook with federated LSTD concept
├── frl.py                 # Real medical environment and MPI-based federated learning implementation
├── loader.py              # Data preprocessing for MIMIC-III dataset
├── lspi_code.py           # Core LSTD implementation
├── lspi_parallel.py       # Parallel implementation using multiprocessing for performance optimization
├── hospital_data/         # Processed hospital data
│   ├── hospital_6.csv
│   ├── hospital_7.csv
│   ├── hospital_8.csv
│   ├── hospital_9.csv
│   └── metadata.json
└── mimic-iii-clinical-database-demo-1.4/  # MIMIC-III demo dataset
```

## Project Phases

### Phase 1: Concept Implementation
- Initial implementation of federated LSTD in a Jupyter notebook (`Fed_LSTD.ipynb`)
- Simple synthetic healthcare environment
- Demonstration of the concept with basic federated averaging

### Phase 2: Real Data Integration
- Incorporation of the MIMIC-III clinical database demo
- Data preprocessing with `loader.py` to extract meaningful patient state attributes
- Creation of hospital-specific datasets to simulate a federated learning scenario
- Implementation of a more realistic medical environment in `frl.py`

### Phase 3: Parallelization and Performance Optimization
- Addition of multiprocessing in `lspi_parallel.py` for improved performance
- Performance timing to measure execution time and efficiency
- Comparison between federated and centralized approaches
- Enhanced policy evaluation with Q-function estimation

## How to Run the Code

### Prerequisites
- Python 3.7+
- Required packages: numpy, pandas, multiprocessing, matplotlib, mpi4py
- MIMIC-III demo dataset (already included in the repository)

### Installation

```bash
# Install required packages
pip install numpy pandas matplotlib mpi4py
```

### Data Preprocessing

```bash
# Process the MIMIC-III data and generate hospital-specific files
python loader.py
```

This will:
1. Load data from the MIMIC-III demo dataset
2. Process patient records to extract meaningful attributes
3. Generate hospital-specific CSV files in the `hospital_data` directory

### Running the Experiments

#### Jupyter Notebook (Phase 1)
Open and run the Jupyter notebook to see the basic concept:

```bash
jupyter notebook Fed_LSTD.ipynb
```

#### Federated Learning with Real Data (Phase 2)

```bash
python frl.py
```

#### Parallel Execution (Phase 3)

```bash
# Run the multiprocessing-based parallel implementation
python lspi_parallel.py
```

## Key Components

### Medical Environment
- Simulates patient states and transitions based on real data
- Calculates rewards based on patient outcomes, treatment decisions, and costs
- Supports both deterministic and stochastic policies

### LSTD Implementation
- Core LSTD algorithm that estimates the value function from samples
- Feature representation for patient states
- Policy evaluation and improvement

### Data Loader (`loader.py`)
- Extracts patient data from the MIMIC-III database
- Calculates clinically relevant features:
  - Diagnosis severity
  - Lab value abnormalities
  - Treatment intensity
- Splits data into hospital-specific files to simulate federated scenarios

### Parallelization
- `lspi_parallel.py`: Uses Python's multiprocessing library for parallel execution across hospitals
- `frl.py`: Uses MPI for distributed execution across multiple processes
- Both implement federated averaging for model aggregation
- Timing measurements included to compare performance gains

## Results

The project demonstrates:
1. How federated learning can be applied to healthcare reinforcement learning
2. Performance comparison between federated and centralized approaches
3. Efficiency gains through parallelization
4. Policy evaluation on real medical data