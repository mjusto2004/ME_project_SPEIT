
from simulation import run_simulation
import numpy as np
import random as rd
import datetime
from pathlib import Path


def create_dataset(N):
    '''Creates a dataset of length N'''
    D = np.zeros((N, 563))

    for i in range(N):
        P0 = 30000
        T0 = 293
        h0 = 300

        a, b, c = rd.uniform(0, 1), rd.uniform(0, 1), rd.uniform(0, 1)
        T, P, h = a*T0/2 + T0/2, b*P0/2 + P0/2, c*h0/2 + h0/2

        simulation = run_simulation(T, P, h)
        D[i] = np.concatenate((np.array([T, P, h]), simulation), axis=0)

    return D


def save_dataset(N):
    '''Creates a dataset and saves it in the data folder'''

    dataset = create_dataset(N)

    # Directory where data_generation.py is located
    folder = Path(__file__).resolve().parent / "datasets"

    # Create the directory if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename using the current date and time
    x = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    y = str(N)
    name = "dataset_" + x + y + ".csv"

    # Full file path
    path = folder / name

    # Save the dataset
    np.savetxt(path, dataset, delimiter=",")

    print(f"Dataset saved in: {path}")


if __name__ == "__main__":
    save_dataset(90)

