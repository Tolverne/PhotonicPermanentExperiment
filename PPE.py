import numpy as np
import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler, Simulator
from itertools import permutations
import matplotlib.pyplot as plt


class PhotonicPermanentExperiment:

    def __init__(self, theta1=0.4, theta2=0.7, theta3=0.5, modes=3):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.modes = modes
        self._build_circuit()

    # -------------------------------------------------------
    # Build physical circuit
    # -------------------------------------------------------

    def _build_circuit(self):

        c = pcvl.Circuit(self.modes)

        c.add(0, comp.BS(self.theta1))
        c.add(1, comp.BS(self.theta2))
        c.add(0, comp.BS(self.theta3))

        self.circuit = c
        self.unitary = np.array(c.compute_unitary())

    # -------------------------------------------------------
    # Draw circuit
    # -------------------------------------------------------

    def draw(self):
        return self.circuit.draw()

    # -------------------------------------------------------
    # Compute permanent (brute force)
    # -------------------------------------------------------

    def permanent(self, A):
        n = A.shape[0]
        total = 0
        for p in permutations(range(n)):
            prod = 1
            for i in range(n):
                prod *= A[i, p[i]]
            total += prod
        return total

    # -------------------------------------------------------
    # Extract 3x3 submatrix (all modes here)
    # -------------------------------------------------------

    def submatrix(self):
        return self.unitary

    # -------------------------------------------------------
    # Compute ideal quantum probability
    # -------------------------------------------------------

    def quantum_probability(self):

        A = self.submatrix()
        perm = self.permanent(A)
        return abs(perm)**2

    # -------------------------------------------------------
    # Compute distinguishable probability
    # -------------------------------------------------------

    def distinguishable_probability(self):

        U = self.unitary
        T = abs(U)**2

        total = 0
        for p in permutations(range(3)):
            prod = 1
            for i in range(3):
                prod *= T[i, p[i]]
            total += prod

        return total

    # -------------------------------------------------------
    # Perceval simulation
    # -------------------------------------------------------

    def simulate(self, eta=1.0, shots=20000):

        noise = pcvl.NoiseModel(indistinguishability=eta, transmittance=1.0)
        proc = pcvl.Processor("SLOS", self.circuit, noise=noise)

        proc.with_input(pcvl.BasicState([1,1,1]))
        proc.min_detected_photons_filter(3)

        sampler = Sampler(proc)
        result = sampler.sample_count(shots)

        probs = {}
        total = sum(result["results"].values())

        for state, count in result["results"].items():
            probs[str(state)] = count / total

        return probs

    # -------------------------------------------------------
    # Sweep indistinguishability
    # -------------------------------------------------------

    def sweep_eta(self, eta_values):

        ideal = self.quantum_probability()
        classical = self.distinguishable_probability()

        results = []

        for eta in eta_values:
            probs = self.simulate(eta)
            target = probs.get(str(pcvl.BasicState([1,1,1])), 0)
            results.append(target)

        return ideal, classical, results
