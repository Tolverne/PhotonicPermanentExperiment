import numpy as np
import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler
from itertools import permutations
import matplotlib.pyplot as plt


class PhotonicExperiment:

    def __init__(self, circuit: pcvl.Circuit, input_state: pcvl.BasicState):
        self.circuit = circuit
        self.input_state = input_state
        self.modes = circuit.m
        self.unitary = np.array(circuit.compute_unitary())

    # -------------------------------------------------------
    # DRAW CIRCUIT (version-safe)
    # -------------------------------------------------------

    def draw(self):
        try:
            return pcvl.pdisplay(self.circuit)
        except Exception:
            print("Diagram display not supported in this environment.")


    # -------------------------------------------------------
    # PERMANENT (brute force, small n only)
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
    # EXTRACT SUBMATRIX FOR GIVEN INPUT/OUTPUT MODES
    # -------------------------------------------------------

    def extract_submatrix(self, input_modes, output_modes):
        A = np.zeros((len(output_modes), len(input_modes)), dtype=complex)
        for i, r in enumerate(output_modes):
            for j, s in enumerate(input_modes):
                A[i, j] = self.unitary[r, s]
        return A

    # -------------------------------------------------------
    # IDEAL QUANTUM PROBABILITY
    # -------------------------------------------------------

    def quantum_probability(self, input_modes, output_modes):
        A = self.extract_submatrix(input_modes, output_modes)
        perm = self.permanent(A)
        return abs(perm)**2

    # -------------------------------------------------------
    # DISTINGUISHABLE PROBABILITY
    # -------------------------------------------------------

    def distinguishable_probability(self, input_modes, output_modes):
        U = self.unitary
        T = abs(U)**2
        total = 0
        for p in permutations(range(len(input_modes))):
            prod = 1
            for i, r in enumerate(output_modes):
                prod *= T[r, input_modes[p[i]]]
            total += prod
        return total

    # -------------------------------------------------------
    # PERCEVAL SAMPLING
    # -------------------------------------------------------

    def simulate(self, eta=1.0, shots=20000):
        noise = pcvl.NoiseModel(indistinguishability=eta, transmittance=1.0)
        proc = pcvl.Processor("SLOS", self.circuit, noise=noise)

        proc.with_input(self.input_state)
        proc.min_detected_photons_filter(sum(self.input_state))

        sampler = Sampler(proc)
        result = sampler.sample_count(shots)

        probs = {}
        total = sum(result["results"].values())

        for state, count in result["results"].items():
            probs[str(state)] = count / total

        return probs

    # -------------------------------------------------------
    # SWEEP INDISTINGUISHABILITY
    # -------------------------------------------------------

    def sweep_eta(self, input_modes, output_modes, eta_values):
        ideal = self.quantum_probability(input_modes, output_modes)
        classical = self.distinguishable_probability(input_modes, output_modes)

        simulated = []

        for eta in eta_values:
            probs = self.simulate(eta)
            key = str(pcvl.BasicState([1 if i in output_modes else 0 for i in range(self.modes)]))
            simulated.append(probs.get(key, 0))

        return ideal, classical, simulated

    # -------------------------------------------------------
    # LATEX: UNITARY MATRIX
    # -------------------------------------------------------

    def latex_unitary(self, precision=3):
        U = self.unitary
        rows = []
        for row in U:
            formatted = " & ".join(
                f"{np.round(val.real,precision)}"
                + (f"+{np.round(val.imag,precision)}i" if abs(val.imag)>1e-6 else "")
                for val in row
            )
            rows.append(formatted)

        body = " \\\\\n".join(rows)

        latex = "\\begin{pmatrix}\n" + body + "\n\\end{pmatrix}"
        return latex

    # -------------------------------------------------------
    # LATEX: PERMANENT EXPANSION
    # -------------------------------------------------------

    def latex_permanent_expansion(self, input_modes, output_modes):
        A = self.extract_submatrix(input_modes, output_modes)
        n = A.shape[0]

        terms = []
        for p in permutations(range(n)):
            factors = []
            for i in range(n):
                factors.append(f"U_{{{output_modes[i]+1},{input_modes[p[i]]+1}}}")
            terms.append(" ".join(factors))

        expansion = " + \n".join(terms)

        latex = "\\begin{align}\n"
        latex += "\\mathrm{Perm}(U) =\n"
        latex += expansion
        latex += "\n\\end{align}"

        return latex



# Example 3-mode circuit
c = pcvl.Circuit(3)
c.add(0, comp.BS(0.4))
c.add(1, comp.BS(0.7))
c.add(0, comp.BS(0.5))

input_state = pcvl.BasicState([1,1,1])

exp = PhotonicExperiment(c, input_state)

exp.draw()

A = exp.extract_submatrix([0,1,2], [0,1,2])
print(A)


print("Permanent:", exp.permanent(A))
print("Quantum probability:", exp.quantum_probability([0,1,2],[0,1,2]))
print("Distinguishable probability:", exp.distinguishable_probability([0,1,2],[0,1,2]))


etas = np.linspace(1,0,6)
ideal, classical, sim = exp.sweep_eta([0,1,2],[0,1,2],etas)

plt.plot(etas, sim, marker='o')
plt.axhline(ideal, linestyle='--', label='Ideal')
plt.axhline(classical, linestyle='--', label='Classical')
plt.legend()
plt.show()

print(exp.latex_unitary())

print(exp.latex_permanent_expansion([0,1,2],[0,1,2]))
