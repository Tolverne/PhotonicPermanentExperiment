from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler


# ============================================================
# Types
# ============================================================

BasicStateLike = Union[pcvl.BasicState, Sequence[int]]


# ============================================================
# Helpers
# ============================================================

def to_basic_state(x: BasicStateLike) -> pcvl.BasicState:
    if isinstance(x, pcvl.BasicState):
        return x
    return pcvl.BasicState(list(x))

def photon_count(st: pcvl.BasicState) -> int:
    # pcvl.BasicState supports sum(st) in many versions; keep robust:
    return int(sum(list(st)))

def as_int_list(st: pcvl.BasicState) -> List[int]:
    return list(st)

def pretty_state(st: pcvl.BasicState) -> str:
    return str(as_int_list(st))

def tv_distance_dict(P: Dict[str, float], Q: Dict[str, float]) -> float:
    keys = set(P.keys()) | set(Q.keys())
    return 0.5 * sum(abs(P.get(k, 0.0) - Q.get(k, 0.0)) for k in keys)

def normalize_dict(P: Dict[str, float]) -> Dict[str, float]:
    s = sum(P.values())
    if s <= 0:
        return P
    return {k: v / s for k, v in P.items()}


# ============================================================
# Circuit Builders (optional convenience)
# ============================================================

class CircuitBuilder:
    """Factory helpers for common circuits. You can also pass any pcvl.Circuit directly."""
    @staticmethod
    def three_mode_chain(theta1=0.4, theta2=0.7, theta3=0.5) -> pcvl.Circuit:
        c = pcvl.Circuit(3)
        c.add(0, comp.BS(theta1))
        c.add(1, comp.BS(theta2))
        c.add(0, comp.BS(theta3))
        return c

    @staticmethod
    def random_mesh(m: int, depth: int = None, seed: int = 7) -> pcvl.Circuit:
        """Simple nearest-neighbour mesh using BS + phase-like randomness through BS params."""
        rng = np.random.default_rng(seed)
        if depth is None:
            depth = 2 * m

        c = pcvl.Circuit(m)
        for layer in range(depth):
            start = 0 if layer % 2 == 0 else 1
            for i in range(start, m - 1, 2):
                theta = float(rng.uniform(0, np.pi/2))
                c.add(i, comp.BS(theta))
        return c


# ============================================================
# Rendering / Diagram
# ============================================================

class CircuitRenderer:
    """
    Version-safe circuit rendering.
    - In notebooks, pcvl.pdisplay(circuit) often works.
    - For scripts/pycharm, you usually want save_svg/save_png.
    """

    @staticmethod
    def try_display(circuit: pcvl.Circuit):
        # Works in Jupyter; may no-op in scripts.
        try:
            return pcvl.pdisplay(circuit)
        except Exception:
            return None

    @staticmethod
    def save_svg(circuit: pcvl.Circuit, path: str = "circuit.svg") -> bool:
        """
        Attempts to save a circuit drawing to SVG.
        Perceval rendering APIs vary; we try a few known import paths.
        """
        # Try a few different internal APIs depending on Perceval version
        try:
            from perceval.rendering.circuit import draw_circuit  # type: ignore
            fig = draw_circuit(circuit)
            fig.savefig(path)
            return True
        except Exception:
            pass

        try:
            from perceval.rendering import CircuitRenderer as CR  # type: ignore
            # Some versions expose a renderer class
            svg = CR().render(circuit, output_format="svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
            return True
        except Exception:
            pass

        return False


# ============================================================
# Matrix / Permanent
# ============================================================

class MatrixTools:
    @staticmethod
    def unitary_of(circuit: pcvl.Circuit) -> np.ndarray:
        return np.array(circuit.compute_unitary(), dtype=complex)

    @staticmethod
    def submatrix(U: np.ndarray, input_modes: Sequence[int], output_modes: Sequence[int]) -> np.ndarray:
        A = np.zeros((len(output_modes), len(input_modes)), dtype=complex)
        for i, r in enumerate(output_modes):
            for j, s in enumerate(input_modes):
                A[i, j] = U[r, s]
        return A

    @staticmethod
    def permanent_bruteforce(A: np.ndarray) -> complex:
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("Permanent requires a square matrix.")
        total = 0.0 + 0.0j
        for p in permutations(range(n)):
            prod = 1.0 + 0.0j
            for i in range(n):
                prod *= A[i, p[i]]
            total += prod
        return total

    @staticmethod
    def distinguishable_event_probability(U: np.ndarray,
                                         input_modes_labeled: Sequence[int],
                                         output_modes_labeled: Sequence[int]) -> float:
        """
        Labeled photons: probability is sum over all matchings between inputs and outputs
        of product |U[out_i, in_{perm(i)}]|^2.
        Collision-free event only (outputs are distinct and length N).
        """
        if len(input_modes_labeled) != len(output_modes_labeled):
            raise ValueError("Input and output lists must have same length.")
        N = len(input_modes_labeled)
        T = np.abs(U) ** 2
        total = 0.0
        for p in permutations(range(N)):
            prod = 1.0
            for i in range(N):
                prod *= float(T[output_modes_labeled[i], input_modes_labeled[p[i]]])
            total += prod
        return total


# ============================================================
# Distributions
# ============================================================

class DistributionModel:
    """
    Computes distributions over Fock outputs.
    - Quantum: via Perceval sampling (supports noise)
    - Distinguishable: labeled-photon classical propagation then aggregation into Fock counts
    """

    @staticmethod
    def quantum_sample_distribution(circuit: pcvl.Circuit,
                                    input_state: pcvl.BasicState,
                                    eta: float = 1.0,
                                    transmittance: float = 1.0,
                                    shots: int = 20000,
                                    postselect_n: Optional[int] = None) -> Tuple[Dict[str, float], float]:
        """
        Returns (distribution_dict, success_rate).
        Distribution dict keys are str(BasicState), values sum to 1 over *kept* events.
        success_rate = kept/shots (useful when loss + postselection)
        """
        noise = pcvl.NoiseModel(indistinguishability=float(eta), transmittance=float(transmittance))
        proc = pcvl.Processor("SLOS", circuit, noise=noise)
        proc.with_input(input_state)

        if postselect_n is None:
            postselect_n = photon_count(input_state)

        # Critical to avoid your earlier Perceval error:
        proc.min_detected_photons_filter(int(postselect_n))

        sampler = Sampler(proc)
        out = sampler.sample_count(int(shots))
        if out is None:
            raise RuntimeError("Perceval sampler returned None (job failure).")

        counts = out["results"]
        kept = sum(counts.values())
        success_rate = kept / float(shots) if shots > 0 else 0.0

        P: Dict[str, float] = {}
        if kept == 0:
            return P, success_rate

        for st, c in counts.items():
            P[str(st)] = P.get(str(st), 0.0) + c / float(kept)

        return normalize_dict(P), success_rate

    @staticmethod
    def distinguishable_distribution(U: np.ndarray,
                                     input_state: pcvl.BasicState) -> Dict[str, float]:
        """
        Fully distinguishable photons distribution (no interference):
        model photons as labeled, each propagating independently with transition probs |U|^2,
        then aggregate into output Fock occupancies.

        Works best for small N (<=4 or 5) because complexity is M^N.
        """
        n_list = as_int_list(input_state)
        M = len(n_list)

        # Expand into labeled photon input modes
        in_modes: List[int] = []
        for mode, occ in enumerate(n_list):
            in_modes.extend([mode] * int(occ))
        N = len(in_modes)

        T = np.abs(U) ** 2  # T[out, in]
        P: Dict[str, float] = {}

        def rec(k: int, occ: List[int], prob: float):
            if k == N:
                st = pcvl.BasicState(occ)
                key = str(st)
                P[key] = P.get(key, 0.0) + prob
                return
            in_k = in_modes[k]
            for out_k in range(M):
                occ2 = occ.copy()
                occ2[out_k] += 1
                rec(k + 1, occ2, prob * float(T[out_k, in_k]))

        rec(0, [0]*M, 1.0)
        return normalize_dict(P)


# ============================================================
# Experiment Spec + Hierarchy
# ============================================================

@dataclass
class EventSpec:
    """
    Defines one event you want to verify.
    For now: collision-free mapping with N photons in distinct input modes and distinct output modes.
    """
    input_modes: Tuple[int, ...]
    output_modes: Tuple[int, ...]

    def __post_init__(self):
        if len(self.input_modes) != len(self.output_modes):
            raise ValueError("EventSpec requires equal-length input/output mode tuples.")
        if len(set(self.input_modes)) != len(self.input_modes):
            # collision-free input assumption for the permanent mapping (simplifies teaching)
            # You can lift this later with repeated rows/cols formalism.
            raise ValueError("EventSpec currently assumes collision-free input modes (all distinct).")
        if len(set(self.output_modes)) != len(self.output_modes):
            raise ValueError("EventSpec currently assumes collision-free output modes (all distinct).")

    @property
    def n(self) -> int:
        return len(self.input_modes)


@dataclass
class ExperimentSpec:
    name: str
    circuit: pcvl.Circuit
    input_state: pcvl.BasicState
    event: Optional[EventSpec] = None


# ============================================================
# LaTeX Reporting
# ============================================================

class LatexReporter:
    @staticmethod
    def matrix_to_pmatrix(A: np.ndarray, precision: int = 3) -> str:
        def fmt(z: complex) -> str:
            r = round(float(np.real(z)), precision)
            im = round(float(np.imag(z)), precision)
            if abs(im) < 10**(-precision):
                return f"{r}"
            sign = "+" if im >= 0 else "-"
            return f"{r}{sign}{abs(im)}i"

        rows = [" & ".join(fmt(x) for x in row) for row in A]
        body = " \\\\\n".join(rows)
        return "\\begin{pmatrix}\n" + body + "\n\\end{pmatrix}"

    @staticmethod
    def permanent_expansion_latex(event: EventSpec) -> str:
        """
        Symbolic permanent expansion in terms of U_{r,s}.
        Uses 1-based indices in LaTeX.
        """
        n = event.n
        perms = list(permutations(range(n)))
        terms = []
        for p in perms:
            factors = []
            for i in range(n):
                r = event.output_modes[i] + 1
                s = event.input_modes[p[i]] + 1
                factors.append(f"U_{{{r},{s}}}")
            terms.append(" ".join(factors))
        return "\\begin{align}\n\\mathrm{Perm}(U_{\\mathrm{sub}}) &= " + " + \\\\\n& ".join(terms) + "\n\\end{align}\n"

    @staticmethod
    def section_snippet(spec: ExperimentSpec,
                        U: np.ndarray,
                        A: Optional[np.ndarray],
                        perm_val: Optional[complex],
                        p_quant: Optional[float],
                        p_dist: Optional[float]) -> str:
        lines = []
        lines.append("\\section{Permanent Verification Experiment}\n")
        lines.append(f"\\subsection{{{spec.name}}}\n")
        lines.append("The interferometer implements a unitary transformation $U$ over spatial modes.\n\n")
        lines.append("\\[\nU = " + LatexReporter.matrix_to_pmatrix(U) + "\n\\]\n\n")

        if spec.event is not None and A is not None:
            lines.append("For the collision-free event with input modes "
                         f"$\\mathbf{{s}}={spec.event.input_modes}$ and output modes "
                         f"$\\mathbf{{r}}={spec.event.output_modes}$, the relevant submatrix is\n\n")
            lines.append("\\[\nU_{\\mathrm{sub}} = " + LatexReporter.matrix_to_pmatrix(A) + "\n\\]\n\n")
            lines.append(LatexReporter.permanent_expansion_latex(spec.event) + "\n")
            if perm_val is not None:
                lines.append("\\[\n\\mathrm{Perm}(U_{\\mathrm{sub}}) = " +
                             LatexReporter._cplx_latex(perm_val) + "\n\\]\n\n")
            if p_quant is not None:
                lines.append("\\[\nP_{\\mathrm{bos}}(\\mathbf{r}|\\mathbf{s}) = \\left|\\mathrm{Perm}(U_{\\mathrm{sub}})\\right|^2 = "
                             + f"{p_quant:.6f}\n\\]\n\n")
            if p_dist is not None:
                lines.append("\\[\nP_{\\mathrm{dist}}(\\mathbf{r}|\\mathbf{s}) = "
                             + f"{p_dist:.6f}\n\\]\n\n")

        return "".join(lines)

    @staticmethod
    def _cplx_latex(z: complex, precision: int = 4) -> str:
        r = round(float(np.real(z)), precision)
        im = round(float(np.imag(z)), precision)
        if abs(im) < 10**(-precision):
            return f"{r}"
        sign = "+" if im >= 0 else "-"
        return f"{r} {sign} {abs(im)} i"


# ============================================================
# The main high-level object
# ============================================================

class PhotonicExperiment:
    """
    High-level experiment object that:
    - keeps spec + cached unitary
    - can render circuit
    - can compute permanents for a chosen event
    - can generate distributions (quantum + distinguishable)
    - can sweep eta/loss
    - can export LaTeX snippets
    """

    def __init__(self, spec: ExperimentSpec):
        self.spec = spec
        self.U = MatrixTools.unitary_of(spec.circuit)

    # ---------- Rendering ----------
    def display_circuit(self):
        return CircuitRenderer.try_display(self.spec.circuit)

    def save_circuit_svg(self, path: str = "circuit.svg") -> bool:
        return CircuitRenderer.save_svg(self.spec.circuit, path)

    # ---------- Event math ----------
    def event_submatrix(self) -> np.ndarray:
        if self.spec.event is None:
            raise ValueError("No EventSpec provided in ExperimentSpec.")
        return MatrixTools.submatrix(self.U, self.spec.event.input_modes, self.spec.event.output_modes)

    def event_permanent(self) -> complex:
        A = self.event_submatrix()
        return MatrixTools.permanent_bruteforce(A)

    def event_prob_bosonic(self) -> float:
        perm = self.event_permanent()
        return float(abs(perm) ** 2)

    def event_prob_distinguishable(self) -> float:
        if self.spec.event is None:
            raise ValueError("No EventSpec provided.")
        return float(MatrixTools.distinguishable_event_probability(self.U,
                                                                  list(self.spec.event.input_modes),
                                                                  list(self.spec.event.output_modes)))

    # ---------- Distributions ----------
    def quantum_distribution(self, eta: float = 1.0, transmittance: float = 1.0,
                             shots: int = 20000, postselect_n: Optional[int] = None) -> Tuple[Dict[str, float], float]:
        return DistributionModel.quantum_sample_distribution(
            self.spec.circuit, self.spec.input_state,
            eta=eta, transmittance=transmittance, shots=shots, postselect_n=postselect_n
        )

    def distinguishable_distribution(self) -> Dict[str, float]:
        return DistributionModel.distinguishable_distribution(self.U, self.spec.input_state)

    # ---------- Comparison utilities ----------
    def compare_quantum_vs_distinguishable(self, eta: float = 1.0, shots: int = 20000,
                                          top_k: int = 25, sort_by: str = "diff"):
        Pq, sr = self.quantum_distribution(eta=eta, shots=shots)
        Pc = self.distinguishable_distribution()

        tv = tv_distance_dict(Pq, Pc)
        print(f"[compare] eta={eta:.3f}  shots={shots}  success_rate={sr:.3f}  TV={tv:.4f}")

        # build aligned lists
        keys = list(set(Pq.keys()) | set(Pc.keys()))
        if sort_by == "q":
            keys.sort(key=lambda k: Pq.get(k, 0.0), reverse=True)
        elif sort_by == "c":
            keys.sort(key=lambda k: Pc.get(k, 0.0), reverse=True)
        else:  # diff
            keys.sort(key=lambda k: abs(Pq.get(k, 0.0) - Pc.get(k, 0.0)), reverse=True)

        keys = keys[:top_k]
        q_vals = [Pq.get(k, 0.0) for k in keys]
        c_vals = [Pc.get(k, 0.0) for k in keys]

        x = np.arange(len(keys))
        plt.figure(figsize=(10, 4))
        plt.plot(x, c_vals, marker="o", label="Distinguishable (classical)")
        plt.plot(x, q_vals, marker="o", label=f"Quantum sampled (eta={eta:.2f})")
        plt.xticks(x, keys, rotation=90)
        plt.ylabel("Probability")
        plt.title(f"Distribution overlay — {self.spec.name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # difference bars
        diff = np.array(q_vals) - np.array(c_vals)
        plt.figure(figsize=(10, 3.6))
        plt.axhline(0.0, linewidth=1)
        plt.bar(x, diff)
        plt.xticks(x, keys, rotation=90)
        plt.ylabel("Quantum - Classical")
        plt.title(f"Most-different outcomes — {self.spec.name}")
        plt.tight_layout()
        plt.show()

        return tv

    def sweep_eta(self, eta_values: Sequence[float], shots: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays: eta_values, TV(eta) between quantum distribution and distinguishable distribution.
        """
        Pc = self.distinguishable_distribution()
        tvs = []
        for eta in eta_values:
            Pq, _ = self.quantum_distribution(eta=float(eta), shots=shots)
            tvs.append(tv_distance_dict(Pq, Pc))
        return np.array(list(eta_values), dtype=float), np.array(tvs, dtype=float)

    def phase_diagram(self, eta_values: Sequence[float], T_values: Sequence[float],
                      shots: int = 20000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (TV_grid, SR_grid, params) as arrays.
        """
        Pc = self.distinguishable_distribution()
        TV = np.zeros((len(T_values), len(eta_values)))
        SR = np.zeros((len(T_values), len(eta_values)))

        for i, T in enumerate(T_values):
            for j, eta in enumerate(eta_values):
                Pq, sr = self.quantum_distribution(eta=float(eta), transmittance=float(T), shots=shots)
                TV[i, j] = tv_distance_dict(Pq, Pc)
                SR[i, j] = sr
        return TV, SR, np.array(list(eta_values), dtype=float)

    # ---------- LaTeX ----------
    def latex_section(self) -> str:
        A = None
        perm_val = None
        p_quant = None
        p_dist = None
        if self.spec.event is not None:
            A = self.event_submatrix()
            perm_val = MatrixTools.permanent_bruteforce(A)
            p_quant = float(abs(perm_val) ** 2)
            p_dist = self.event_prob_distinguishable()
        return LatexReporter.section_snippet(self.spec, self.U, A, perm_val, p_quant, p_dist)


# ============================================================
# Example usage (safe starter)
# ============================================================

if __name__ == "__main__":
    # Build a physically plausible 3-mode circuit with nonzero permanent
    c = CircuitBuilder.three_mode_chain(theta1=0.4, theta2=0.7, theta3=0.5)
    input_state = pcvl.BasicState([1, 1, 1])

    # Choose a collision-free event: input (0,1,2) -> output (0,1,2)
    event = EventSpec(input_modes=(0, 1, 2), output_modes=(0, 1, 2))

    spec = ExperimentSpec(
        name="3-photon / 3-mode permanent verification (BS chain)",
        circuit=c,
        input_state=input_state,
        event=event
    )

    exp = PhotonicExperiment(spec)

    # Try to display; if in PyCharm, save SVG instead
    exp.display_circuit()
    ok = exp.save_circuit_svg("circuit.svg")
    print("Saved circuit.svg:", ok)

    # Print analytic event values
    A = exp.event_submatrix()
    perm = exp.event_permanent()
    print("U_sub=\n", A)
    print("Perm(U_sub) =", perm)
    print("Bosonic event prob |Perm|^2 =", exp.event_prob_bosonic())
    print("Distinguishable event prob  =", exp.event_prob_distinguishable())

    # Compare distributions and sweep eta
    exp.compare_quantum_vs_distinguishable(eta=1.0, shots=20000, top_k=20, sort_by="diff")

    etas = np.linspace(1.0, 0.0, 6)
    xs, tvs = exp.sweep_eta(etas, shots=20000)
    plt.figure()
    plt.plot(xs, tvs, marker="o")
    plt.xlabel("eta")
    plt.ylabel("TV( quantum(eta), distinguishable )")
    plt.title("Indistinguishability → classicalisation")
    plt.grid(True)
    plt.show()

    # Export LaTeX snippet
    tex = exp.latex_section()
    with open("experiment_section.tex", "w", encoding="utf-8") as f:
        f.write(tex)
    print("Wrote experiment_section.tex")
