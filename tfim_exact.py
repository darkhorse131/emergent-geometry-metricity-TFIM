# tfim_exact.py
# Exact TFIM (OBC) solver via JW + BdG with correct Gaussian-state entropy.
# Conventions:
#   Spin Hamiltonian:  H = -J * sum_{i=1}^{L-1} σ_i^x σ_{i+1}^x  -  h * sum_{i=1}^L σ_i^z
#   OBC, lattice spacing a=1, J>0.
#
# This module is aligned with the PRB Rapid plan and integrates cleanly with:
#   - mi_tools.py  (model-agnostic post-processing, MI-floor policy, fits, Δ diagnostic)
#   - run_tfim_study.py (campaign driver)
#
# Key points:
#   • Entropies / mutual information are in NATs.
#   • JW fermionic two-mode MI is used (Gaussian ground state).
#   • Optional bulk trimming (α) for site-averaging / pair lists.
#   • E0 computed from BdG spectrum; at J=0 gives E0 = -h L; at h=J=1, E0/L → -4/π.

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.special  # for xlogy

__all__ = [
    "construct_bdg_hamiltonian",
    "solve_bdg",
    "entropy_gaussian_subsystem",
    "compute_pair_mi_by_r",
    "calculate_averaged_I_r",
]


# -----------------------------------------------------------------------------
# BdG construction & solver
# -----------------------------------------------------------------------------
def construct_bdg_hamiltonian(L: int, h: float, J: float = 1.0) -> np.ndarray:
    """
    Build the 2L×2L BdG matrix H_BdG for TFIM with OBC in the Nambu basis
        Psi = (c_1, ..., c_L, c_1^†, ..., c_L^†)^T.

    Spin convention:
        H_spin = -J * sum_{i=1}^{L-1} σ_i^x σ_{i+1}^x - h * sum_{i=1}^{L} σ_i^z

    JW mapping yields a quadratic fermion Hamiltonian:
        H = sum_{ij} A_{ij} c_i^† c_j
            + (1/2) * sum_{ij} [ B_{ij} c_i^† c_j^† + B_{ij}^* c_j c_i ],

    with
        A_ii = 2h,
        A_{i,i+1} = A_{i+1,i} = -J,
        B_{i,i+1} = -J,   B_{i+1,i} = +J,  (B is real and antisymmetric),
        all other entries 0 (OBC).

    The BdG matrix is:
        H_BdG = [[ A,  B],
                 [-B, -A]]   (real Hermitian since A=A^T, B=-B^T).
    """
    # A: real symmetric (LxL)
    A = np.zeros((L, L), dtype=float)
    np.fill_diagonal(A, 2.0 * h)
    if L > 1:
        off = -J * np.ones(L - 1, dtype=float)
        A += np.diag(off, k=1) + np.diag(off, k=-1)

    # B: real antisymmetric (LxL)
    B = np.zeros((L, L), dtype=float)
    if L > 1:
        B += np.diag(-J * np.ones(L - 1, dtype=float), k=1)
        B += np.diag(+J * np.ones(L - 1, dtype=float), k=-1)  # ensures B^T = -B

    # Assemble BdG (2L x 2L)
    H_BdG = np.block([
        [A,  B],
        [-B, -A],
    ])
    return H_BdG


def solve_bdg(L: int, h: float, J: float = 1.0, eps: float = 1e-12):
    """
    Diagonalize the BdG Hamiltonian and build exact ground-state correlators.

    Returns
    -------
    C : (L,L) ndarray
        Normal correlator C_ij = <c_i^† c_j>.
    F : (L,L) ndarray
        Anomalous correlator F_ij = <c_i c_j>.
    E_gs : float
        Ground-state many-body energy E0 = -0.5 * sum_{ε>0} ε.  (This choice recovers
        E0 = -h L at J=0 and E0/L → -4/π at h=J=1 in our convention.)
    evals : (2L,) ndarray
        BdG eigenvalues (sorted ascending).
    """
    H = construct_bdg_hamiltonian(L, h, J)
    evals, evecs = la.eigh(H)  # Hermitian eigenproblem
    evals = np.array(evals, dtype=float)

    # Positive energies -> Bogoliubov quasiparticles (u,v)
    pos = evals > eps
    Upos = evecs[:L, pos]
    Vpos = evecs[L:, pos]

    # Correlators (Gaussian ground state)
    # At T=0, occupations = 0 for ε>0 quasiparticles ⇒
    #   C = ⟨c^† c⟩ = V V†,  F = ⟨c c⟩ = U V†
    C = Vpos @ Vpos.conj().T
    F = Upos @ Vpos.conj().T

    # Enforce exact symmetries numerically
    C = 0.5 * (C + C.conj().T)    # Hermitian
    F = 0.5 * (F - F.T)           # Antisymmetric

    # Ground-state energy (E/L → -4/π at h=J=1; E = -hL at J=0)
    E_gs = -0.5 * float(np.sum(evals[pos]))

    return C, F, E_gs, evals


# -----------------------------------------------------------------------------
# Gaussian entropy for subsystems (correct 1/2 pairing factor; nats)
# -----------------------------------------------------------------------------
def _build_G_sub(C_sub: np.ndarray, F_sub: np.ndarray) -> np.ndarray:
    """
    Build the 2n×2n Nambu correlation matrix G for a subsystem (indices in A):

        G_A = [[ <c c^†>, <c c> ],
               [ <c^† c^†>, <c^† c> ]]

    where
        <c c^†> = I - C^T,     <c^† c> = C,
        <c c>   = F,           <c^† c^†> = F^†.

    Returns
    -------
    G : (2n,2n) ndarray (Hermitian, spectrum in [0,1] with (λ,1-λ) pairs).
    """
    n = C_sub.shape[0]
    TL = np.eye(n, dtype=C_sub.dtype) - C_sub.T
    TR = F_sub
    BL = F_sub.conj().T
    BR = C_sub
    G = np.block([[TL, TR],
                  [BL, BR]])
    G = 0.5 * (G + G.conj().T)  # Hermiticity
    return G


def entropy_gaussian_subsystem(C_sub: np.ndarray, F_sub: np.ndarray) -> float:
    """
    Von Neumann entropy S (nats) of a fermionic Gaussian state on a subsystem,
    using the 2n×2n Nambu correlation matrix (valid also with pairing).

    The eigenvalues of G_A appear as pairs (λ_k, 1-λ_k). The correct entropy is:
        S = -1/2 * Tr[ G_A ln G_A + (I - G_A) ln(I - G_A) ]
          = sum_{k=1}^n H2(λ_k)   (nats),
    i.e. half the doubled sum over all 2n eigenvalues.
    """
    G = _build_G_sub(C_sub, F_sub)
    w = la.eigvalsh(G)
    w = np.clip(np.real(w), 0.0, 1.0)

    # Sum of binary entropies over all 2n eigenvalues (this is doubled)
    S_doubled = -np.sum(scipy.special.xlogy(w, w) + scipy.special.xlogy(1.0 - w, 1.0 - w))

    # Divide by 2 to account for (λ, 1-λ) pairing in Nambu space.
    S = 0.5 * S_doubled
    return max(0.0, float(S))


# -----------------------------------------------------------------------------
# Mutual information: pair-level and averaged (nats)
# -----------------------------------------------------------------------------
def compute_pair_mi_by_r(
    L: int,
    C: np.ndarray,
    F: np.ndarray,
    r_max: int,
    bulk_trim_fraction: float = 0.0
) -> dict[int, list[float]]:
    """
    Compute per-pair JW two-mode mutual information grouped by separation r.

    I(i:j) = S({i}) + S({j}) - S({i,j}), with S computed from Gaussian correlators (nats).

    Args
    ----
    L : int
        System size.
    C, F : (L,L) arrays
        Ground-state correlators (normal and anomalous).
    r_max : int
        Max separation r to include (1..r_max).
    bulk_trim_fraction : float in [0, 0.5)
        Trim αL sites from each end before forming pairs to reduce OBC edge effects.

    Returns
    -------
    dict {r: [I(i, i+r) values in nats]}
      Pairs producing non-finite entropies are skipped; tiny negative MI are clipped to 0.0.
      No MI-floor is applied here (handled in post-processing).

    Notes
    -----
    This convenience function stores all per-pair values and is therefore O(#pairs) in memory.
    For production averaging with dispersion, prefer `calculate_averaged_I_r(..., return_pair_stats=True)`,
    which streams statistics in O(1) memory per r.
    """
    r_max = int(max(1, r_max))
    alpha = float(bulk_trim_fraction) if bulk_trim_fraction is not None else 0.0
    alpha = max(0.0, min(alpha, 0.49))
    left = int(alpha * L)
    right_limit = L - left

    # Precompute single-site entropies for all sites needed
    S1 = np.empty(L, dtype=float)
    for i in range(L):
        C_i = C[i:i+1, i:i+1]
        F_i = F[i:i+1, i:i+1]
        S1[i] = entropy_gaussian_subsystem(C_i, F_i)

    mi_by_r: dict[int, list[float]] = {}
    for r in range(1, r_max + 1):
        vals: list[float] = []
        i_start = left
        i_stop = max(left, right_limit - r)  # exclusive
        for i in range(i_start, i_stop):
            j = i + r
            # Single-site contributions
            Si = S1[i]
            Sj = S1[j]
            # Two-site entropy
            idx = [i, j]
            C_ij = C[np.ix_(idx, idx)]
            F_ij = F[np.ix_(idx, idx)]
            Sij = entropy_gaussian_subsystem(C_ij, F_ij)
            Iij = Si + Sj - Sij
            if np.isfinite(Iij):
                vals.append(float(Iij) if Iij >= 0.0 else 0.0)  # clip tiny negatives
        mi_by_r[r] = vals
    return mi_by_r


def calculate_averaged_I_r(
    L: int,
    C: np.ndarray,
    F: np.ndarray,
    r_max: int,
    bulk_trim_fraction: float = 0.0,
    return_pair_stats: bool = False,
):
    """
    Site-averaged mutual information I(r) for single-site subsystems (nats), using
    the exact Gaussian-state entropy (with pairing).

    I(i;j) = S({i}) + S({j}) - S({i,j})

    Args
    ----
    L, C, F : system size and correlators
    r_max : maximum separation to include (positive integer)
    bulk_trim_fraction : optional trimming α in [0, 0.5) to reduce OBC edge effects
    return_pair_stats : if True, also return per-r pair counts and dispersion
        • N_r_values : number of (i,j) pairs used at each r
        • sigma_r_jk : sample standard deviation across those pairs at fixed r
                       (std-err available as sigma/√N if needed downstream)

    Returns
    -------
    If return_pair_stats is False (backward-compatible):
        r_values : list[int]
        avg_I_r_values : list[float] (np.nan if no valid pairs at that r)

    If return_pair_stats is True:
        r_values, avg_I_r_values, N_r_values, sigma_r_jk

    Implementation notes
    --------------------
    • This function streams statistics via Welford’s algorithm, keeping O(1) memory per r.
    • Any non-finite MI values are skipped; tiny negatives are clipped to 0.0 before accumulation.
    • No MI-floor is applied here (that policy is handled by post-processing in mi_tools / runner).
    """
    r_max = int(max(1, r_max))
    alpha = float(bulk_trim_fraction) if bulk_trim_fraction is not None else 0.0
    alpha = max(0.0, min(alpha, 0.49))
    left = int(alpha * L)
    right_limit = L - left

    # Precompute single-site entropies
    S1 = np.empty(L, dtype=float)
    for i in range(L):
        C_i = C[i:i+1, i:i+1]
        F_i = F[i:i+1, i:i+1]
        S1[i] = entropy_gaussian_subsystem(C_i, F_i)

    r_values = list(range(1, r_max + 1))
    avg_I_r_values: list[float] = []

    N_r_values: list[int] = []
    sigma_r_jk: list[float] = []

    # For each separation r, stream (mean, variance) via Welford
    for r in r_values:
        n = 0
        mean = 0.0
        M2 = 0.0

        i_start = left
        i_stop = max(left, right_limit - r)  # exclusive
        for i in range(i_start, i_stop):
            j = i + r
            # 1-site contributions
            Si = S1[i]
            Sj = S1[j]
            # 2-site entropy
            idx = [i, j]
            C_ij = C[np.ix_(idx, idx)]
            F_ij = F[np.ix_(idx, idx)]
            Sij = entropy_gaussian_subsystem(C_ij, F_ij)
            Iij = Si + Sj - Sij
            if not np.isfinite(Iij):
                continue
            if Iij < 0.0:
                Iij = 0.0  # clip tiny negatives

            # Welford update
            n_new = n + 1
            delta = Iij - mean
            mean += delta / n_new
            delta2 = Iij - mean
            M2 += delta * delta2
            n = n_new

        # finalize stats for this r
        if n > 0:
            avg_I_r_values.append(float(mean))
        else:
            avg_I_r_values.append(np.nan)

        N_r_values.append(int(n))
        if n > 1:
            var = M2 / (n - 1)  # sample variance
            sigma_r_jk.append(float(np.sqrt(var)))
        else:
            sigma_r_jk.append(float("nan"))

    if return_pair_stats:
        return r_values, avg_I_r_values, N_r_values, sigma_r_jk
    else:
        return r_values, avg_I_r_values