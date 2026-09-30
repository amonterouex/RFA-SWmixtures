"""RFA for square-well / square-shoulder mixtures.

Python translation of the supplied Mathematica notebook.  The notation and
comments follow the notebook as closely as practical, while the numerical
implementation avoids repeated symbolic algebra.

The mixture is multicomponent: binary, ternary, and larger mixtures use the
same code.  As in the notebook, a common shell width Delta is assumed and the
hard-core cross diameter is additive, sigma_ij = (sigma_i + sigma_j)/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Literal

import numpy as np
from scipy.linalg import expm, lu_factor, lu_solve
from scipy.optimize import root


Array = np.ndarray


@dataclass(frozen=True)
class SolverInfo:
    """Small collection of diagnostics written to the output file."""

    success: bool
    message: str
    nfev: int
    max_continuity_residual: float
    eq55_condition_number: float


class RFAMixture:
    """Rational-function approximation (RFA) for a multicomponent mixture."""

    def __init__(
        self,
        mole_fractions: Array,
        diameters: Array,
        epsilon: Array,
        delta: float,
        rho: float,
        beta: float,
        *,
        tolerance: float = 1.0e-10,
    ) -> None:
        self.x = np.asarray(mole_fractions, dtype=float)
        self.diameters = np.asarray(diameters, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.delta = float(delta)
        self.rho = float(rho)
        self.beta = float(beta)
        self.tolerance = float(tolerance)
        self.n = self.x.size

        self._validate_input()

        # Additive hard-core cross diameters, sigma_ij.
        self.sigma = 0.5 * (self.diameters[:, None] + self.diameters[None, :])
        self.lambda_ = self.sigma + self.delta
        self.lambda_diag = self.diameters + self.delta
        self.sigma_min = float(np.min(self.diameters))

        self.exp_minus_beta_epsilon = np.exp(-self.beta * self.epsilon)
        self.exp_plus_beta_epsilon = np.exp(+self.beta * self.epsilon)

        self.packing_fraction = (
            np.pi / 6.0 * self.rho * np.sum(self.x * self.diameters**3)
        )

        # Eq. (55) is affine in L^(1).  Its matrix is independent of Lbar^(1),
        # so build/factor it once and reuse the factorization inside FindRoot.
        self._build_eq55_linear_system()

        self.solver_info: SolverInfo | None = None
        self.Lambda: Array | None = None
        self.L0: Array | None = None
        self.L1: Array | None = None
        self.Lb0: Array | None = None
        self.Lb1: Array | None = None
        self.mu: Array | None = None

        self._C: Array | None = None
        self._B: Array | None = None
        self._H_xi: Array | None = None
        self._H_xib: Array | None = None

        # Numerical caches used only after the coefficient problem is solved.
        self._xi_cache: dict[float, tuple[Array, Array]] = {}
        self._gg_cache: dict[complex, tuple[Array, Array]] = {}

    def _validate_input(self) -> None:
        if self.n < 1:
            raise ValueError("At least one species is required.")
        if self.diameters.shape != (self.n,):
            raise ValueError("diameters must have the same length as mole_fractions.")
        if self.epsilon.shape != (self.n, self.n):
            raise ValueError(
                f"epsilon must be a {self.n} x {self.n} matrix for {self.n} species."
            )
        if np.any(self.x < 0.0):
            raise ValueError("Mole fractions cannot be negative.")
        if abs(float(np.sum(self.x)) - 1.0) > self.tolerance:
            raise ValueError("The sum of the mole fractions must equal 1.")
        if np.any(self.diameters <= 0.0):
            raise ValueError("All diameters must be positive.")
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative.")
        if self.rho < 0.0:
            raise ValueError("rho must be non-negative.")
        if not np.allclose(
            self.epsilon,
            self.epsilon.T,
            atol=self.tolerance,
            rtol=0.0,
        ):
            raise ValueError("epsilon must be symmetric.")

    # ------------------------------------------------------------------
    # Determination of the RFA coefficients
    # ------------------------------------------------------------------

    def _lambda_l0_lb0(self, L1: Array, Lb1: Array) -> tuple[Array, Array, Array]:
        """Eq. (59) and Eq. (58): Lambda_j, L_ij^(0), and Lbar_ij^(0)."""

        e = self.exp_minus_beta_epsilon
        xcol = self.x[:, None]
        sigcol = self.diameters[:, None]
        lamcol = self.lambda_diag[:, None]

        denominator = 1.0 + (np.pi / 3.0) * self.rho * np.sum(xcol* ( e * sigcol**3 - (e - 1.0) * lamcol**3), axis=0,)
        numerator = 1.0 + np.pi * self.rho * np.sum( xcol * (L1 * sigcol**2 + Lb1 * lamcol**2), axis=0,)
        Lambda = numerator / denominator

        # Eq. (58): L_ij^(0)
        L0 = e * Lambda[None, :]

        # Eq. (58): Lbar_ij^(0)
        Lb0 = (1.0 - e) * Lambda[None, :]
        return Lambda, L0, Lb0

    def _eq55_residual(self, L1: Array, Lb1: Array) -> Array:
        """Eq. (55): low-s consistency condition."""

        _, L0, Lb0 = self._lambda_l0_lb0(L1, Lb1)
        rhs = np.empty((self.n, self.n), dtype=float)

        for i in range(self.n):
            for j in range(self.n):
                sum1 = 0.0
                sum2 = 0.0
                for k in range(self.n):
                    sum1 += self.x[k] * self.sigma[i, k] * (
                        L1[k, j] * self.diameters[k] ** 2
                        + Lb1[k, j] * self.lambda_diag[k] ** 2
                        - L0[k, j] * self.diameters[k] ** 3 / 3.0
                        - Lb0[k, j] * self.lambda_diag[k] ** 3 / 3.0
                    )
                    sum2 += self.x[k] * (
                        L0[k, j] * self.diameters[k] ** 4 / 4.0
                        + Lb0[k, j] * self.lambda_diag[k] ** 4 / 4.0
                        - L1[k, j] * self.diameters[k] ** 3
                        - Lb1[k, j] * self.lambda_diag[k] ** 3
                    )

                rhs[i, j] = (
                    -Lb1[i, j]
                    + Lb0[i, j] * self.delta
                    + self.sigma[i, j]
                    + np.pi * self.rho * sum1
                    + (np.pi / 3.0) * self.rho * sum2
                )

        return L1 - rhs

    def _build_eq55_linear_system(self) -> None:
        """Build the constant linear operator multiplying L^(1) in Eq. (55)."""

        zero = np.zeros((self.n, self.n), dtype=float)
        f0 = self._eq55_residual(zero, zero).ravel()
        size = self.n * self.n
        matrix = np.empty((size, size), dtype=float)

        for column in range(size):
            basis = np.zeros((self.n, self.n), dtype=float)
            basis.ravel()[column] = 1.0
            matrix[:, column] = self._eq55_residual(basis, zero).ravel() - f0

        self._eq55_matrix = matrix
        self._eq55_lu = lu_factor(matrix)
        self._eq55_condition_number = float(np.linalg.cond(matrix))

    def _solve_l1_from_lb1(self, Lb1: Array) -> Array:
        """Solution of Eq. (55): L^(1) in terms of Lbar^(1)."""

        zero = np.zeros((self.n, self.n), dtype=float)
        constant = self._eq55_residual(zero, Lb1).ravel()
        return lu_solve(self._eq55_lu, -constant).reshape(self.n, self.n)

    def _ahat_coefficients( self, L1: Array, Lb1: Array, L0: Array, Lb0: Array,) -> tuple[Array, Array, Array]:
        """Eq. (45): Ahat_ij(s) = A0 + A1*s + A2*s^2."""

        factor = 2.0 * np.pi * self.rho * self.x[:, None]
        sigcol = self.diameters[:, None]
        lamcol = self.lambda_diag[:, None]

        A0 = factor * (-Lb0 - L0)
        A1 = factor * (L0 * sigcol + Lb0 * lamcol - L1 - Lb1)
        A2 = factor * (
            -Lb0 * lamcol**2 / 2.0
            - L0 * sigcol**2 / 2.0
            + L1 * sigcol
            + Lb1 * lamcol
        )
        return A0, A1, A2

    def _companion_system(self,L1: Array, Lb1: Array, L0: Array, Lb0: Array,) -> tuple[Array, Array]:
        """State-space form used to invert Xi(s) without symbolic algebra."""

        A0, A1, A2 = self._ahat_coefficients(L1, Lb1, L0, Lb0)
        z = np.zeros((self.n, self.n), dtype=float)
        eye = np.eye(self.n)

        # P(s) = s^3 I + A2 s^2 + A1 s + A0.
        # The block companion matrix gives P(s)^(-1), s P(s)^(-1), and
        # s^2 P(s)^(-1) as blocks of (sI-C)^(-1) B.
        C = np.block(
            [
                [z, eye, z],
                [z, z, eye],
                [-A0, -A1, -A2],
            ]
        )
        B = np.vstack([z, z, eye])
        return C, B

    def _xi_matrix_for_coefficients(
        self,
        t: float,
        L1: Array,
        Lb1: Array,
        *,
        barred: bool = False,
    ) -> Array:
        """Inverse Laplace transform of Xi_ij(s) or Xibar_ij(s)."""

        _, L0, Lb0 = self._lambda_l0_lb0(L1, Lb1)
        if abs(t) <= 1.0e-15:
            return Lb1.copy() if barred else L1.copy()

        C, B = self._companion_system(L1, Lb1, L0, Lb0)
        exp_block = expm(C * t) @ B
        block_sP = exp_block[self.n : 2 * self.n, :]
        block_s2P = exp_block[2 * self.n : 3 * self.n, :]

        if barred:
            return Lb0 @ block_sP + Lb1 @ block_s2P
        return L0 @ block_sP + L1 @ block_s2P

    def solve_coefficients(self) -> SolverInfo:
        """Solve Eq. (56) for Lbar^(1), then prepare the final RFA model."""

        # Initial guess for Lbar_ij^(1), as in the Mathematica notebook.
        seed = (
            self.lambda_
            * (1.0 - self.exp_minus_beta_epsilon)
            * (1.0 + self.rho / 2.0)
        )

        # Eq. (56): impose continuity of the cavity functions to obtain
        # Lbar_ij^(1).
        def continuity_residual(flat_Lb1: Array) -> Array:
            Lb1 = flat_Lb1.reshape(self.n, self.n)
            L1 = self._solve_l1_from_lb1(Lb1)
            xi_delta = self._xi_matrix_for_coefficients(self.delta, L1, Lb1)
            residual = Lb1 - (self.exp_plus_beta_epsilon - 1.0) * xi_delta
            return residual.ravel()

        solution = root(
            continuity_residual,
            seed.ravel(),
            method="hybr",
            options={"xtol": self.tolerance},
        )

        max_residual = float(np.max(np.abs(solution.fun)))
        info = SolverInfo(
            success=bool(solution.success),
            message=str(solution.message),
            nfev=int(solution.nfev),
            max_continuity_residual=max_residual,
            eq55_condition_number=self._eq55_condition_number,
        )
        self.solver_info = info

        if not solution.success:
            raise RuntimeError(
                "Coefficient solver did not converge: "
                f"{solution.message}; max residual={max_residual:.3e}"
            )

        self.Lb1 = solution.x.reshape(self.n, self.n)
        self.L1 = self._solve_l1_from_lb1(self.Lb1)
        self.Lambda, self.L0, self.Lb0 = self._lambda_l0_lb0(self.L1, self.Lb1)

        self._C, self._B = self._companion_system(
            self.L1, self.Lb1, self.L0, self.Lb0
        )
        zeros = np.zeros((self.n, self.n), dtype=float)
        self._H_xi = np.hstack([zeros, self.L0, self.L1])
        self._H_xib = np.hstack([zeros, self.Lb0, self.Lb1])

        # Eq. (53b): Xibar'_ij(0)
        xib_prime_zero = np.empty((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                total = 0.0
                for k in range(self.n):
                    total += self.Lb1[i, k] * self.x[k] * (
                        self.L1[k, j] * self.diameters[k]
                        + self.Lb1[k, j] * self.lambda_diag[k]
                        - self.L0[k, j] * self.diameters[k] ** 2 / 2.0
                        - self.Lb0[k, j] * self.lambda_diag[k] ** 2 / 2.0
                    )
                xib_prime_zero[i, j] = (self.Lb0[i, j] - 2.0 * np.pi * self.rho * total)

        # Laplace transform relation in the notebook:
        # Xi'_ij(s) = s Xi_ij(s) - L_ij^(1).
        # In time space, for r>0, this is simply d Xi_ij(r)/dr.
        exp_delta = expm(self._C * self.delta) @ self._B
        xi_prime_delta = self._H_xi @ self._C @ exp_delta

        # Eq. (51): mu_ij
        self.mu = (xib_prime_zero + (1.0 - self.exp_plus_beta_epsilon) * xi_prime_delta) / self.lambda_

        self._xi_cache.clear()
        self._gg_cache.clear()
        return info

    def _require_solved(self) -> None:
        if self.L1 is None:
            raise RuntimeError("Call solve_coefficients() before evaluating correlations.")

    # ------------------------------------------------------------------
    # xi_ij(r), xibar_ij(r), g_ij(r), and y_ij(r)
    # ------------------------------------------------------------------

    @staticmethod
    def _time_key(t: float) -> float:
        # Grid arithmetic can generate tiny round-off differences.  Rounding the
        # cache key lets different species pairs reuse the same matrix exponential.
        return round(float(t), 14)

    def _xi_pair_matrices(self, t: float) -> tuple[Array, Array]:
        self._require_solved()
        key = self._time_key(t)
        cached = self._xi_cache.get(key)
        if cached is not None:
            return cached

        if abs(key) <= 1.0e-15:
            result = (self.L1.copy(), self.Lb1.copy())
        else:
            exp_block = expm(self._C * key) @ self._B
            result = (self._H_xi @ exp_block, self._H_xib @ exp_block)

        self._xi_cache[key] = result
        return result

    # Eq. (21): phi_m(x)
    @staticmethod
    def _phi1(z: complex | Array) -> complex | Array:
        z = np.asarray(z, dtype=complex)
        out = np.expm1(-z) + z
        small = np.abs(z) < 1.0e-4
        if np.any(small):
            zs = z[small] if z.ndim else z
            series = zs**2 / 2.0 - zs**3 / 6.0 + zs**4 / 24.0 - zs**5 / 120.0
            if z.ndim:
                out[small] = series
            elif small:
                out = series
        return out.item() if np.ndim(out) == 0 else out

    @staticmethod
    def _phi2(z: complex | Array) -> complex | Array:
        z = np.asarray(z, dtype=complex)
        out = np.expm1(-z) + z - z**2 / 2.0
        small = np.abs(z) < 1.0e-4
        if np.any(small):
            zs = z[small] if z.ndim else z
            series = -zs**3 / 6.0 + zs**4 / 24.0 - zs**5 / 120.0 + zs**6 / 720.0
            if z.ndim:
                out[small] = series
            elif small:
                out = series
        return out.item() if np.ndim(out) == 0 else out

    @staticmethod
    def _complex_key(s: complex) -> complex:
        return complex(round(float(np.real(s)), 14), round(float(np.imag(s)), 14))

    def _gg_matrices(self, s: complex) -> tuple[Array, Array]:
        """Eq. (39a): the two terms entering G_ij^(0)(s)."""

        self._require_solved()
        key = self._complex_key(s)
        cached = self._gg_cache.get(key)
        if cached is not None:
            return cached

        s = complex(s)
        if abs(s) < 1.0e-14:
            raise ValueError("The G(s) expression is singular at s=0.")

        # Eq. (39b): A_ij(s)
        sig_s = self.diameters[:, None] * s
        lam_s = self.lambda_diag[:, None] * s
        A = (2.0 * np.pi * self.rho * self.x[:, None] / s**3) * (
            self.L0 * self._phi2(sig_s)
            + self.L1 * s * self._phi1(sig_s)
            + self.Lb0 * self._phi2(lam_s)
            + self.Lb1 * s * self._phi1(lam_s)
        )

        denominator = np.eye(self.n, dtype=complex) + A

        # Eq. (30): L_ij(s) and Lbar_ij(s)
        Ls = self.L0 + self.L1 * s
        Lbs = self.Lb0 + self.Lb1 * s

        # Eq. (39a).  Solve linear systems rather than forming an explicit inverse.
        GG0 = np.linalg.solve(denominator.T, Ls.T).T
        GG0b = np.linalg.solve(denominator.T, Lbs.T).T

        self._gg_cache[key] = (GG0, GG0b)
        return GG0, GG0b

    def _laplace_component(self, s: complex, i: int, j: int, barred: bool) -> complex:
        GG0, GG0b = self._gg_matrices(s)
        matrix = GG0b if barred else GG0
        return matrix[i, j] / s**2

    def _euler_ilt( self, t: float, i: int, j: int, *, barred: bool, mode: Literal["euler", "euler_plus"] = "euler",) -> float:
        """Numerical inverse Laplace transform used in the notebook."""

        if t <= 0.0:
            raise ValueError("Euler inverse Laplace transform requires t > 0.")

        if mode == "euler":
            aa, ntr, m_euler = 19.1, 15, 11
        elif mode == "euler_plus":
            # EulerILTplus is slower but more accurate than EulerILT.
            aa, ntr, m_euler = 30.0, 100, 30
        else:
            raise ValueError("mode must be 'euler' or 'euler_plus'.")

        hh = np.pi / t
        xx = aa / (2.0 * t)
        prefactor = np.exp(aa / 2.0) / t

        partial = 0.5 * np.real(self._laplace_component(xx, i, j, barred))
        for nn in range(1, ntr + 1):
            s = xx + 1j * nn * hh
            partial += (-1.0 if nn % 2 else 1.0) * np.real(self._laplace_component(s, i, j, barred))

        # Euler acceleration of the alternating tail.
        weighted_sum = comb(m_euler, 0) * partial
        for k in range(1, m_euler + 1):
            nn = ntr + k
            s = xx + 1j * nn * hh
            partial += (-1.0 if nn % 2 else 1.0) * np.real(self._laplace_component(s, i, j, barred))
            weighted_sum += comb(m_euler, k) * partial

        return float(prefactor * weighted_sum / 2.0**m_euler)

    def g0_asymmetric(self, r: float, i: int, j: int, *, inversion: Literal["euler", "euler_plus"] = "euler",) -> float:
        """Inverse Laplace transform of G_ij^(0)(s): g_ij^(0)(r)."""

        self._require_solved()
        sigma = self.sigma[i, j]
        lam = self.lambda_[i, j]

        if r < sigma:
            return 0.0

        if r <= sigma + self.sigma_min:
            xi, _ = self._xi_pair_matrices(r - sigma)
            value = xi[i, j] / r

            if r <= lam:
                return float(np.real(value))

            _, xib_shift = self._xi_pair_matrices(r - lam)
            value += xib_shift[i, j] / r
            return float(np.real(value))

        value = self._euler_ilt(r - sigma, i, j, barred=False, mode=inversion) / r
        value += self._euler_ilt(r - lam, i, j, barred=True, mode=inversion) / r
        return float(value)

    # Symmetrization of g_ij^(0)(r)
    def g0(self, r: float, i: int, j: int, *, inversion: Literal["euler", "euler_plus"] = "euler",) -> float:
        gij = self.g0_asymmetric(r, i, j, inversion=inversion)
        if i == j:
            return gij
        gji = self.g0_asymmetric(r, j, i, inversion=inversion)
        return 0.5 * (gij + gji)

    # Eq. (34): Q_ij(r)
    def Q(self, r: float, i: int, j: int) -> float:
        self._require_solved()
        lam = self.lambda_[i, j]
        return float(self.mu[i, j] * r * (r - lam) / lam)

    # Eq. (7): y_ij^(0)(r), followed by the correction in the notebook.
    def y(self, r: float, i: int, j: int, *, inversion: Literal["euler", "euler_plus"] = "euler",) -> float:
        g0 = self.g0(r, i, j, inversion=inversion)
        if r <= self.lambda_[i, j]:
            y0 = self.exp_plus_beta_epsilon[i, j] * g0
        else:
            y0 = g0

        if self.sigma[i, j] <= r <= self.lambda_[i, j]:
            y0 += self.Q(r, i, j)
        return float(y0)

    # g_ij(r)
    def g(self, r: float, i: int, j: int, *, inversion: Literal["euler", "euler_plus"] = "euler",) -> float:
        g0 = self.g0(r, i, j, inversion=inversion)
        if self.sigma[i, j] <= r <= self.lambda_[i, j]:
            g0 += self.exp_minus_beta_epsilon[i, j] * self.Q(r, i, j)
        return float(g0)

    def g_and_y(self, r: float, i: int, j: int, *, inversion: Literal["euler", "euler_plus"] = "euler",) -> tuple[float, float]:
        """Evaluate g_ij(r) and y_ij(r) together, reusing the same g0 calculation."""

        g0 = self.g0(r, i, j, inversion=inversion)
        in_shell = self.sigma[i, j] <= r <= self.lambda_[i, j]

        correction = self.Q(r, i, j) if in_shell else 0.0

        g_value = g0 + self.exp_minus_beta_epsilon[i, j] * correction

        y_value = (
            self.exp_plus_beta_epsilon[i, j] * g0
            if r <= self.lambda_[i, j]
            else g0
        )
        y_value += correction

        return float(g_value), float(y_value)

    # ------------------------------------------------------------------
    # htilde_ij(q)
    # ------------------------------------------------------------------

    def _qtilde(self, q: float) -> Array:
        """Fourier transform of Q_ij(r), vectorized over all species pairs."""

        if q <= 0.0:
            raise ValueError("q must be > 0 for the implemented Qtilde expression.")

        lam = self.lambda_
        sig = self.sigma
        term = (
            4.0 * q * lam * np.cos(q * lam)
            + q
            * (2.0 * lam - 6.0 * sig - q**2 * lam * sig**2 + q**2 * sig**3)
            * np.cos(q * sig)
            + (-6.0 + q**2 * lam**2) * np.sin(q * lam)
            + (6.0 + q**2 * (2.0 * lam - 3.0 * sig) * sig) * np.sin(q * sig)
        )
        return 4.0 * np.pi * self.mu * term / (q**5 * lam)

    def htilde_matrix(self, q: float) -> Array:
        """Eq. (6) plus Eq. (34) correction, symmetrized in i <-> j."""

        self._require_solved()
        if q <= 0.0:
            raise ValueError("q must be strictly positive.")

        s = 1j * q
        GG0, GG0b = self._gg_matrices(s)

        # Eq. (6): htilde_ij^(0)(q), before symmetrization.
        raw = -4.0 * np.pi * np.real(( np.exp(-self.sigma * s) * GG0 + np.exp(-self.lambda_ * s) * GG0b - 1.0) / s**3)
        h0 = 0.5 * (raw + raw.T)

        # htilde_ij(q)
        return h0 + self.exp_minus_beta_epsilon * self._qtilde(q)

    def htilde(self, q: float, i: int, j: int) -> float:
        return float(self.htilde_matrix(q)[i, j])
