"""Run the RFA square-well / square-shoulder mixture calculation.

Input can be supplied in two ways:

    python run_rfa.py input.dat

reads the parameters from ``input.dat``, while

    python run_rfa.py

asks for the same information interactively from the command line.

The calculation writes four files derived from the output base name:

    <output>_solver.dat
    <output>_g.dat
    <output>_y.dat
    <output>_fourier.dat

For g(r), y(r), and htilde(q), the first column is the independent variable
(r or q) and every following column corresponds to one selected particle pair.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rfa_sw_mixture import RFAMixture


@dataclass
class RunConfig:
    mole_fractions: np.ndarray
    diameters: np.ndarray
    epsilon: np.ndarray
    delta: float
    rho: float
    beta: float
    rmax: float = 9.0
    dr: float = 0.1
    qmin: float = 0.01
    qmax: float = 50.0
    dq: float = 0.1
    inversion: str = "euler"
    pairs: list[tuple[int, int]] | None = None


# ============================================================================
# INPUT FROM input.dat
# ============================================================================


def _numbers(text: str) -> list[float]:
    """Convert a whitespace/comma-separated line to floating-point values."""
    return [float(item) for item in text.replace(",", " ").split()]


def read_input(path: Path) -> RunConfig:
    """Read the input.dat format."""

    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, str] = {}
    epsilon_rows: list[list[float]] = []
    reading_epsilon = False

    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if reading_epsilon:
            if line.lower() == "end":
                reading_epsilon = False
            else:
                epsilon_rows.append(_numbers(line))
            continue

        if "=" not in line:
            raise ValueError(f"Cannot parse line: {raw_line!r}")

        key, value = (part.strip() for part in line.split("=", 1))
        key = key.lower()
        if key == "epsilon":
            if value:
                raise ValueError("Write epsilon as a matrix on the following lines, ending with 'end'.")
            reading_epsilon = True
        else:
            values[key] = value

    if reading_epsilon:
        raise ValueError("epsilon matrix is missing its terminating 'end' line.")

    x = np.asarray(_numbers(values["mole_fractions"]), dtype=float)
    diameters = np.asarray(_numbers(values["diameters"]), dtype=float)
    epsilon = np.asarray(epsilon_rows, dtype=float)
    n = x.size

    if "n_species" in values and int(values["n_species"]) != n:
        raise ValueError("n_species does not match the length of mole_fractions.")

    pairs: list[tuple[int, int]] | None = None
    pair_text = values.get("pairs", "all").strip().lower()
    if pair_text != "all":
        pairs = []
        for token in pair_text.replace(",", " ").split():
            pieces = token.replace(":", "-").split("-")
            if len(pieces) != 2:
                raise ValueError("pairs must be 'all' or entries such as 1-1 1-2 2-2.")
            i, j = int(pieces[0]), int(pieces[1])
            if not (1 <= i <= n and 1 <= j <= n):
                raise ValueError(f"Pair {token} is outside 1..{n}.")
            i, j = min(i, j), max(i, j)
            pairs.append((i - 1, j - 1))
        pairs = sorted(set(pairs))

    config = RunConfig(
        mole_fractions=x,
        diameters=diameters,
        epsilon=epsilon,
        delta=float(values["delta"]),
        rho=float(values["rho"]),
        beta=float(values["beta"]),
        rmax=float(values.get("rmax", 9.0)),
        dr=float(values.get("dr", 0.1)),
        qmin=float(values.get("qmin", 0.01)),
        qmax=float(values.get("qmax", 50.0)),
        dq=float(values.get("dq", 0.1)),
        inversion=values.get("inversion", "euler").strip().lower(),
        pairs=pairs,
    )

    _validate_config(config)
    return config


# ============================================================================
# INTERACTIVE COMMAND-LINE INPUT
# ============================================================================


def _read_int(
    prompt: str,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    """Read an integer interactively from the command line."""

    while True:
        suffix = f" [default: {default}]" if default is not None else ""
        text = input(f"{prompt}{suffix}: ").strip()

        if text == "" and default is not None:
            value = default
        else:
            try:
                value = int(text)
            except ValueError:
                print("Please enter an integer.")
                continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue

        return value


def _read_float(
    prompt: str,
    default: float | None = None,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Read a floating-point number interactively."""

    while True:
        suffix = f" [default: {default}]" if default is not None else ""
        text = input(f"{prompt}{suffix}: ").strip()

        if text == "" and default is not None:
            value = float(default)
        else:
            try:
                # Accept either decimal point or decimal comma for a scalar.
                value = float(text.replace(",", "."))
            except ValueError:
                print("Please enter a valid number.")
                continue

        if positive and value <= 0.0:
            print("Please enter a positive value.")
            continue

        if nonnegative and value < 0.0:
            print("Please enter a non-negative value.")
            continue

        return value


def _read_vector(
    prompt: str,
    n: int,
    default: list[float] | None = None,
) -> np.ndarray:
    """Read a vector containing exactly n floating-point values."""

    while True:
        if default is not None:
            default_text = " ".join(str(v) for v in default)
            suffix = f" [default: {default_text}]"
        else:
            suffix = ""

        text = input(f"{prompt}{suffix}: ").strip()

        if text == "" and default is not None:
            return np.asarray(default, dtype=float)

        # Accept common vector delimiters. Semicolons always separate values.
        normalized = text.strip()
        for char in "[](){}":
            normalized = normalized.replace(char, " ")
        normalized = normalized.replace(";", " ")

        # Support decimal commas when values are otherwise separated by spaces,
        # e.g. "0,7 0,2 0,1". Other commas are treated as separators.
        normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
        normalized = normalized.replace(",", " ")

        try:
            values = [float(item) for item in normalized.split()]
        except ValueError:
            print("Please enter numbers separated by spaces or semicolons.")
            continue

        if len(values) != n:
            print(f"Please enter exactly {n} values.")
            continue

        return np.asarray(values, dtype=float)


def _read_choice(
    prompt: str,
    choices: tuple[str, ...],
    default: str,
) -> str:
    """Read one value from a fixed set of choices."""

    choices_text = "/".join(choices)

    while True:
        text = input(
            f"{prompt} ({choices_text}) [default: {default}]: "
        ).strip().lower()

        if text == "":
            return default

        if text in choices:
            return text

        print(f"Please choose one of: {choices_text}")


def _read_pairs(n: int) -> list[tuple[int, int]] | None:
    """
    Read the particle pairs to calculate.

    Returns None for all unique pairs. Species indices entered by the user are
    1-based and are converted internally to 0-based indices.
    """

    while True:
        text = input(
            "Particle pairs [default: all]\n"
            "  Use 'all', or for example: 1-1 1-2 2-2\n"
            "pairs: "
        ).strip().lower()

        if text == "" or text == "all":
            return None

        pairs: list[tuple[int, int]] = []

        try:
            for token in text.replace(",", " ").split():
                pieces = token.replace(":", "-").split("-")

                if len(pieces) != 2:
                    raise ValueError

                i = int(pieces[0])
                j = int(pieces[1])

                if not (1 <= i <= n and 1 <= j <= n):
                    raise ValueError

                # Store only the unique ordering i <= j.
                i, j = min(i, j), max(i, j)
                pairs.append((i - 1, j - 1))

        except ValueError:
            print(
                f"Invalid pair specification. Species indices must be between 1 and {n}."
            )
            continue

        return sorted(set(pairs))


def read_input_interactive() -> RunConfig:
    """
    Ask for the RFA input parameters interactively.

    The returned RunConfig is the same type as the one produced by read_input(),
    so all solver and output routines are independent of how the data were read.
    """

    print("=" * 72)
    print("RFA square-well / square-shoulder mixture")
    print("Interactive input")
    print("Press Enter to accept a displayed default value.")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Number of species
    # ------------------------------------------------------------------
    n = _read_int(
        "Number of species",
        default=3,
        minimum=1,
    )

    # Use the original ternary case study as the default when appropriate.
    if n == 3:
        default_x = [0.7, 0.2, 0.1]
        default_diameters = [1.0, 1.5, 2.0]
    else:
        default_x = None
        default_diameters = None

    # ------------------------------------------------------------------
    # Mole fractions
    # ------------------------------------------------------------------
    while True:
        mole_fractions = _read_vector("Mole fractions x_i", n, default=default_x,)

        if np.any(mole_fractions < 0.0):
            print("Mole fractions cannot be negative.")
            continue

        xsum = float(np.sum(mole_fractions))
        if not np.isclose(xsum, 1.0, rtol=0.0, atol=1.0e-10):
            print(
                "Mole fractions must sum to 1. "
                f"Current sum = {xsum:.15g}"
            )
            continue

        break

    # Hard-core diameters
    while True:
        diameters = _read_vector("Diameters sigma_i", n, default=default_diameters,)
        if np.any(diameters <= 0.0):
            print("All diameters must be positive.")
            continue
        break

    # Common shell width
    delta = _read_float("Common shell width Delta", default=0.1, nonnegative=True,)

    # epsilon_ij matrix
    print()
    print(f"Enter the symmetric {n} x {n} epsilon_ij matrix.")
    print("Enter one complete row at a time.")

    if n == 3:
        default_epsilon = [[-1.0, -0.9, -0.8], [-0.9, -0.7, -0.6], [-0.8, -0.6, -0.5],]
    else:
        default_epsilon = None

    while True:
        epsilon_rows = []

        for i in range(n):
            default_row = (
                default_epsilon[i]
                if default_epsilon is not None
                else None
            )
            row = _read_vector(f"epsilon row {i + 1}", n, default=default_row,)
            epsilon_rows.append(row)

        epsilon = np.asarray(epsilon_rows, dtype=float)

        if not np.allclose(epsilon, epsilon.T, rtol=0.0, atol=1.0e-10):
            print("\nepsilon must be symmetric: epsilon_ij = epsilon_ji.")
            print("Please enter the matrix again.\n")
            continue

        break

    # Thermodynamic state
    rho = _read_float("Number density rho", default=0.2, nonnegative=True,)
    beta = _read_float("Inverse temperature beta", default=1.0,)

    # Real-space output grid
    print("\nReal-space grid")
    dr = _read_float("Grid spacing dr", default=0.1, positive=True,)
    while True:
        rmax = _read_float("Maximum r", default=9.0, positive=True,)
        if rmax < float(np.min(diameters)):
            print("rmax must be at least as large as the smallest diameter.")
            continue
        break

    # Fourier-space output grid
    print("\nFourier-space grid")
    qmin = _read_float("Minimum q", default=0.01, positive=True,)
    dq = _read_float("Grid spacing dq", default=0.1, positive=True,)
    while True:
        qmax = _read_float("Maximum q", default=50.0, positive=True,)
        if qmax < qmin:
            print("qmax must be greater than or equal to qmin.")
            continue
        break

    # Numerical inverse Laplace transform
    print()
    inversion = _read_choice("Inverse Laplace method", choices=("euler", "euler_plus"), default="euler",)

    # Particle pairs
    print()
    pairs = _read_pairs(n)

    config = RunConfig(
        mole_fractions=mole_fractions,
        diameters=diameters,
        epsilon=epsilon,
        delta=delta,
        rho=rho,
        beta=beta,
        rmax=rmax,
        dr=dr,
        qmin=qmin,
        qmax=qmax,
        dq=dq,
        inversion=inversion,
        pairs=pairs,
    )

    _validate_config(config)

    print()
    print("=" * 72)
    print("Input complete.")
    print("=" * 72)
    print()

    return config


# ============================================================================
# COMMON INPUT VALIDATION
# ============================================================================


def _validate_config(config: RunConfig) -> None:
    """Validate parameters that are common to file and interactive input."""

    n = config.mole_fractions.size

    if n < 1:
        raise ValueError("At least one species is required.")
    if config.diameters.shape != (n,):
        raise ValueError("diameters must have the same length as mole_fractions.")
    if config.epsilon.shape != (n, n):
        raise ValueError(f"epsilon must be a {n} x {n} matrix.")
    if np.any(config.mole_fractions < 0.0):
        raise ValueError("Mole fractions cannot be negative.")
    if not np.isclose(np.sum(config.mole_fractions), 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("Mole fractions must sum to 1.")
    if np.any(config.diameters <= 0.0):
        raise ValueError("All diameters must be positive.")
    if config.delta < 0.0:
        raise ValueError("delta must be non-negative.")
    if config.rho < 0.0:
        raise ValueError("rho must be non-negative.")
    if not np.allclose(config.epsilon, config.epsilon.T, atol=1.0e-10, rtol=0.0):
        raise ValueError("epsilon must be symmetric.")
    if config.dr <= 0.0 or config.dq <= 0.0:
        raise ValueError("dr and dq must be positive.")
    if config.qmin <= 0.0:
        raise ValueError("qmin must be > 0.")
    if config.rmax <= 0.0 or config.qmax < config.qmin:
        raise ValueError("Invalid r/q output range.")
    if config.inversion not in {"euler", "euler_plus"}:
        raise ValueError("inversion must be 'euler' or 'euler_plus'.")


# ============================================================================
# OUTPUT
# ============================================================================


def grid(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive floating-point grid without cumulative arange drift."""

    if start > stop:
        return np.empty(0, dtype=float)
    count = int(np.floor((stop - start) / step + 1.0e-12)) + 1
    return start + step * np.arange(count, dtype=float)


def _output_paths(path: Path) -> tuple[Path, Path, Path, Path]:
    """Return the four output filenames derived from the user-supplied name."""

    base = path.with_suffix("")
    solver_path = base.with_name(base.name + "_solver.dat")
    g_path = base.with_name(base.name + "_g.dat")
    y_path = base.with_name(base.name + "_y.dat")
    fourier_path = base.with_name(base.name + "_fourier.dat")
    return solver_path, g_path, y_path, fourier_path


def write_output(path: Path, model: RFAMixture, config: RunConfig) -> tuple[Path, Path, Path, Path]:
    """
    Write solver information, g(r), y(r), and Fourier data to separate files.

    The first column of each numerical data file is r or q. Each following
    column contains one selected particle pair.
    """

    n = model.n
    pairs = config.pairs
    if pairs is None:
        # Unique index pairs: g, y, and htilde are symmetric in i <-> j.
        pairs = [(i, j) for i in range(n) for j in range(i, n)]

    solver_path, g_path, y_path, fourier_path = _output_paths(path)

    # Ensure that a user-specified output directory exists.
    solver_path.parent.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. SOLVER INFORMATION
    # ========================================================================
    with solver_path.open("w", encoding="utf-8") as out:
        out.write("# RFA for square-well / square-shoulder mixtures\n")
        out.write("# Python translation of RFA_SW_mixt_annotated_Faster.nb\n")
        out.write("# Species indices in this file are 1-based.\n\n")

        out.write("# [INPUT]\n")
        out.write(f"# n_species = {n}\n")
        out.write(
            "# mole_fractions = "
            + " ".join(f"{v:.15g}" for v in model.x)
            + "\n"
        )
        out.write(
            "# diameters = "
            + " ".join(f"{v:.15g}" for v in model.diameters)
            + "\n"
        )
        out.write(f"# delta = {model.delta:.15g}\n")
        out.write(f"# rho = {model.rho:.15g}\n")
        out.write(f"# beta = {model.beta:.15g}\n")
        out.write(f"# packing_fraction = {model.packing_fraction:.15g}\n")
        out.write(f"# inversion = {config.inversion}\n")
        out.write("# epsilon matrix:\n")
        for row in model.epsilon:
            out.write("#   " + " ".join(f"{v:.15g}" for v in row) + "\n")
        out.write("\n")

        info = model.solver_info
        if info is None:
            raise RuntimeError("The model must be solved before writing output.")

        out.write("# [SOLVER]\n")
        out.write(f"# success = {info.success}\n")
        out.write(f"# message = {info.message}\n")
        out.write(f"# function_evaluations = {info.nfev}\n")
        out.write(
            "# max_continuity_residual = "
            f"{info.max_continuity_residual:.15f}\n"
        )
        out.write(
            "# eq55_condition_number = "
            f"{info.eq55_condition_number:.15f}\n\n"
        )

        out.write("# [LAMBDA_J]\n")
        out.write("# columns: j Lambda_j\n")
        for j, value in enumerate(model.Lambda, start=1):
            out.write(f"{j:d} {value:.15f}\n")
        out.write("\n")

        out.write("# [COEFFICIENTS]\n")
        out.write(
            "# columns: i j sigma_ij lambda_ij "
            "L0_ij L1_ij Lbar0_ij Lbar1_ij mu_ij\n"
        )
        for i in range(n):
            for j in range(n):
                out.write(
                    f"{i + 1:d} {j + 1:d} "
                    f"{model.sigma[i, j]:.15f} "
                    f"{model.lambda_[i, j]:.15f} "
                    f"{model.L0[i, j]:.15f} "
                    f"{model.L1[i, j]:.15f} "
                    f"{model.Lb0[i, j]:.15f} "
                    f"{model.Lb1[i, j]:.15f} "
                    f"{model.mu[i, j]:.15f}\n"
                )

    # ========================================================================
    # 2. RADIAL DISTRIBUTION FUNCTION g_ij(r)
    # 3. CAVITY FUNCTION y_ij(r)
    # ========================================================================

    # All pair columns share a common r grid. Start at the smallest contact
    # distance among the selected pairs.
    rmin = min(model.sigma[i, j] for i, j in pairs)
    r_values = grid(rmin, config.rmax, config.dr)

    # Include sigma_ij and lambda_ij explicitly because these physical
    # boundaries do not necessarily lie exactly on the regular dr grid.
    boundaries: list[float] = []
    for i, j in pairs:
        sigma_ij = model.sigma[i, j]
        lambda_ij = model.lambda_[i, j]

        if rmin <= sigma_ij <= config.rmax:
            boundaries.append(float(sigma_ij))
        if rmin <= lambda_ij <= config.rmax:
            boundaries.append(float(lambda_ij))

    if boundaries:
        r_values = np.concatenate((r_values, np.asarray(boundaries, dtype=float)))

    # Remove numerical duplicates and restore increasing order.
    r_values = np.unique(np.round(r_values, decimals=14))
    r_values.sort()

    g_labels = [f"g_{i + 1}_{j + 1}" for i, j in pairs]
    y_labels = [f"y_{i + 1}_{j + 1}" for i, j in pairs]

    with (
        g_path.open("w", encoding="utf-8") as gout,
        y_path.open("w", encoding="utf-8") as yout,
    ):
        gout.write("# Radial distribution functions g_ij(r)\n")
        gout.write("# Species indices are 1-based.\n")
        gout.write("# Columns: r " + " ".join(g_labels) + "\n")

        yout.write("# Cavity functions y_ij(r)\n")
        yout.write("# Species indices are 1-based.\n")
        yout.write("# Columns: r " + " ".join(y_labels) + "\n")

        # Evaluate g and y together so the same g0 calculation is reused.
        for r in r_values:
            g_row = [float(r)]
            y_row = [float(r)]

            for i, j in pairs:
                if r < model.sigma[i, j] - 1.0e-12:
                    # A shared radial grid necessarily contains points below
                    # contact for some larger particle pairs. The physical
                    # hard-core values are exactly zero there.
                    g_value = 0.0
                    y_value = 0.0
                else:
                    g_value, y_value = model.g_and_y(
                        float(r),
                        i,
                        j,
                        inversion=config.inversion,
                    )

                g_row.append(g_value)
                y_row.append(y_value)

            gout.write(" ".join(f"{value:.15f}" for value in g_row) + "\n")
            yout.write(" ".join(f"{value:.15f}" for value in y_row) + "\n")

    # ========================================================================
    # 4. FOURIER-SPACE DATA
    # ========================================================================
    q_values = grid(config.qmin, config.qmax, config.dq)
    fourier_labels = [f"htilde_{i + 1}_{j + 1}" for i, j in pairs]

    with fourier_path.open("w", encoding="utf-8") as out:
        out.write("# Fourier transforms htilde_ij(q)\n")
        out.write("# Species indices are 1-based.\n")
        out.write("# Columns: q " + " ".join(fourier_labels) + "\n")

        for q in q_values:
            # Evaluate the complete htilde matrix once per q. This avoids
            # repeating the same matrix algebra independently for every pair.
            h_matrix = model.htilde_matrix(float(q))
            row = [float(q)] + [float(h_matrix[i, j]) for i, j in pairs]
            out.write(" ".join(f"{value:.15f}" for value in row) + "\n")

    return solver_path, g_path, y_path, fourier_path


# ============================================================================
# MAIN PROGRAM
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("RFA calculation for multicomponent square-well/square-shoulder mixtures.")
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="?",
        type=Path,
        help=("Optional input .dat file. If omitted, the parameters are requested interactively."),
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output.dat",
        type=Path,
        help=("Base output name. For example, '-o case.dat' creates case_solver.dat, case_g.dat, case_y.dat, and case_fourier.dat."),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    if args.input is None:
        config = read_input_interactive()
    else:
        config = read_input(args.input)

    # ------------------------------------------------------------------
    # Construct and solve the RFA model.
    # From this point onward, it makes no difference whether the data came
    # from an input file or from the interactive command line.
    # ------------------------------------------------------------------
    model = RFAMixture(
        config.mole_fractions,
        config.diameters,
        config.epsilon,
        config.delta,
        config.rho,
        config.beta,
    )

    info = model.solve_coefficients()

    output_files = write_output(
        args.output,
        model,
        config,
    )

    print(f"Solved {model.n}-species mixture in {info.nfev} nonlinear evaluations.")
    print(f"Max continuity residual: {info.max_continuity_residual:.3e}")
    print("Wrote:")
    for output_file in output_files:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()
