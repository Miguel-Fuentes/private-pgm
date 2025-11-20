from mbi import Dataset
import numpy as np
from scipy.optimize import minimize
import time
import pandas as pd
from typing import List, Tuple


class SparseMatrix:
    """
    A memory-efficient representation of a sparse matrix where each column has
    exactly one non-zero element (a '1').

    This structure is common in representing categorical data, where each data
    record (column) belongs to exactly one category (row) in a given marginal.
    It's optimized for fast matrix-vector products.

    Attributes:
        shape (tuple): The dimensions of the matrix (num_cells, num_records).
    """

    def __init__(self, dataset: Dataset, clique: Tuple[str, ...]):
        """
        Initializes the SparseMatrix.

        Args:
            dataset: The dataset object, containing the domain and records.
            clique: A tuple of attribute names that define this matrix's mapping.
        """
        self.domain = dataset.domain
        self.clique = clique
        self.num_records = dataset.records

        # Determine the shape of the matrix from the domain information
        clique_indices = [self.domain.attributes.index(attr) for attr in self.clique]
        levels = np.array([self.domain.shape[idx] for idx in clique_indices])
        num_cells = np.prod(levels)
        self.shape = (num_cells, self.num_records)

        # Pre-compute the index map for efficient matrix operations
        self._precompute_index_map(dataset.df, levels)

    def _precompute_index_map(self, df: "pd.DataFrame", levels: np.ndarray):
        """
        Calculates a 1D array where each entry is the row index corresponding
        to a data record (column). This is the core of the sparse representation.
        """
        data_values = df[list(self.clique)].values

        # Convert multi-dimensional indices to a flat 1D index
        if len(levels) > 1:
            strides = np.concatenate((np.cumprod(levels[1:][::-1])[::-1], [1]))
        else:
            strides = np.array([1])

        self.index_map = data_values @ strides.astype(int)

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs the matrix-vector product ($A cdot v$), also known as "scatter-add".
        It maps values from the vector (representing records) to the output
        array (representing marginal cells).

        Args:
            vector: A numpy array of length equal to the number of records.

        Returns:
            The resulting numpy array of length equal to the number of marginal cells.
        """
        assert vector.size == self.num_records, (
            f"Input vector size {vector.size} must match matrix columns {self.num_records}"
        )
        output = np.zeros(self.shape[0])
        np.add.at(output, self.index_map, vector)
        return output

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs the transpose-matrix-vector product ($A^T cdot v$), known as "gather".
        It gathers values from the vector (representing marginal cells) according
        to the record-to-cell mapping.

        Args:
            vector: A numpy array of length equal to the number of marginal cells.

        Returns:
            The resulting numpy array of length equal to the number of records.
        """
        assert vector.size == self.shape[0], (
            f"Input vector size {vector.size} must match matrix rows {self.shape[0]}"
        )
        return vector[self.index_map]

    def __repr__(self) -> str:
        return f"<SparseMatrix shape={self.shape} clique={self.clique}>"


def public_support(
    public_data: Dataset,
    cliques: List[Tuple[str, ...]],
    variances: List[float],
    measurements: List[np.ndarray],
    lambda_reg: float = 1e-5,
    max_iters: int = 2000,
    tol: float = 1e-8,
    verbose: bool = True,
    EPS=1e-15,
) -> Dataset:
    """
    Finds weights for a public dataset that best match a set of noisy measurements,
    using entropy regularization for smoothness.

    Uses log-space parameterization for optimization on the simplex.

    Args:
        public_data: The public dataset to be re-weighted.
        cliques: A list of attribute cliques corresponding to each measurement.
        variances: A list of variances for each noisy measurement.
        measurements: A list of the noisy marginal measurements.
        lambda_reg: Regularization parameter for entropy term.
        max_iters: Maximum number of optimization iterations.
        tol: Convergence tolerance.
        verbose: Whether to print progress information.

    Returns:
        A new Dataset object with the optimized weights.
    """
    start_time = time.time()
    n_variables = public_data.records

    if verbose:
        print(f"Problem size: {n_variables} variables")

    # Check for empty or near-zero measurements
    total = sum(np.sum(m) for m in measurements) / len(measurements)
    if total < 1e-9:
        if verbose:
            print(
                "Warning: Total of all measurements is near zero. Returning uniform weights."
            )
        uniform_weights = np.full(public_data.records, total / public_data.records)
        return Dataset(
            df=public_data.df, domain=public_data.domain, weights=uniform_weights
        )

    # Normalize measurements and variances
    norm_measurements = [m / total for m in measurements]
    norm_variances = [v / total**2 for v in variances]

    # Calculate inverse-variance weights for the loss function
    raw_term_weights = [1.0 / (v + 1e-12) for v in norm_variances]
    sum_raw_weights = sum(raw_term_weights)
    num_measurements = len(measurements)
    term_weights = [w * num_measurements / sum_raw_weights for w in raw_term_weights]

    # Set up sparse matrix operators
    sparse_matrices = [SparseMatrix(public_data, cl) for cl in cliques]

    # Scale only least squares terms for numerical stability
    measurement_norms = [np.linalg.norm(m) for m in norm_measurements]
    max_weighted_norm = max(
        w * norm for w, norm in zip(term_weights, measurement_norms)
    )

    # Normalize least squares terms
    ls_scale = 1.0 / max(max_weighted_norm, 1e-10)

    if verbose:
        print(f"Least squares scaling factor: {ls_scale:.2e}")
        print(f"Using lambda_reg value directly: {lambda_reg:.2e}")

    # Function to convert from log space to probability space (simplex)
    def log_to_prob(log_x):
        log_x_shifted = log_x - np.max(log_x)
        prob = np.exp(log_x_shifted)
        return prob / np.sum(prob)

    # Define objective function in log space
    def objective_func(log_x):
        # Convert to probability space
        x_prob = log_to_prob(log_x)

        # Least squares loss
        ls_loss = 0.0
        for M, y, weight in zip(sparse_matrices, norm_measurements, term_weights):
            Mx = M.matvec(x_prob)
            residual = Mx - y
            ls_loss += weight * np.sum(residual**2)

        ls_loss *= ls_scale

        # Entropy term in log space
        # For numerical stability, use the fact that x_prob is already normalized
        entropy = -np.sum(x_prob * np.log(np.maximum(x_prob, EPS)))
        entropy_term = lambda_reg * entropy

        total_obj = ls_loss + entropy_term
        return total_obj

    # Gradient in log space
    def gradient_func(log_x):
        # Convert to probability space
        x_prob = log_to_prob(log_x)

        # Compute standard gradient with respect to x_prob
        grad_prob = np.zeros_like(x_prob)

        # Gradient of least squares terms
        for M, y, weight in zip(sparse_matrices, norm_measurements, term_weights):
            Mx = M.matvec(x_prob)
            residual = Mx - y
            grad_term = M.rmatvec(residual) * 2.0 * weight * ls_scale
            grad_prob += grad_term

        # Gradient of entropy term with respect to x_prob
        entropy_grad_prob = -lambda_reg * (1.0 + np.log(np.maximum(x_prob, EPS)))
        grad_prob += entropy_grad_prob

        # Transform gradient from probability space to log space using chain rule
        # For softmax parameterization, the gradient transformation is:
        # ∂f/∂log_x_i = x_prob_i * (∂f/∂x_prob_i - sum_j(x_prob_j * ∂f/∂x_prob_j))
        mean_grad_prob = np.sum(x_prob * grad_prob)
        grad_log = x_prob * (grad_prob - mean_grad_prob)

        return grad_log

    # Progress monitoring
    iterations = 0
    best_obj = float("inf")
    last_print_time = time.time()
    objective_history = []

    def progress_callback(log_x):
        nonlocal iterations, best_obj, last_print_time
        iterations += 1

        obj_val = objective_func(log_x)
        objective_history.append(obj_val)

        if obj_val < best_obj:
            best_obj = obj_val

        # Print progress at regular intervals
        current_time = time.time()
        if verbose and (iterations % 20 == 0 or current_time - last_print_time > 10):
            last_print_time = current_time
            x_prob = log_to_prob(log_x)
            print(f"Iter {iterations}: obj={obj_val:.4e}")
            print(
                f"  Probabilities - Min: {np.min(x_prob):.2e}, Max: {np.max(x_prob):.2e}, Sum: {np.sum(x_prob):.8f}"
            )

            # Check for convergence by looking at recent progress
            if len(objective_history) >= 3:
                recent_progress = abs(
                    objective_history[-3] - objective_history[-1]
                ) / max(abs(objective_history[-3]), 1e-10)
                print(f"  Recent progress: {recent_progress:.2e}")

    # Initial point in log space (uniform distribution)
    log_x0 = np.zeros(n_variables) - np.log(n_variables)  # log(1/n) for each component

    if verbose:
        print("Starting L-BFGS optimization in log space...")
        print(
            "This approach naturally enforces the simplex constraint through parameterization"
        )

    # Optimization in log space (unconstrained)
    result = minimize(
        objective_func,
        log_x0,
        method="L-BFGS",
        jac=gradient_func,
        callback=progress_callback,
        options={
            "ftol": tol,
            "gtol": tol * 10,  # Slightly larger gradient tolerance
            "maxiter": max_iters,
            "maxcor": min(20, n_variables * 2),  # Memory parameter
            "maxfun": max_iters * 2,
        },
    )

    # Convert final solution back to probability space
    optimized_weights = log_to_prob(result.x)

    # Scale weights back to original total
    optimized_weights *= total

    time_taken = time.time() - start_time

    if verbose:
        print(f"\nOptimization complete in {time_taken:.2f} seconds")
        print(f"Final objective: {best_obj:.6e}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(
            f"Final weight statistics - Min: {np.min(optimized_weights):.6e}, "
            f"Max: {np.max(optimized_weights):.6e}, "
            f"Sum: {np.sum(optimized_weights):.6f}"
        )

    return Dataset(
        df=public_data.df, domain=public_data.domain, weights=optimized_weights
    )


def systematic_sample(df, weights, random_state=None):
    """
    Transforms a weighted pandas DataFrame into an unweighted one using
    systematic sampling.

    This method is a low-variance, unbiased technique that creates a new
    dataset whose size is equal to the rounded sum of the original weights.
    An item with weight `w` is guaranteed to be selected either floor(w) or
    ceil(w) times.

    Args:
        df (pd.DataFrame): The input DataFrame.
        weights (np.ndarray or pd.Series): A NumPy array or pandas Series
                                           containing the non-negative float
                                           weights. Its length cannot be
                                           greater than the number of rows
                                           in df.
        random_state (int, optional): Seed for the random number generator
                                      to ensure reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A new, unweighted DataFrame where rows have been
                      repeated according to the systematic sampling procedure.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if len(weights) > len(df):
        raise ValueError(
            "Length of weights vector cannot be greater than the number of rows in df."
        )
    if (np.array(weights) < 0).any():
        raise ValueError("Weights cannot be negative.")

    sum_of_weights = np.sum(weights)
    target_size = int(round(sum_of_weights))

    if target_size == 0:
        return df.iloc[[]].reset_index(drop=True)

    rng = np.random.RandomState(random_state)

    step = sum_of_weights / target_size
    start_point = rng.uniform(low=0.0, high=step)
    pointers = start_point + np.arange(target_size) * step

    cumulative_weights = np.cumsum(weights)

    # Use binary search to efficiently find which interval each pointer falls into
    selected_indices = np.searchsorted(cumulative_weights, pointers)

    unweighted_df = df.iloc[selected_indices]
    return unweighted_df.reset_index(drop=True)
