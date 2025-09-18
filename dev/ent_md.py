import numpy as np
import pandas as pd
from scipy.special import logsumexp, xlogy
from typing import Optional, Callable, Tuple, List
from mbi import Dataset


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

    def _precompute_index_map(self, df: 'pd.DataFrame', levels: np.ndarray):
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
        Performs the matrix-vector product ($A \cdot v$), also known as "scatter-add".
        It maps values from the vector (representing records) to the output
        array (representing marginal cells).

        Args:
            vector: A numpy array of length equal to the number of records.

        Returns:
            The resulting numpy array of length equal to the number of marginal cells.
        """
        assert vector.size == self.num_records, \
            f"Input vector size {vector.size} must match matrix columns {self.num_records}"
        output = np.zeros(self.shape[0])
        np.add.at(output, self.index_map, vector)
        return output

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs the transpose-matrix-vector product ($A^T \cdot v$), known as "gather".
        It gathers values from the vector (representing marginal cells) according
        to the record-to-cell mapping.

        Args:
            vector: A numpy array of length equal to the number of marginal cells.

        Returns:
            The resulting numpy array of length equal to the number of records.
        """
        assert vector.size == self.shape[0], \
            f"Input vector size {vector.size} must match matrix rows {self.shape[0]}"
        return vector[self.index_map]

    def __repr__(self) -> str:
        return f"<SparseMatrix shape={self.shape} clique={self.clique}>"


def compute_loss_and_grad(
    weights_vector: np.ndarray,
    operators: List[SparseMatrix],
    measurements: List[np.ndarray],
    term_weights: List[float]
) -> Tuple[float, np.ndarray]:
    """
    Computes the total weighted squared loss and its gradient.

    The loss is the sum of weighted squared differences between the true
    measurements ($y_i$) and the marginals computed from the current weights_vector ($w$).
    The formula is:
    $$
    L(w) = \sum_{i} \\alpha_i \| M_i w - y_i \|_2^2
    $$
    where $M_i$ are the operators and $\\alpha_i$ are the term_weights.

    Args:
        weights_vector: The optimization variable ($w$).
        operators: A list of SparseMatrix operators ($M_i$).
        measurements: A list of target marginal measurement vectors ($y_i$).
        term_weights: A list of weights for each term in the loss function ($\\alpha_i$).

    Returns:
        A tuple containing (total_loss, total_gradient).
    """
    total_loss = 0.0
    total_grad = np.zeros_like(weights_vector)

    for op, meas, weight in zip(operators, measurements, term_weights):
        # Calculate the estimated marginals from the current weights
        estimated_marginal = op.matvec(weights_vector)
        difference = estimated_marginal - meas

        # Accumulate the weighted loss and gradient
        total_loss += weight * np.sum(difference**2)
        total_grad += 2 * weight * op.rmatvec(difference)

    return total_loss, total_grad


class AcceleratedEMD:
    """
    An implementation of Accelerated Entropic Mirror Descent for optimization
    over the probability simplex. 🚀

    This optimizer incorporates several key features for robustness and speed:
    - **Nesterov-style acceleration** in the dual (log-weight) space.
    - **Backtracking line search** based on KL-divergence.
    - **Adaptive momentum clipping** to prevent overshooting.
    - **Unified restart condition** to reset momentum when progress falters.
    - **Stall detection** to switch off acceleration if it's no longer helping.
    - **Scale-invariant KKT gap** for a reliable convergence criterion.
    """

    def __init__(self,
                 loss_and_grad_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
                 dimension: int,
                 total_weight: float = 1.0,
                 step_size: float = 1.0,
                 grad_tol: float = 5e-3,
                 line_search_beta: float = 0.9,
                 step_increase_factor: float = 1.05,
                 max_step: float = 50.0,
                 stall_patience: int = 40,
                 stall_tolerance: float = 1e-4,
                 momentum_clip_factor: float = 2_000.0):
        """
        Initializes the optimizer.

        Args:
            loss_and_grad_fn: A function that takes a weight vector and returns (loss, gradient).
            dimension: The dimension of the weight vector.
            total_weight: The value the primal weights must sum to (e.g., 1.0).
            step_size: The initial step size for the line search.
            grad_tol: Relative tolerance for the KKT stationarity condition to determine convergence.
            line_search_beta: Factor to shrink step size during backtracking (must be < 1).
            step_increase_factor: Factor to grow step size after a successful step.
            max_step: An upper bound on the step size.
            stall_patience: Iterations to wait with insufficient progress before disabling acceleration.
            stall_tolerance: The relative loss change threshold to define a "stall".
            momentum_clip_factor: Limits momentum norm to be `clip_factor * gradient_norm`.
        """
        self.loss_and_grad_fn = loss_and_grad_fn
        self.dimension = dimension
        self.total_weight = total_weight
        self.initial_step_size = step_size
        self.grad_tol = grad_tol
        self.line_search_beta = line_search_beta
        self.step_increase_factor = step_increase_factor
        self.max_step = max_step
        self.stall_patience = stall_patience
        self.stall_tolerance = stall_tolerance
        self.momentum_clip_factor = momentum_clip_factor

        # --- Initialize algorithm state ---
        log_uniform = np.log(self.total_weight / self.dimension)
        self._log_weights = np.full(self.dimension, log_uniform)
        self._log_momentum = np.full(self.dimension, log_uniform)

        self._current_step_size = self.initial_step_size
        self._acceleration_counter = 0.0
        self._previous_loss = float('inf')
        self._is_accelerated = True
        self._stall_counter = 0

    def _map_to_primal(self, log_w: np.ndarray) -> np.ndarray:
        """Maps a dual log-weight vector to the primal probability simplex."""
        # Use logsumexp for numerical stability against underflow/overflow
        return np.exp(log_w - logsumexp(log_w)) * self.total_weight

    def run(self, max_iters: int = 2000, verbose_freq: int = 50) -> np.ndarray:
        """
        Runs the main optimization loop.

        Args:
            max_iters: The maximum number of iterations to perform.
            verbose_freq: The frequency at which to print progress updates.

        Returns:
            The final optimized weight vector.
        """
        print("--- Starting Accelerated Entropic Mirror Descent ---")
        for iteration in range(max_iters):
            # --- 1. Evaluate current point and check for convergence ---
            current_weights = self._map_to_primal(self._log_weights)
            current_loss, current_grad = self.loss_and_grad_fn(current_weights)

            stationarity_gap = self._check_convergence(current_weights, current_grad)
            if stationarity_gap < self.grad_tol and iteration > 0:
                print(f"\nConverged at iteration {iteration}: "
                      f"Rel. KKT gap ({stationarity_gap:.2e}) is below tolerance ({self.grad_tol}).")
                break

            # --- 2. Perform acceleration step ---
            theta = 2.0 / (self._acceleration_counter + 2.0) if self._is_accelerated else 1.0
            log_extrap_weights = (1.0 - theta) * self._log_weights + theta * self._log_momentum

            # --- 3. Control momentum and check for restart ---
            if self._is_accelerated and self._handle_restart(iteration, current_loss, current_grad, log_extrap_weights):
                continue # Skip to the next iteration if a restart occurred

            # --- 4. Evaluate extrapolated point for the update ---
            extrap_weights = self._map_to_primal(log_extrap_weights)
            extrap_loss, extrap_grad = self.loss_and_grad_fn(extrap_weights)

            # --- 5. Perform line search and update weights ---
            step_size, log_weights_next, log_momentum_next = self._line_search(
                extrap_loss, extrap_grad, extrap_weights, theta
            )

            if step_size < 1e-14:
                print("Warning: Step size collapsed to zero. Stopping.")
                break

            self._log_weights = log_weights_next
            self._log_momentum = log_momentum_next

            # --- 6. Log progress and update state for next iteration ---
            self._log_progress(iteration, verbose_freq, current_loss, stationarity_gap, np.linalg.norm(current_grad))
            self._update_state(iteration, current_loss, step_size)
        else:
            print(f"\nOptimization stopped: Maximum iterations ({max_iters}) reached.")

        print("--- Optimization finished. ---")
        return self._map_to_primal(self._log_weights)

    def _check_convergence(self, weights: np.ndarray, grad: np.ndarray) -> float:
        """
        Calculates the relative KKT stationarity gap. A small gap indicates
        that the solution is close to optimal.
        """
        # Consider only gradients for non-zero weights
        active_grad = grad[weights > 1e-12]
        if active_grad.size == 0:
            return 0.0

        # KKT condition for the simplex is that all active gradients are equal.
        # We measure the gap between the max and min active gradients.
        absolute_gap = np.max(active_grad) - np.min(active_grad)
        avg_grad_magnitude = np.mean(np.abs(active_grad))

        # Normalize by the average magnitude to make it scale-invariant
        return absolute_gap / (avg_grad_magnitude + 1e-9)

    def _handle_restart(self, iteration: int, current_loss: float, current_grad: np.ndarray, log_extrap_weights: np.ndarray) -> bool:
        """
        Checks restart conditions and resets acceleration if needed. A restart
        is triggered if momentum is pointing in a bad direction or if the
        loss unexpectedly increases.
        """
        # Clip momentum to prevent it from growing excessively large relative to the gradient
        momentum_direction = self._log_momentum - self._log_weights
        grad_norm = np.linalg.norm(current_grad)
        adaptive_threshold = self.momentum_clip_factor * (grad_norm + 1e-9)
        mom_norm = np.linalg.norm(momentum_direction)
        if mom_norm > adaptive_threshold:
            momentum_direction *= adaptive_threshold / mom_norm
            self._log_momentum = self._log_weights + momentum_direction

        # Proactive check: Is momentum pointing against the gradient at the extrapolated point?
        _, extrap_grad = self.loss_and_grad_fn(self._map_to_primal(log_extrap_weights))
        proactive_check = np.dot(extrap_grad, momentum_direction) > 0

        # Reactive check: Did the loss at the main iterate increase?
        reactive_check = current_loss > self._previous_loss and iteration > 0

        if proactive_check or reactive_check:
            reason = "extrapolated gradient" if proactive_check else "loss increase"
            print(f"--- Restart at iteration {iteration} (reason: {reason}) ---")
            self._log_momentum = np.copy(self._log_weights)
            self._acceleration_counter = 0.0
            self._stall_counter = 0
            self._current_step_size = self.initial_step_size
            self._previous_loss = float('inf')
            return True
        return False

    def _line_search(self, extrap_loss: float, extrap_grad: np.ndarray,
                     extrap_weights: np.ndarray, theta: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Performs backtracking line search using the KL-divergence geometry.

        This method seeks a step size $\\eta$ that satisfies the following
        condition, which guarantees sufficient decrease for mirror descent:
        $$
        f(x_{k+1}) \\le f(y_k) + \\langle \\nabla f(y_k), x_{k+1} - y_k \\rangle + \\frac{1}{\\eta} D_{KL}(x_{k+1} \\| y_k)
        $$
        where $f$ is the loss, $y_k$ is the extrapolated point, $x_{k+1}$ is the next
        point, and $D_{KL}$ is the KL-divergence.
        """
        step_size = self._current_step_size
        while True:
            # --- Perform a mirror descent update step for the momentum sequence ---
            log_momentum_next_unproj = self._log_momentum - step_size * extrap_grad
            log_momentum_next = log_momentum_next_unproj - logsumexp(log_momentum_next_unproj) + np.log(self.total_weight)

            # --- Update the main iterate via the same momentum coefficient ---
            log_weights_next = (1.0 - theta) * self._log_weights + theta * log_momentum_next

            # --- Check if the line search condition is met ---
            next_weights = self._map_to_primal(log_weights_next)
            new_loss, _ = self.loss_and_grad_fn(next_weights)

            # This bound is derived from the properties of mirror descent
            kl_div = np.sum(xlogy(next_weights, next_weights) - xlogy(next_weights, extrap_weights))
            upper_bound = extrap_loss + np.dot(extrap_grad, next_weights - extrap_weights) + (1.0 / step_size) * kl_div

            if new_loss <= upper_bound:
                return step_size, log_weights_next, log_momentum_next

            step_size *= self.line_search_beta
            if step_size < 1e-14: # Step size has collapsed
                return step_size, self._log_weights, self._log_momentum

    def _update_state(self, iteration: int, current_loss: float, last_step_size: float):
        """Updates the optimizer's internal state at the end of an iteration."""
        # Center log-weights to prevent them from drifting to +/- infinity
        shift = np.mean(self._log_weights)
        self._log_weights -= shift
        self._log_momentum -= shift

        # Check for stalls
        if self._is_accelerated:
            relative_change = abs(self._previous_loss - current_loss) / (abs(self._previous_loss) + 1e-9)
            if relative_change < self.stall_tolerance:
                self._stall_counter += 1
            else:
                self._stall_counter = 0

            if self._stall_counter >= self.stall_patience:
                print(f"--- Switching to non-accelerated phase at iteration {iteration} (progress stalled) ---")
                self._is_accelerated = False
                self._current_step_size = self.initial_step_size # Reset step size

        # Update counters and step size for the next iteration
        self._previous_loss = current_loss
        if self._is_accelerated:
            self._acceleration_counter += 1.0

        self._current_step_size = min(last_step_size * self.step_increase_factor, self.max_step)

    def _log_progress(self, iteration: int, freq: int, loss: float, kkt_gap: float, grad_norm: float):
        """Prints a summary of the current optimization state."""
        if iteration % freq == 0:
            accel_status = "ON" if self._is_accelerated else "OFF"
            mom_norm = np.linalg.norm(self._log_momentum - self._log_weights)
            print(f"Iter {iteration:4d}: Loss = {loss:.6f}, Step = {self._current_step_size:.4f}, "
                  f"Accel = {accel_status}, Rel. KKT Gap = {kkt_gap:.2e}, "
                  f"Mom. Norm = {mom_norm:.2f}, Grad. Norm = {grad_norm:.2f}")


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
        raise ValueError("Length of weights vector cannot be greater than the number of rows in df.")
    if (np.array(weights) < 0).any():
        raise ValueError("Weights cannot be negative.")

    sum_of_weights = np.sum(weights)
    target_size = int(round(sum_of_weights))

    if target_size == 0:
        return df.iloc[[]].reset_index(drop=True)

    rng = np.random.RandomState(random_state)

    # To ensure the number of samples is exactly target_size, we generate
    # that many pointers, spaced evenly across the sum of weights.
    step = sum_of_weights / target_size
    start_point = rng.uniform(low=0.0, high=step)
    pointers = start_point + np.arange(target_size) * step

    cumulative_weights = np.cumsum(weights)

    # Use binary search to efficiently find which interval each pointer falls into
    selected_indices = np.searchsorted(cumulative_weights, pointers)

    unweighted_df = df.iloc[selected_indices]
    return unweighted_df.reset_index(drop=True)


def public_support_old(public_data: Dataset,
                   cliques: List[Tuple[str, ...]],
                   variances: List[float],
                   measurements: List[np.ndarray],
                   max_iters: int = 1000) -> Dataset:
    """
    Finds weights for a public dataset that best match a set of noisy measurements.

    Args:
        public_data: The public dataset to be re-weighted.
        cliques: A list of attribute cliques corresponding to each measurement.
        variances: A list of variances for each noisy measurement.
        measurements: A list of the noisy marginal measurements.
        max_iters: Maximum number of optimization iterations.

    Returns:
        A new Dataset object with the optimized weights.
    """

    total = sum(np.sum(m) for m in measurements) / len(measurements)

    if total < 1e-9:
        print("Warning: Total of all measurements is near zero. Returning uniform weights.")
        uniform_weights = np.full(public_data.records, total / public_data.records)
        return Dataset(df=public_data.df, domain=public_data.domain, weights=uniform_weights)

    norm_measurements = [m / total for m in measurements]
    norm_variances = [v / total**2 for v in variances]

    # Calculate inverse-variance weights for the loss function.
    # Add a small epsilon for stability if variance is zero.
    raw_term_weights = [1.0 / (v + 1e-12) for v in norm_variances]

    # Stabilize term weights by normalizing them to have an average of 1.
    sum_raw_weights = sum(raw_term_weights)
    num_measurements = len(measurements)
    term_weights = [w * num_measurements / sum_raw_weights for w in raw_term_weights]

    # Set up the optimization problem
    operators = [SparseMatrix(public_data, cl) for cl in cliques]
    loss_fn = lambda w: compute_loss_and_grad(w, operators, norm_measurements, term_weights)

    optimizer = AcceleratedEMD(
        loss_and_grad_fn=loss_fn,
        dimension=public_data.records,
        total_weight=1.0  # We operate on the normalized simplex
    )

    # Run the optimization to find the weights
    optimized_weights = optimizer.run(max_iters)

    # Scale the final weights back to the original total
    optimized_weights *= total
    return Dataset(df=public_data.df, domain=public_data.domain, weights=optimized_weights)


def public_support(public_data: Dataset,
                   cliques: List[Tuple[str, ...]],
                   variances: List[float],
                   measurements: List[np.ndarray],
                   lambda_reg: float = 1e-5,
                   max_iters: int = 1000,
                   tol: float = 1e-8,
                   verbose: bool = True) -> Dataset:
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
            print("Warning: Total of all measurements is near zero. Returning uniform weights.")
        uniform_weights = np.full(public_data.records, total / public_data.records)
        return Dataset(df=public_data.df, domain=public_data.domain, weights=uniform_weights)

    # Normalize measurements and variances
    norm_measurements = [m / total for m in measurements]
    norm_variances = [v / total**2 for v in variances]

    # Calculate inverse-variance weights for the loss function
    raw_term_weights = [1.0 / (v + 1e-12) for v in norm_variances]
    sum_raw_weights = sum(raw_term_weights)
    num_measurements = len(measurements)
    term_weights = [w * num_measurements / sum_raw_weights for w in raw_term_weights]

    # Set up sparse matrix operators
    sparse_matrices = [ent_md.SparseMatrix(public_data, cl) for cl in cliques]

    # Scale only least squares terms for numerical stability
    measurement_norms = [np.linalg.norm(m) for m in norm_measurements]
    max_weighted_norm = max(w * norm for w, norm in zip(term_weights, measurement_norms))

    # Normalize least squares terms
    ls_scale = 1.0 / max(max_weighted_norm, 1e-10)

    if verbose:
        print(f"Least squares scaling factor: {ls_scale:.2e}")
        print(f"Using lambda_reg value directly: {lambda_reg:.2e}")

    # Small constant to prevent numerical issues
    EPS = 1e-15

    # Function to convert from log space to probability space (simplex)
    def log_to_prob(log_x):
        # Shift for numerical stability (prevent overflow)
        log_x_shifted = log_x - np.max(log_x)
        prob = np.exp(log_x_shifted)
        # Normalize to ensure sum = 1
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
        entropy_term = lambda_reg * entropy  # Note: sign flipped because entropy is negative

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
        entropy_grad_prob = lambda_reg * (1.0 + np.log(np.maximum(x_prob, EPS)))
        grad_prob += entropy_grad_prob

        # Transform gradient from probability space to log space using chain rule
        # We need to compute ∂f/∂log_x = (∂f/∂x_prob) * (∂x_prob/∂log_x)
        # The Jacobian ∂x_prob/∂log_x has a special structure due to the softmax

        # More efficient way to compute this transformation:
        # For softmax parameterization, the gradient transformation is:
        # ∂f/∂log_x_i = x_prob_i * (∂f/∂x_prob_i - sum_j(x_prob_j * ∂f/∂x_prob_j))
        mean_grad_prob = np.sum(x_prob * grad_prob)
        grad_log = x_prob * (grad_prob - mean_grad_prob)

        return grad_log

    # Progress monitoring
    iterations = 0
    best_obj = float('inf')
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
            print(f"  Probabilities - Min: {np.min(x_prob):.2e}, Max: {np.max(x_prob):.2e}, Sum: {np.sum(x_prob):.8f}")

            # Check for convergence by looking at recent progress
            if len(objective_history) >= 3:
                recent_progress = abs(objective_history[-3] - objective_history[-1]) / max(abs(objective_history[-3]), 1e-10)
                print(f"  Recent progress: {recent_progress:.2e}")

    # Initial point in log space (uniform distribution)
    log_x0 = np.zeros(n_variables) - np.log(n_variables)  # log(1/n) for each component

    if verbose:
        print("Starting L-BFGS optimization in log space...")
        print("This approach naturally enforces the simplex constraint through parameterization")

    # Optimization in log space (unconstrained)
    result = minimize(
        objective_func,
        log_x0,
        method='L-BFGS-B',  # Could use regular L-BFGS but L-BFGS-B handles bounds better if needed
        jac=gradient_func,
        callback=progress_callback,
        options={
            'ftol': tol,
            'gtol': tol * 10,  # Slightly larger gradient tolerance
            'maxiter': max_iters,
            'maxcor': min(20, n_variables),  # Memory parameter
            'maxfun': max_iters * 2
        }
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
        print(f"Final weight statistics - Min: {np.min(optimized_weights):.6e}, "
              f"Max: {np.max(optimized_weights):.6e}, "
              f"Sum: {np.sum(optimized_weights):.6f}")

    return Dataset(df=public_data.df, domain=public_data.domain, weights=optimized_weights)