from cka import CKACalculator

def CKA(
    X: jnp.ndarray, Y: jnp.ndarray, kernel: str = "linear", sigma_frac: float = 0.4
) -> jnp.ndarray:
    """Centered Kernel Alignment."""
    if kernel == "linear":
        K, L = linear_kernel(X, Y)
    elif kernel == "rbf":
        K, L = rbf_kernel(X, Y, sigma_frac)
    return cast(jnp.ndarray, HSIC(K, L) / jnp.sqrt(HSIC(K, K) * HSIC(L, L)))


@jax.jit
def linear_kernel(X: jnp.ndarray, Y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    K = X @ X.T
    L = Y @ Y.T
    return K, L


@jax.jit
def rbf_kernel(
    X: jnp.ndarray, Y: jnp.ndarray, sigma_frac: float = 0.4
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute radial basis function kernels."""

    # Define helper for euclidean distance
    def euclidean_dist_matrix(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """Compute matrix of pairwise, squared Euclidean distances."""
        norms_1 = (X**2).sum(axis=1)
        norms_2 = (Y**2).sum(axis=1)
        return jnp.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * jnp.dot(X, Y.T))

    # Define σ as a fraction of the median distance between examples
    dist_X = euclidean_dist_matrix(X, X)
    dist_Y = euclidean_dist_matrix(Y, Y)
    sigma_x = sigma_frac * jnp.percentile(dist_X, 0.5)
    sigma_y = sigma_frac * jnp.percentile(dist_Y, 0.5)
    K = jnp.exp(-dist_X / (2 * sigma_x**2))
    L = jnp.exp(-dist_Y / (2 * sigma_y**2))
    return K, L


@jax.jit
def HSIC(K: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    """Hilbert-Schmidt Independence Criterion."""
    m = K.shape[0]
    H = jnp.eye(m) - 1 / m * jnp.ones((m, m))
    numerator = jnp.trace(K @ H @ L @ H)
    return numerator / (m - 1) ** 2