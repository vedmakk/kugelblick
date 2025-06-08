import numpy as np



def mm_to_px(mm: float, dpi: float) -> float:
    """Convert millimetres to pixels based on a given DPI.

    Parameters
    ----------
    mm : float
        Length in millimetres.
    dpi : float
        Dots per inch.

    Returns
    -------
    float
        Corresponding length in pixels.
    """
    # One inch is exactly 25.4 millimetres
    return mm * dpi / 25.4


def generate_radial_mapping(
    r_max: int,
    gap_px: float,
    refr_index: float,
    num_samples: int | None = None,
):
    """Compute the mapping from *observed* radial coordinate ``r_obs`` (in pixels)
    to the *print-plane* radial coordinate ``r_plane`` (in pixels) for a sphere.

    The observer is assumed to look orthogonally (along the $-z$ axis) through the
    spherical lens onto the printed plane.

    Parameters
    ----------
    r_max : int
        Sphere radius in **pixels**.
    gap_px : float
        Gap between the bottom of the sphere and the print plane (in pixels).
    refr_index : float
        Refractive index of the sphere material (e.g. 1.46 for quartz).
    num_samples : int, optional
        Number of discrete radial samples (defaults to ``r_max + 1``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(r_obs_values, r_plane_values)`` – two 1-D arrays (length
        ``num_samples``) that can directly be fed into ``numpy.interp`` to map
        between the two radii (``r_plane = g(r_obs)``).
    """

    if num_samples is None:
        num_samples = r_max + 1

    # Discrete radial positions that will be traced (observer/image space).
    r_obs = np.linspace(0.0, float(r_max), num_samples, dtype=np.float64)

    # Pre-allocate array for the radial coordinate on the print plane.
    r_plane = np.full_like(r_obs, np.nan)

    # Constant vectors and scalars used throughout.
    n_air = 1.0
    n_sph = float(refr_index)

    # Sphere geometry (coordinate system: +z points *upwards*).
    R = float(r_max)
    z_bottom = gap_px  # Bottom of the sphere (tangent point to plane if gap==0)
    z_center = z_bottom + R

    # Incident ray direction (orthographic, coming straight down along –z).
    I = np.array([0.0, -1.0, 0.0])  # shape (3,)

    # Convenience scalars.
    eta_enter = n_air / n_sph  # air -> sphere

    # Iterate in vectorised form – trace all rays at once.
    # ---------------------------------------------------
    # Compute the intersection point with the *upper* hemisphere for every r.
    # z_top = z_center + sqrt(R^2 - r^2)
    tmp = np.maximum(0.0, R**2 - r_obs**2)
    z_top = z_center + np.sqrt(tmp)

    # Cartesian coordinates of intersection points P1, shape (N, 3).
    P1 = np.stack([r_obs, z_top, np.zeros_like(r_obs)], axis=1)  # (N,3)

    # Outward normal at P1 (unit vectors).
    N1 = (P1 - np.array([0.0, z_center, 0.0])) / R  # (N,3)

    # --- Refraction: air -> sphere -----------------------------------------
    cos_i1 = -np.einsum("ij,j->i", N1, I)
    # Clamp to prevent rounding errors beyond [-1,1]
    cos_i1 = np.clip(cos_i1, -1.0, 1.0)
    sin2_t1 = eta_enter**2 * (1.0 - cos_i1**2)

    # Internal rays that would undergo total internal reflection at the *first*
    # interface do not exist for n_sph>n_air, so this branch is not strictly
    # necessary – keep it for completeness.
    valid = sin2_t1 <= 1.0

    cos_t1 = np.sqrt(np.maximum(0.0, 1.0 - sin2_t1))
    T1 = (eta_enter * I) + ((eta_enter * cos_i1 - cos_t1)[:, None] * N1)
    # Normalise to unit length to counteract tiny numeric drift.
    T1 /= np.linalg.norm(T1, axis=1)[:, None]

    # For invalid rays (should not occur) leave r_plane as NaN – they will be
    # ignored later.

    # --- Second intersection with the sphere (exit point P2) --------------
    # Given that P1 lies on the sphere surface and T1 is a unit vector, the
    # parametric distance to the second intersection is:
    #   s_exit = -2 * dot(d0, T1)
    d0 = P1 - np.array([0.0, z_center, 0.0])  # (N,3)
    s_exit = -2.0 * np.einsum("ij,ij->i", d0, T1)

    # Coordinates of exit points P2.
    P2 = P1 + (T1 * s_exit[:, None])

    # Outward normal at P2.
    N2 = (P2 - np.array([0.0, z_center, 0.0])) / R

    # --- Refraction: sphere -> air ----------------------------------------
    eta_exit = n_sph / n_air  # >1

    # Note: For the *second* refraction (sphere → air) the incident ray *T1*
    # is already **inside** the sphere.  The outward surface normal *N2* thus
    # points *into the same general direction* as the incident ray for all
    # rays that actually exit through the lower hemisphere.
    #
    # Using the same convention as above (cos_i = -dot(N, I)) would therefore
    # yield a *negative* cos_i for these rays.  While mathematically valid,
    # the ensuing formula
    #
    #   T = eta * I + (eta * cos_i - cos_t) * N
    #
    # would flip the sign of the transmitted direction and send the ray
    # *upwards* again – clearly wrong.  Instead we compute the cosine of the
    # incident angle directly via ``dot(N, I)`` which is **positive** in the
    # typical exit scenario and keeps the transmitted ray heading downwards
    # towards the print plane.
    cos_i2 = np.einsum("ij,ij->i", N2, T1)  # no leading minus sign here!
    cos_i2 = np.clip(cos_i2, -1.0, 1.0)
    sin2_t2 = eta_exit**2 * (1.0 - cos_i2**2)

    # Identify rays that undergo total internal reflection at the *bottom*
    # interface – they will never leave the sphere along the desired path.
    tir_mask = sin2_t2 > 1.0

    # For rays that pass, compute outgoing direction T2.
    cos_t2 = np.sqrt(np.maximum(0.0, 1.0 - sin2_t2))
    T2 = (eta_exit * T1) + ((eta_exit * cos_i2 - cos_t2)[:, None] * N2)
    T2 /= np.linalg.norm(T2, axis=1)[:, None]

    # --- Intersection with the print plane z = 0 ---------------------------
    # Plane equation: z = 0 (remember: +z is *upwards*).
    # Parametric intersection: P = P2 + T2 * s  ==>  z-component == 0
    T2_z = T2[:, 1]  # y-component corresponds to z-axis in our notation
    P2_z = P2[:, 1]

    # Avoid division by zero for horizontal rays (should not occur here).
    with np.errstate(divide="ignore", invalid="ignore"):
        s_plane = -P2_z / T2_z

    X_plane = P2[:, 0] + T2[:, 0] * s_plane

    r_plane_computed = np.abs(X_plane)

    # Store results where everything was valid (no TIR, ray intersects plane
    # in front of the sphere, etc.).
    # A parametric distance ``s_plane`` of **zero** corresponds to rays that
    # leave the sphere exactly at the point where it touches the print plane
    # (i.e. for *gap_px == 0* and *r_obs == 0*).  These rays are perfectly
    # valid and should be included in the mapping.  Therefore we also accept
    # *s_plane == 0* here.
    ok = (~tir_mask) & np.isfinite(s_plane) & (s_plane >= 0)
    r_plane[ok] = r_plane_computed[ok]

    # For invalid rays (mostly those beyond the critical angle) we simply keep
    # NaN so that the caller can decide how to treat those radii.

    return r_obs, r_plane
