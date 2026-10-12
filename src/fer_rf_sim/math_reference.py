"""Offline mathematical reference for the implemented FER model."""

from html import escape


MATH_SECTIONS = (
    (
        "notation",
        "Notation and units",
        """
        <p>The simulation models electron transmission through an STM vacuum
        barrier, integrates it into a current, and averages that current over
        a sinusoidal RF bias. Post-processing then follows constant-current
        contours and their voltage derivatives.</p>
        <table cellspacing="8">
        <tr><th align="left">Symbol</th><th align="left">Meaning</th></tr>
        <tr><td>E, E<sub>F</sub></td><td>Electron energy in eV, measured from the
        emitting electrode's band bottom, and the Fermi energies of the tip and
        sample (5.5 eV each by default; simulation settings).</td></tr>
        <tr><td>V, A</td><td>DC bias and peak RF voltage amplitude in V.</td></tr>
        <tr><td>Z, d</td><td>Tip height Z and barrier thickness d = Z in nm.
        The kernel receives the height directly, without conversion.</td></tr>
        <tr><td>&phi;<sub>t</sub>, &phi;<sub>s</sub></td><td>Tip and sample
        work functions in eV.</td></tr>
        <tr><td>F</td><td>Signed barrier slope in V/nm; an electron's
        potential-energy change is numerically in eV.</td></tr>
        <tr><td>a</td><td>a = 5.12 nm<sup>-1</sup> eV<sup>-1/2</sup>,
        approximately sqrt(2 m<sub>e</sub> e) / hbar in these units.</td></tr>
        <tr><td>T, I</td><td>Dimensionless transmission probability and
        model current in arbitrary units, not calibrated amperes.</td></tr>
        </table>
        <p>The energy reference is the electrode band bottom, not its Fermi
        level. Voltage-to-energy conversion uses the numerical equality
        of eV and an electron charge times V.</p>
        <p><b>Calculation chain:</b> barrier &rarr; transmission
        &rarr; instantaneous current &rarr; RF phase average
        &rarr; constant-current contour &rarr; spectral analysis.</p>
        """,
    ),
    (
        "barrier",
        "Vacuum barrier and bias polarity",
        """
        <p>For negative bias the implementation exchanges the electrode work
        functions and uses V<sub>eff</sub> = |V|. For positive bias,
        &phi;<sub>1</sub> = &phi;<sub>t</sub> and
        &phi;<sub>2</sub> = &phi;<sub>s</sub>; for negative bias these swap.</p>
        <p>With E<sub>F,e</sub> the Fermi energy of the emitting electrode
        (tip for V &ge; 0, sample for V &lt; 0), and Fermi levels aligned at zero bias:</p>
        <p>W<sub>0</sub> = &phi;<sub>1</sub> + E<sub>F,e</sub> - E<br>
        W<sub>d</sub> = &phi;<sub>2</sub> + E<sub>F,e</sub> - E - V<sub>eff</sub><br>
        F = (V<sub>eff</sub> + &phi;<sub>1</sub> - &phi;<sub>2</sub>) / d<br>
        W(x) = W<sub>0</sub> - F x, &nbsp; 0 &le; x &le; d</p>
        <p>W is the barrier energy relative to the incident electron.
        Unequal work functions can give a negative F even after taking |V|.
        The model requires d &gt; 0 and clips the final probability to
        0 &le; T &le; 1.</p>
        <p><b>Implementation:</b> physics.transmission.</p>
        """,
    ),
    (
        "wkb",
        "Schrodinger equation and WKB tunnelling",
        """
        <p>Inside the barrier the one-dimensional stationary Schrodinger
        equation can be written in the model's units as</p>
        <p class="equation">d<sup>2</sup>&psi;/dx<sup>2</sup>
        = a<sup>2</sup> W(x) &psi;(x).</p>
        <p>For W(x) &gt; 0 the wave is evanescent, with local decay rate
        &kappa;(x) = a sqrt(W(x)). The WKB approximation integrates this
        decay across the barrier:</p>
        <p class="equation">T<sub>WKB</sub>
        = f exp[-2 a &int;<sub>0</sub><sup>d</sup> sqrt(W(x)) dx]<br>
        = f exp[-4 a (W<sub>0</sub><sup>3/2</sup>
        - W<sub>d</sub><sup>3/2</sup>) / (3 F)].</p>
        <p>This branch is valid only when both W<sub>0</sub> &gt; 0 and
        W<sub>d</sub> &gt; 0. F is signed; using |F| in the denominator would
        be incorrect for a barrier that rises with x. The dimensionless
        factor f can be fitted against the Airy branch, not against a
        measured absolute current. <b>WKB is an explicit comparison mode,
        not the active default transmission.</b></p>
        <h3>Rectangular limit</h3>
        <p>The expression above would divide by zero at F = 0. When
        the endpoints coincide, its limit is</p>
        <p class="equation">T<sub>rect</sub>
        = f exp[-2 a d sqrt(W<sub>0</sub>)].</p>
        <p>The implementation actually evaluates the exponent as -2S with
        the rationalized action S defined in the low-field section. It
        avoids division by F at every field. This WKB limit omits the
        boundary-matching prefactor of the active rectangular-barrier result.</p>
        <p>The exponential explains why a small change in tip height can
        strongly change current. Probability clipping to [0, 1] occurs
        after evaluating the transmission approximation.</p>
        <p><b>Implementation:</b> physics.transmission, transmission_wkb.</p>
        """,
    ),
    (
        "airy",
        "Airy transmission and boundary matching",
        """
        <p>A linear barrier admits Airy-function solutions. Define</p>
        <p class="equation">&eta;(x) = (a / |F|)<sup>2/3</sup> W(x)<br>
        q = sign(F) (a<sup>2</sup> |F|)<sup>1/3</sup><br>
        &psi;(x) = c<sub>A</sub> Ai(&eta;(x))
        + c<sub>B</sub> Bi(&eta;(x)).</p>
        <p>Ai and Bi satisfy y''(&eta;) = &eta; y(&eta;). Matching the
        wavefunction and its derivative to electrode plane waves determines
        the transmitted amplitude. The transmission is a flux ratio, so it
        includes the outgoing-to-incoming wave-number ratio k<sub>3</sub>/k<sub>1</sub>.</p>
        <p class="equation">k<sub>1</sub> = a sqrt(max(E, 10<sup>-3</sup>))<br>
        k<sub>3</sub> = a sqrt(max(E + |V| - E<sub>F,e</sub> + E<sub>F,c</sub>, 10<sup>-12</sup>)),</p>
        <p>where E<sub>F,e</sub> and E<sub>F,c</sub> are the Fermi energies of the
        emitting and collecting electrodes. They are equal by default.</p>
        <p>Let A<sub>0</sub> = Ai(&eta;(0)), B<sub>0</sub> = Bi(&eta;(0)),
        A<sub>d</sub> = Ai(&eta;(d)), B<sub>d</sub> = Bi(&eta;(d)). Primes
        below mean derivatives with respect to &eta;, not x.</p>
        <p class="equation">t<sub>1</sub> = A'<sub>0</sub> B'<sub>d</sub>
        - A'<sub>d</sub> B'<sub>0</sub><br>
        t<sub>2</sub> = A<sub>0</sub> B<sub>d</sub> - A<sub>d</sub> B<sub>0</sub><br>
        t<sub>3</sub> = A<sub>d</sub> B'<sub>0</sub> - A'<sub>0</sub> B<sub>d</sub><br>
        t<sub>4</sub> = A<sub>0</sub> B'<sub>d</sub> - A'<sub>d</sub> B<sub>0</sub><br>
        D = [(q/k<sub>1</sub>) t<sub>1</sub>
        + (k<sub>3</sub>/q) t<sub>2</sub>]<sup>2</sup><br>
        &nbsp;&nbsp;&nbsp;&nbsp;+ [(k<sub>3</sub>/k<sub>1</sub>) t<sub>3</sub>
        + t<sub>4</sub>]<sup>2</sup><br>
        T<sub>Airy</sub> = (k<sub>3</sub>/k<sub>1</sub>) (4/&pi;<sup>2</sup>) / D.</p>
        <p>This unscaled tilted formula can describe oscillatory transmission
        when part of the barrier is classically allowed. It requires nonzero
        F. In that branch, denominators with |D| &lt; 10<sup>-12</sup> are
        replaced by infinity, producing zero rather than a divergent probability.</p>
        <p>When both endpoints are positive, the active default instead
        evaluates an equivalent <b>exponentially scaled</b> Airy expression.
        Its near-zero-field case uses analytic rectangular matching, as
        detailed in the next topic.</p>
        <h3>Numerical Airy evaluation</h3>
        <p>When the barrier is not positive at both ends, transmission uses
        exact SciPy Airy values at every &eta;: exponentially scaled (airye)
        for &eta; &gt; 0, unscaled for &eta; &le; 0, with the exp(&plusmn;s)
        factors recombined analytically. The legacy airy_all helper below
        (two-term asymptotics past |&eta;| = 8) is no longer used by
        transmission; its truncation error made ln T jagged in bias and
        produced a grid-scale wobble in constant-current dZ/dV.</p>
        <p>For &eta; &gt; 8, put t = (2/3)&eta;<sup>3/2</sup>:</p>
        <p class="equation">Ai &asymp; exp(-t) (1 - 5/(72t))
        / (2 sqrt(&pi;) &eta;<sup>1/4</sup>)<br>
        Ai' &asymp; -&eta;<sup>1/4</sup> exp(-t) (1 - 7/(72t)) / (2 sqrt(&pi;))<br>
        Bi &asymp; exp(t) (1 + 5/(72t)) / (sqrt(&pi;) &eta;<sup>1/4</sup>)<br>
        Bi' &asymp; &eta;<sup>1/4</sup> exp(t) (1 + 7/(72t)) / sqrt(&pi;).</p>
        <p>The growing Bi terms are computed in log space and capped at
        exp(log(maximum float64) - 1) to avoid overflow.</p>
        <p>For &eta; &lt; -8, put u = -&eta;,
        &xi; = (2/3)u<sup>3/2</sup> + &pi;/4,
        H = 1/(sqrt(&pi;) u<sup>1/4</sup>),
        c = 1 + 5/(72&xi;), c' = 1 + 7/(72&xi;):</p>
        <p class="equation">Ai &asymp; H sin(&xi;) c<br>
        Bi &asymp; H cos(&xi;) / c<br>
        Ai' &asymp; -H sqrt(u) cos(&xi;) c'<br>
        Bi' &asymp; H sqrt(u) sin(&xi;) / c'.</p>
        <p>Here c' is the named correction factor, not a derivative.
        These formulas describe exactly the approximation used in this code.</p>
        <p><b>Implementation:</b> physics.transmission, transmission_airy, airy_all.</p>
        """,
    ),
    (
        "hybrid",
        "Stable low-field matching and the WKB prefactor",
        """
        <p><b>Active default:</b> scaled Airy boundary matching, including
        an analytic near-rectangular limit. The compatibility mode name
        "hybrid" does not mean that a WKB/Airy switch is still applied.
        The current model is trapezoidal-scaled-airy-5.</p>
        <h3>A stable barrier action</h3>
        <p>For two positive barrier endpoints the integrated decay is
        evaluated without subtracting nearly equal powers or dividing by F:</p>
        <p class="equation">S = a &int;<sub>0</sub><sup>d</sup> sqrt(W(x)) dx<br>
        = (2 a d/3) (W<sub>0</sub> + sqrt(W<sub>0</sub> W<sub>d</sub>)
        + W<sub>d</sub>) / (sqrt(W<sub>0</sub>) + sqrt(W<sub>d</sub>))<br>
        &delta; = exp(-2S).</p>
        <p>This is the rationalized form of the WKB action, but the
        transmission below also retains the electrode boundary matching.</p>
        <h3>Near-rectangular analytic matching</h3>
        <p>When |W<sub>0</sub> - W<sub>d</sub>| &le;
        10<sup>-6</sup> max(W<sub>0</sub>, W<sub>d</sub>), put
        &kappa; = a sqrt((W<sub>0</sub> + W<sub>d</sub>)/2),
        C = (1 + &delta;)/2, H = (1 - &delta;)/2. Then</p>
        <p class="equation">T = 4 k<sub>1</sub> k<sub>3</sub> &kappa;<sup>2</sup> &delta;
        / [ &kappa;<sup>2</sup> (k<sub>1</sub> + k<sub>3</sub>)<sup>2</sup> C<sup>2</sup><br>
        &nbsp;&nbsp;+ (&kappa;<sup>2</sup> - k<sub>1</sub> k<sub>3</sub>)<sup>2</sup> H<sup>2</sup> ].</p>
        <p>C and H are attenuated versions of cosh(S) and sinh(S).
        H is computed using -expm1(-2S)/2 to retain precision for small S.
        At exactly zero field S = &kappa; d. This limit occurs when
        |V| + &phi;<sub>1</sub> - &phi;<sub>2</sub> = 0, which need not be
        at zero bias: 4/5 eV tip/sample work functions give F = 0 at +1 V.</p>
        <h3>Exponentially scaled Airy matching</h3>
        <p>For tilted, fully forbidden barriers, SciPy airye gives
        Atilde(&eta;) = Ai(&eta;) exp(&zeta;) and
        Btilde(&eta;) = Bi(&eta;) exp(-&zeta;), where
        &zeta; = (2/3)&eta;<sup>3/2</sup>; derivatives have the same
        scaling. Let s = sign(F), U = exp[-S(1+s)] and
        R = exp[-S(1-s)]. The scaled determinant terms are</p>
        <p class="equation">t<sub>1</sub> = Atilde'<sub>0</sub> Btilde'<sub>d</sub> U
        - Atilde'<sub>d</sub> Btilde'<sub>0</sub> R<br>
        t<sub>2</sub> = Atilde<sub>0</sub> Btilde<sub>d</sub> U
        - Atilde<sub>d</sub> Btilde<sub>0</sub> R<br>
        t<sub>3</sub> = Atilde<sub>d</sub> Btilde'<sub>0</sub> R
        - Atilde'<sub>0</sub> Btilde<sub>d</sub> U<br>
        t<sub>4</sub> = Atilde<sub>0</sub> Btilde'<sub>d</sub> U
        - Atilde'<sub>d</sub> Btilde<sub>0</sub> R<br>
        D<sub>scaled</sub> = [(q/k<sub>1</sub>) t<sub>1</sub>
        + (k<sub>3</sub>/q) t<sub>2</sub>]<sup>2</sup><br>
        &nbsp;&nbsp;+ [(k<sub>3</sub>/k<sub>1</sub>) t<sub>3</sub> + t<sub>4</sub>]<sup>2</sup><br>
        T = (k<sub>3</sub>/k<sub>1</sub>) (4/&pi;<sup>2</sup>) &delta; / D<sub>scaled</sub>.</p>
        <p>All exponential factors are non-growing. This avoids the
        overflow/cancellation problem of unscaled Airy functions near F = 0.
        If either endpoint is nonpositive, the unscaled tilted branch is
        used at nonzero field; a zero-field point without positive endpoints
        retains zero transmission.</p>
        <h3>Legacy WKB prefactor</h3>
        <p>The legacy constants are F<sub>1</sub> = 0.2 V/nm and
        b = 0.05 V/nm. F1 and F_blend arguments remain accepted for
        compatibility, but do not control the active transmission. Earlier
        models used a smoothstep blend in log probability; that transition
        is no longer applied.</p>
        <p>The simulation no longer computes or records this factor; it is
        available only as the standalone helper fit_wkb_prefactor for comparing
        explicit WKB against Airy. It samples distances d<sub>i</sub>
        and voltages V<sub>i</sub> = d<sub>i</sub>F<sub>1</sub>
        - &phi;<sub>t</sub> + &phi;<sub>s</sub>, then evaluates unscaled
        WKB and Airy on an energy grid below E<sub>F</sub>. At each distance
        it minimizes the squared mismatch over energies whose two barrier
        endpoints are positive:</p>
        <p class="equation">f<sub>i</sub> = arg min<sub>f</sub>
        &Sigma;<sub>j</sub> (f T<sub>WKB,ij</sub> - T<sub>Airy,ij</sub>)<sup>2</sup><br>
        = &Sigma;<sub>j</sub>(T<sub>Airy,ij</sub> T<sub>WKB,ij</sub>)
        / &Sigma;<sub>j</sub>T<sub>WKB,ij</sub><sup>2</sup><br>
        f = mean of finite positive f<sub>i</sub> with nonzero denominator.</p>
        <p>No valid samples is an error. The factor applies only
        to explicit WKB evaluation, not the active Airy result. This is a
        numerical comparison, not a fit to experimental resonances or a
        tunnelling-current calibration.</p>
        <p><b>Implementation:</b> physics._forbidden_barrier_transmission,
        transmission, fit_wkb_prefactor.</p>
        """,
    ),
    (
        "current",
        "From transmission to instantaneous current",
        """
        <p>Transmission is integrated with an occupation/supply weight.
        For the instantaneous voltage v, the implemented model is</p>
        <p class="equation">w(E, v) = min(|v|, E<sub>F,e</sub> - E),
        &nbsp; 0 &le; E &le; E<sub>F,e</sub><br>
        I<sub>0</sub>(v, Z) = sign(v)
        &int;<sub>0</sub><sup>E<sub>F,e</sub></sup>
        w(E, v) T(E, v, Z) dE.</p>
        <p>The upper limit is the emitting electrode's Fermi energy (a low-temperature
        assumption): the tip's for v &ge; 0 and the sample's for v &lt; 0.
        For |v| &lt; E<sub>F,e</sub>, the weight is |v| up to
        E<sub>F,e</sub> - |v|, then decreases linearly to zero. For
        |v| &ge; E<sub>F,e</sub>, it is E<sub>F,e</sub> - E throughout.
        The sign sets the current direction, and I<sub>0</sub>(0, Z) = 0.</p>
        <h3>Discrete energy integral</h3>
        <p>Only supplied energy samples inside [0, E<sub>F,e</sub>] are kept.
        For Y<sub>j</sub> = w(E<sub>j</sub>, v) T(E<sub>j</sub>, v, Z),
        the composite trapezoid rule gives</p>
        <p class="equation">I<sub>0</sub> &asymp; sign(v)
        &Sigma;<sub>j</sub> (E<sub>j+1</sub> - E<sub>j</sub>)
        (Y<sub>j+1</sub> + Y<sub>j</sub>) / 2.</p>
        <p>The code does not insert exact endpoints or the occupation kink.
        Therefore a restricted energy range integrates only its retained
        portion, and energy resolution must be checked for convergence.
        Fewer than two retained samples yields a zero trapezoid integral.</p>
        <p><b>Units:</b> the stored current prefactor is 1.0 and current units
        are marked normalized. There is no absolute-current prefactor, area,
        temperature-dependent Fermi function, or material-specific DOS in
        this integral. "Normalized" does not mean divided by its maximum.</p>
        <p><b>Implementation:</b> quadrature.current, current_batch.</p>
        """,
    ),
    (
        "rf",
        "RF phase averaging and Chebyshev quadrature",
        """
        <p>The voltage at RF phase &theta; is v(&theta;) = V + A cos(&theta;).
        A is a peak voltage, not RMS or peak-to-peak. Assuming a quasistatic
        response, the measured model current is a phase average:</p>
        <p class="equation">I(V, Z, A) = (1/(2&pi;))
        &int;<sub>0</sub><sup>2&pi;</sup>
        I<sub>0</sub>(V + A cos(&theta;), Z) d&theta;<br>
        = (1/&pi;) &int;<sub>-1</sub><sup>1</sup>
        I<sub>0</sub>(V + A t, Z) / sqrt(1 - t<sup>2</sup>) dt.</p>
        <p>The substitution t = cos(&theta;) exposes the Chebyshev weight.
        Gauss-Chebyshev quadrature uses equal-weight nodes:</p>
        <p class="equation">&theta;<sub>k</sub> = (2k - 1)&pi; / (2N),
        &nbsp; k = 1, ..., N<br>
        t<sub>k</sub> = cos(&theta;<sub>k</sub>)<br>
        I(V, Z, A) &asymp; (1/N)
        &Sigma;<sub>k</sub> I<sub>0</sub>(V + A t<sub>k</sub>, Z).</p>
        <p>N is the RF quadrature-point setting. Every phase uses its own
        transmission, current sign and occupation weight. Averaging T first
        and integrating it with the DC occupation weight is generally not
        equivalent; the active simulation averages complete currents.</p>
        <h3>Equivalent broadening kernel</h3>
        <p>For A &gt; 0, the voltage offset has the arcsine density</p>
        <p class="equation">p<sub>A</sub>(u) = 1 / (&pi; sqrt(A<sup>2</sup> - u<sup>2</sup>)),
        &nbsp; |u| &lt; A,<br>
        I(V, Z, A) = &int;<sub>-A</sub><sup>A</sup>
        p<sub>A</sub>(u) I<sub>0</sub>(V + u, Z) du.</p>
        <p>The density integrates to one. It weights the turning points
        strongly because a sinusoid spends more time there. At A = 0 the
        average reduces to the DC curve. A drive that crosses zero bias
        includes both current signs.</p>
        <p>As a small-amplitude mathematical check, for a sufficiently smooth
        DC curve the continuous phase average is
        I = I<sub>0</sub> + (A<sup>2</sup>/4) I<sub>0</sub>''
        + (A<sup>4</sup>/64) I<sub>0</sub>'''' + ... . The code evaluates
        quadrature, not this series. RF frequency and dynamical feedback
        do not enter this quasistatic model.</p>
        <p><b>Implementation:</b> quadrature.chebyshev_phases, cosine_nodes,
        CGQ, rf_averaged_current; simulation._rf_current_grid.</p>
        """,
    ),
    (
        "lut",
        "Sampling grids and lookup-table interpolation",
        """
        <p>Simulation axes are evenly spaced: n<sub>E</sub> energies,
        n<sub>V</sub> biases, n<sub>Z</sub> heights and n<sub>A</sub> amplitudes.
        Current has axis order (Z, V, A); the transmission LUT has order
        (Z, V, E). The energy upper bound includes the configured extra
        energy margin, but the current integral still masks to E &le; E<sub>F</sub>.</p>
        <p>A lookup table stores the same default Airy probability as the direct
        path. Its voltage coverage extends from V<sub>min</sub> - A<sub>max</sub>
        to V<sub>max</sub> + A<sub>max</sub> so the RF excursion is covered.
        Upscaling by an integer s changes an axis of n points to</p>
        <p class="equation">n<sub>new</sub> = (n - 1) s + 1.</p>
        <p>This subdivides each existing interval while keeping both
        endpoints. Within a LUT cell, RegularGridInterpolator performs
        trilinear interpolation:</p>
        <p class="equation">T(Z, V, E) &asymp;
        &Sigma;<sub>i,j,k = 0,1</sub>
        w<sub>i</sub>(&alpha;) w<sub>j</sub>(&beta;) w<sub>k</sub>(&gamma;)
        T<sub>ijk</sub>,<br>
        w<sub>0</sub>(t) = 1 - t, &nbsp; w<sub>1</sub>(t) = t.</p>
        <p>&alpha;, &beta;, &gamma; are fractional positions along height,
        voltage and energy in that cell. Out-of-bounds queries raise an
        error; the LUT does not extrapolate. Direct and LUT paths then use
        the same energy integral and RF-node averaging.</p>
        <p>LUT interpolation is a numerical approximation, not new physics.
        Sharp resonances may need a finer table. Compare against direct
        evaluation and increase energy, voltage, height and phase sampling
        independently to test convergence.</p>
        <p><b>Implementation:</b> lut_build._upscaled_axis, build_lut, interp_lut;
        simulation.direct_rf_current_grid, lut_rf_current_grid.</p>
        """,
    ),
    (
        "constant-current",
        "Constant-current contours and voltage derivatives",
        """
        <p>At a target current I<sub>*</sub>, each (V, A) column is inverted
        to find a height Z<sub>*</sub> satisfying I(Z<sub>*</sub>, V, A) = I<sub>*</sub>.
        For neighboring samples that bracket the target:</p>
        <p class="equation">Z<sub>*</sub> = Z<sub>1</sub>
        + (Z<sub>2</sub> - Z<sub>1</sub>)
        (I<sub>*</sub> - I<sub>1</sub>) / (I<sub>2</sub> - I<sub>1</sub>).</p>
        <p>The setpoint is a magnitude, so I is replaced by |I| (negative
        bias). A column is accepted when |I| crosses the target exactly once;
        noise far from the target does not reject it. Multiple crossings,
        plateaus at the target, nonfinite columns and targets outside the
        sampled range produce NaN. Exact endpoints are included; inversion
        never extrapolates.</p>
        <h3>Fixed-height conductance-like derivative</h3>
        <p>First, G = &part;I/&part;V is computed on the original current
        grid using NumPy gradient with the actual voltage coordinates.
        On a uniform grid the interior rule is</p>
        <p class="equation">G<sub>j</sub> &asymp;
        (I<sub>j+1</sub> - I<sub>j-1</sub>) / (2 &Delta;V).</p>
        <p>Endpoints use first-order one-sided differences; nonuniform grids
        use coordinate-aware coefficients. The already-computed G is then
        linearly interpolated to the inverted height:</p>
        <p class="equation">G<sub>*</sub> = G<sub>1</sub>
        + (G<sub>2</sub> - G<sub>1</sub>)
        (Z<sub>*</sub> - Z<sub>1</sub>) / (Z<sub>2</sub> - Z<sub>1</sub>).</p>
        <p>Its units are model-current units/V. This is the fixed-height
        partial derivative evaluated on a constant-current contour, not
        a total derivative along that contour.</p>
        <h3>Relation to dZ/dV</h3>
        <p>Implicit differentiation of a smooth constant-current contour gives</p>
        <p class="equation">0 = (&part;I/&part;V)<sub>Z,A</sub>
        + (&part;I/&part;Z)<sub>V,A</sub> dZ<sub>*</sub>/dV<br>
        dZ<sub>*</sub>/dV = -(&part;I/&part;V) / (&part;I/&part;Z).</p>
        <p>This relation needs a nonzero height derivative. The GUI's
        interpolated G is not dZ/dV; that ratio is an explanatory relation,
        not a calculation performed by the present GUI.</p>
        <p>The GUI selects positive setpoints on a 501-point logarithmic
        grid. Current heatmaps display log<sub>10</sub>|I|, so zero current
        is undefined and masked. Negative-current setpoints are not offered.
        Optional post-processing upsampling interpolates I and G separately;
        it does not recalculate G from an upsampled current.</p>
        <p><b>Implementation:</b> analysis.build_current_array,
        invert_current_for_constant_I_4D_vectorized, constant_current_slice.</p>
        """,
    ),
    (
        "peaks",
        "Peak positions, prominence and widths",
        """
        <p>The packaged peak helper calls SciPy's local-maximum detector.
        For negative peaks it detects maxima in s = -y; otherwise s = y.
        The returned peak height keeps the original signal's sign.</p>
        <p>Prominence measures how far a peak rises above its surrounding
        contour, not simply above zero. Starting at a peak, the algorithm
        searches each side to a higher peak or the signal boundary, finds
        the side minima, and uses the higher of the two bases:</p>
        <p class="equation">P = s<sub>peak</sub>
        - max(s<sub>left base</sub>, s<sub>right base</sub>)<br>
        s<sub>width</sub> = s<sub>peak</sub> - r P<br>
        width = x<sub>right crossing</sub> - x<sub>left crossing</sub>.</p>
        <p>Crossings at the chosen level are interpolated between samples.
        The default r = 0.5 is a <b>half-prominence width</b>, which is not
        necessarily full width at half maximum when the baseline is nonzero
        or neighboring peaks overlap.</p>
        <p>Position and width use the x-axis units. Height and prominence
        use signal units. Minimum peak distance is in samples, even on an
        uneven x grid. Boundary samples are not peaks; a flat maximum uses
        its middle sample, rounded down.</p>
        <p>The helper requires at least three finite samples and a strictly
        increasing x axis. It does not smooth data, fit Gaussian/Lorentzian
        lineshapes, refine the peak center beyond its sample, or estimate
        uncertainty. The GUI's Peak Extraction tab uses this detector for
        measured lock-in spectra and simulated fixed-height dI/dV or
        constant-current dI/dV and dZ/dV. Missing samples are processed as
        separate finite segments, never bridged across gaps.</p>
        <p>RF Dependence matches nearby peaks one-to-one in increasing RF
        amplitude, within a configurable maximum bias shift from the last
        observation. Unmatched peaks create new tracks. Unknown-RF spectra
        remain extractable but are excluded from RF analysis, which requires
        two distinct known amplitudes. Each shift is relative to that track's
        first observation at its lowest RF amplitude, not necessarily RF-off.
        Repeated measurements are retained; plots show their median at each
        amplitude. Crossing or overlapping peaks can make matching ambiguous.</p>
        <p><b>Implementation:</b> peaks.detect_peaks, extract_spectra,
        real_peak_spectra, simulated_peak_spectra, track_rf_peaks.</p>
        """,
    ),
    (
        "broadening",
        "Broadening measured spectra and handling edges",
        """
        <p>The signal-broadening helper applies the same arcsine average to
        an interpolated measured curve f rather than recalculating a barrier:</p>
        <p class="equation">B<sub>A</sub>[f](V) = (1/N)
        &Sigma;<sub>k</sub> f(V + A cos((2k - 1)&pi;/(2N))).</p>
        <p>Bias and A must be in the same units. The output retains the
        input signal's units. At A = 0 this is interpolation without RF
        broadening. The default is N = 100 and linear interpolation;
        optional cubic interpolation falls back to linear if fewer than
        four distinct bias points are available.</p>
        <p>Inputs are sorted by bias, and duplicate biases retain the first
        supplied signal value. Nonfinite inputs are rejected. Values outside
        the interpolation domain are NaN, not extrapolated.</p>
        <h3>Complete-cycle support</h3>
        <p>If the measured domain is [V<sub>min</sub>, V<sub>max</sub>], the
        default valid output region is</p>
        <p class="equation">V<sub>min</sub> + A &le; V
        &le; V<sub>max</sub> - A.</p>
        <p>A configurable edge-cutoff factor c replaces A with cA; the
        legacy broadenCGQ wrapper uses cubic interpolation and c = 1.5.
        A smaller cutoff may still encounter NaNs in the interpolation.
        There is no truncated-kernel renormalization or artificial padding.
        Masked edges must be removed before using the finite-input peak
        detector.</p>
        <p><b>Implementation:</b> broadening.broaden_signal,
        broaden_index_spectrum, broadenCGQ; quadrature.CGQ.</p>
        """,
    ),
    (
        "experimental-units",
        "Experimental units, RF power and display",
        """
        <p>The data loader can add SI-converted columns:</p>
        <p class="equation">V = 10<sup>-3</sup> V<sub>mV</sub><br>
        L<sub>V</sub> = 10<sup>-3</sup> L<sub>mV</sub><br>
        I<sub>A</sub> = 10<sup>-9</sup> I<sub>nA</sub>.</p>
        <p>The Real Data plot still uses the original bias-mV and lock-in-mV
        columns. A lock-in output voltage is not automatically conductance
        or dZ/dV: converting it needs instrument gain, modulation amplitude
        and phase calibration. The GUI does not average replicate spectra
        or rescale their signal amplitudes.</p>
        <h3>Power to peak voltage</h3>
        <p>For a sinusoid dissipating power P into a resistance R:</p>
        <p class="equation">P<sub>W</sub> = 10<sup>(p<sub>dBm</sub> - 30)/10</sup><br>
        V<sub>RMS</sub> = sqrt(R P<sub>W</sub>)<br>
        A<sub>peak</sub> = sqrt(2 R P<sub>W</sub>)<br>
        A<sub>peak</sub> = sqrt(2) A<sub>RMS</sub> = A<sub>pp</sub> / 2.</p>
        <p>The default impedance is 50 ohms. A source voltage into 50 ohms
        is not a calibrated junction voltage; attenuation and coupling are
        separate experimental quantities. The amplitude-transfer ratio is
        20 log<sub>10</sub>(A<sub>junction</sub>/A<sub>source</sub>) dB,
        and the corresponding power ratio at equal impedance is the square
        of the amplitude ratio.</p>
        <p>Explicit per-file RF mapping takes precedence over amplitude
        metadata, then power metadata. Missing RF metadata stays unknown;
        file ordering does not assign amplitudes.</p>
        <p><b>Implementation:</b> units.amplitude_to_peak, dbm_to_peak_volts;
        real_data.RealDataRepository.</p>
        """,
    ),
    (
        "recipe-fits",
        "Standalone smoothing and RF-amplitude fits",
        """
        <p><b>Optional recipes, not the GUI simulation.</b> Dataset-specific
        scripts use additional signal processing and instrument calibrations.
        Their calibration constants should not be applied to unrelated data.</p>
        <h3>Topography and lock-in comparison</h3>
        <p>A recipe converts its RelativeZ channel by
        Z<sub>nm</sub> = 1.721 &times; 10<sup>-4</sup> RelativeZ, uses
        fourfold cubic interpolation, and smooths with a Savitzky-Golay
        filter. That filter fits a low-degree polynomial by least squares
        in a moving window and evaluates it at the central sample.</p>
        <p>It estimates the topographic derivative at bias midpoints:</p>
        <p class="equation">D<sub>i</sub> = (Z<sub>i+1</sub> - Z<sub>i</sub>)
        / (V<sub>i+1</sub> - V<sub>i</sub>)<br>
        RMS(y) = sqrt(mean(y<sup>2</sup>))<br>
        L<sub>scaled</sub> = L RMS(D) / RMS(L).</p>
        <p>The RMS ratio is evaluated on a common-overlap grid. It matches
        signal size, not sign or lock-in phase. Without sufficient finite
        overlap or nonzero lock-in RMS the signal stays unscaled; the
        displayed derivative units are then not established by this step.</p>
        <h3>Fitting RF broadening to an RF-on curve</h3>
        <p>A measured RF-off reference f<sub>off</sub> is broadened by a trial
        A and optionally shifted by a signal offset Y:</p>
        <p class="equation">r<sub>i</sub> =
        B<sub>A</sub>[f<sub>off</sub>]<sub>i</sub> + Y - f<sub>on,i</sub><br>
        J(A, Y) = sqrt(mean<sub>valid</sub>(r<sub>i</sub><sup>2</sup>))
        [1 + 0.1 (1 - N<sub>valid</sub>/N)].</p>
        <p>The extra factor penalizes loss of comparisons near edges.
        Fewer than ten valid comparisons returns a large penalty
        (10<sup>6</sup>). A numerical optimizer minimizes J. Y has signal
        units, not voltage units. These scripts compare by row/index, so
        matching bias axes must be checked; convergence or junction
        calibration is not guaranteed by the optimizer alone.</p>
        <p>The Ag(111) surface-state recipe broadens a measured curve; it
        does not implement an analytic surface-state dispersion or DOS law.</p>
        <p><b>Implementation:</b> standalone FER_dZdV_peak_analysis.py,
        FER_spectrum_plotter.py and Ag111 SS broadening.py.</p>
        """,
    ),
    (
        "fer-index",
        "FER peak-index scaling and field estimates",
        """
        <p><b>Standalone peak-index recipes.</b> A triangular-well
        semiclassical resonance approximation motivates the index coordinate</p>
        <p class="equation">x<sub>n</sub> = (n - 1/4)<sup>2/3</sup>,
        &nbsp; n = 1, 2, ...<br>
        V<sub>n</sub> = m x<sub>n</sub> + b.</p>
        <p>The quarter-index offset is the turning-point phase correction.
        The scripts assign sequential indices to detected peaks and perform
        an unweighted linear least-squares fit on a selected high-index
        subset. Missed or spurious peaks therefore change the indexing.
        The intercept b is interpreted as a work-function estimate in the
        voltage convention, not an independently calibrated material value.</p>
        <p>One recipe states the field relation</p>
        <p class="equation">V<sub>n</sub> = &phi;
        + 0.9458 F<sup>2/3</sup> (n - 1/4)<sup>2/3</sup><br>
        m = 0.9458 F<sup>2/3</sup>,
        &nbsp; F = (m/0.9458)<sup>3/2</sup>.</p>
        <p><b>Existing inconsistency:</b> FER_dZdV_peak_analysis.py uses
        this power-3/2 conversion, while FER_peak_index_analysis.py reports
        F = m/0.9458. They are not equivalent. The scripts do not establish
        the coefficient's field-unit convention through executable unit
        conversions; these legacy estimates should not be confused with
        the explicitly V/nm barrier slope used by the simulator.</p>
        <p>A separate fit-quality helper forms
        F<sub>n</sub> = V<sub>n</sub>/(Z<sub>topo</sub> + Z<sub>0</sub>)
        and fits (V<sub>n</sub> - &phi;)<sup>3/2</sup>
        = c n F<sub>n</sub> + b. Real fractional powers require
        V<sub>n</sub> &ge; &phi;, and a nonzero height is required. These
        legacy helpers do not guard all such domains.</p>
        <p><b>Implementation:</b> standalone FER_peak_index_analysis.py
        and FER_dZdV_peak_analysis.py. This fit is not implemented in the
        GUI's Peak Extraction tab.</p>
        """,
    ),
    (
        "empirical-models",
        "Empirical center-shift and linewidth models",
        """
        <p><b>Standalone fitted-data plotter.</b> This script consumes
        previously fitted peak centers and widths; it does not fit raw
        spectral lineshapes itself. Its baseline is the first explicitly
        RF-off center in each relevant group:</p>
        <p class="equation">&Delta;C = C - C<sub>RF-off</sub>.</p>
        <p>There is no baseline averaging or propagation of center
        uncertainty in this subtraction. The optional phenomenological
        center-shift models include</p>
        <p class="equation">Linear: &Delta;C = m A + b<br>
        Quadratic: &Delta;C = a A<sup>2</sup> + b A + c<br>
        Lambert W: &Delta;C = a Re[W<sub>k</sub>(b max(A, 10<sup>-12</sup>))] + c.</p>
        <p>Lambert W is defined by W(x) exp(W(x)) = x. The scripts explore
        its branches, and an inverse parameterization also appears.
        Taking only the real part does not enforce a real-branch domain.
        These are fitted curve shapes, not a derived FER transmission law.</p>
        <h3>Linewidth sigmoid</h3>
        <p class="equation">width(A) = w<sub>low</sub>
        + (w<sub>high</sub> - w<sub>low</sub>)
        / [1 + exp(-(A - A<sub>0</sub>)/k)].</p>
        <p>A<sub>0</sub> locates the transition and k controls its amplitude
        scale. Width retains the fitted input units. Parameters are not
        physically constrained in the legacy fit.</p>
        <h3>Fit statistics and outlier handling</h3>
        <p class="equation">SSE = &Sigma;<sub>i</sub> (y<sub>i</sub> - yhat<sub>i</sub>)<sup>2</sup><br>
        R<sup>2</sup> = 1 - SSE / &Sigma;<sub>i</sub>(y<sub>i</sub> - mean(y))<sup>2</sup><br>
        IQR = Q<sub>75</sub> - Q<sub>25</sub><br>
        MAD = median(|y - median(y)|).</p>
        <p>The recipes use IQR/MAD-based rejection and, for some center
        models, inverse bin-count weighting plus soft-L1 robust residual
        fitting. Soft-L1 uses &rho;(s) = 2(sqrt(1 + s) - 1), with s a
        scaled squared residual, reducing large residuals' influence.
        Constant data gives a zero R<sup>2</sup> denominator. Outlier
        rejection and a good score do not establish parameter uncertainty
        or a physical mechanism; fitted covariance is not generally reported.</p>
        <p><b>Implementation:</b> standalone FER_real_data_plotter.py.</p>
        """,
    ),
    (
        "image-charge",
        "Optional image-charge helper (not enabled)",
        """
        <p>An image-charge correction helper exists, but the active
        transmission function does <b>not</b> call it. The implemented
        barrier is the uncorrected trapezoid described earlier.</p>
        <p>For helper thickness D, voltage V, work function &phi;<sub>t</sub>
        and relative permittivity &epsilon;<sub>r</sub>, it constructs</p>
        <p class="equation">(V/D) s<sup>3</sup>
        + (&phi;<sub>t</sub> + V) s<sup>2</sup>
        + &phi;<sub>t</sub> D s - 4.255 D/&epsilon;<sub>r</sub> = 0.</p>
        <p>Despite the helper's historical Cardano name, the current code
        uses numerical polynomial roots, discards non-real roots and keeps
        the two smallest real roots s<sub>1</sub>, s<sub>2</sub>. With
        L = 3.7/D, it returns</p>
        <p class="equation">&phi;<sub>im</sub> = -[
        V(s<sub>1</sub> + s<sub>2</sub>)/(2D)<br>
        &nbsp;&nbsp;- (1.15 L D/(s<sub>2</sub> - s<sub>1</sub>))
        log |s<sub>2</sub>(D - s<sub>1</sub>)
        / (s<sub>1</sub>(D - s<sub>2</sub>))|].</p>
        <p>The coefficients inherit the helper's legacy unit convention;
        no validated conversion connects them to the active nm barrier.
        Missing real roots or singular log arguments can produce undefined
        values. This should not be treated as an enabled or calibrated
        physical correction.</p>
        <p><b>Implementation:</b> physics.image_potential,
        _cardano_two_smallest.</p>
        """,
    ),
    (
        "limits",
        "Model assumptions and convergence checks",
        """
        <p>The core calculation is a one-dimensional, free-electron,
        linear vacuum-barrier model with zero-temperature Fermi levels, quasistatic
        sinusoidal RF bias and normalized current. It is not a full
        atomistic STM model.</p>
        <ul>
        <li>No temperature-dependent occupations, inelastic scattering,
        material-specific DOS or frequency-dependent junction response is
        included in the active current calculation.</li>
        <li>The image-charge helper is inactive. The energy-level diagram
        script is illustrative geometry, not an independent DOS or
        electrostatic calculation.</li>
        <li>Near-rectangular matching, Airy asymptotics, probability clipping,
        finite energy grids and LUT interpolation each introduce numerical
        choices. Matching formulas does not certify agreement with experiment.</li>
        <li>NaN marks an unsupported inversion or broadened edge. It is not
        a zero signal and should not be silently replaced by zero.</li>
        <li>Derivative and peak estimates can be more sampling-sensitive
        than current itself. Smoothing changes peak positions and widths;
        interpolation cannot recover information absent from the data.</li>
        </ul>
        <p>Useful checks are A = 0 recovering the DC result, the F &rarr; 0
        analytic rectangular limit, probabilities within [0, 1], direct/LUT agreement,
        and stability on independently refined grids. RF averaging need not
        give zero current at V = 0 for an asymmetric junction.</p>
        <p>These pages describe the current implementation. A loaded legacy
        artifact can have unknown provenance and may have been generated
        by an earlier model. Current GUI setpoints are not calibrated
        experimental currents, and optional standalone fit models are not
        automatically part of the GUI simulation.</p>
        """,
    ),
)


def math_reference_html() -> str:
    contents = " ".join(
        f'<p><a href="#{anchor}">{escape(title)}</a></p>'
        for anchor, title, _ in MATH_SECTIONS
    )
    sections = "".join(
        f'<h2><a name="{anchor}"></a>{escape(title)}</h2>{body}'
        for anchor, title, body in MATH_SECTIONS
    )
    return f"""<html><head><style>
        body {{ color: #242d30; font-family: 'Segoe UI'; font-size: 11pt; }}
        h2 {{ color: #075e59; font-size: 15pt; margin-top: 24px; }}
        h3 {{ color: #39484d; font-size: 12pt; margin-top: 16px; }}
        p {{ margin-top: 10px; margin-bottom: 10px; }}
        p.equation {{ font-family: 'Cambria'; font-size: 12pt; color: #172e29; }}
        a {{ color: #087f78; }}
        th {{ color: #075e59; }}
        </style></head><body><h2>Contents</h2>{contents}{sections}</body></html>"""