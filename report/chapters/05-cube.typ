#import "../common.typ": *

= Models for the fixed cube <sec-cube>

#figure(img("cube_learning_curve.png", w: 88%),
  caption: [Error on the 100 independent simulations against the number of training simulations.
  Shaded: 95 % confidence interval. Both axes are logarithmic: POD + GP is three orders of magnitude
  better than the others from 90 simulations on.],
) <fig-learning>

== Baseline: the mean field

Predicting the mean training field everywhere gives an MAE of
#fmt(get("mean", 720).sim_MAE, d: 1) °C. This is the bar to beat, and it shows that the fields vary a lot
between simulations (37 to 98 °C on average).

== Multi-layer perceptron (MLP)

#idea(title: "Why")[The most direct idea: a neural network that takes the 3 inputs and outputs
the 560 temperatures. It needs no knowledge of the physics or the mesh; this makes it the natural
first try.]

*How it works.* Four fully connected layers, $3 -> 128 -> 256 -> 256 -> 560$, with ReLU activations
(243 k parameters). Inputs and outputs are standardised (zero mean, unit variance, from the training
set). We minimise the mean squared error with Adam (learning rate $10^(-3)$, batch 64) and keep the
epoch with the best validation error. We compared two schedules: a constant learning rate with early
stopping (patience 30 epochs), and a *cosine* schedule (10 warm-up epochs, then the rate decreases
smoothly to 0 over 1500 epochs).

#figure(
  table(columns: 6, align: (left, right, right, right, right, right),
    [MAE per simulation (°C)], [90], [180], [360], [720], [95 % CI at 720],
    [mean field], ..(90, 180, 360, 720).map(n => fmt(get("mean", n).sim_MAE, d: 2)), [],
    [MLP, constant lr], ..(90, 180, 360, 720).map(n => fmt(get("mlp", n).sim_MAE)),
      [#fmt(get("mlp", 720).sim_MAE_lo) – #fmt(get("mlp", 720).sim_MAE_hi)],
    [MLP, cosine lr], ..(90, 180, 360, 720).map(n => fmt(get("mlp_cos", n).sim_MAE)),
      [#fmt(get("mlp_cos", 720).sim_MAE_lo) – #fmt(get("mlp_cos", 720).sim_MAE_hi)],
    [POD + ridge], ..(90, 180, 360, 720).map(n => fmt(get("pod_ridge", n).sim_MAE)),
      [#fmt(get("pod_ridge", 720).sim_MAE_lo) – #fmt(get("pod_ridge", 720).sim_MAE_hi)],
    [POD + GP], ..(90, 180, 360, 720).map(n => fmt(get("pod_gp", n).sim_MAE)),
      [#fmt(get("pod_gp", 720).sim_MAE_lo) – #fmt(get("pod_gp", 720).sim_MAE_hi)],
  ),
  caption: [All cube models on the same 100 independent simulations. The coordinate MLP
  (@sec-coord) reaches 0.093 °C with 360 simulations.],
) <tab-cube>

*Results.* The MLP works: with 720 simulations its error is 0.064 °C, 150 times lower than the
baseline and 0.3 % of the temperature range. But it is _data hungry_: multiplying the data by 8 (90 to 720 simulations) divides the error by
5 to 6, with no sign of a plateau. The cosine schedule is better at every size (−12 to −30 %). @fig-training explains why: with a constant rate, the validation error
jumps up and down and early stopping picks a noisy point; with the cosine schedule, the end of
training is calm and the network settles in a good minimum.

#figure(img("cube_training_curves.png"),
  caption: [Training (solid) and validation (dashed) losses of the MLP. With 90 simulations the
  network memorises the training set (training loss 300× below validation): this is overfitting.
  With 720 simulations the two curves stay together.],
) <fig-training>

#figure(
  grid(columns: 4, gutter: 3pt,
    img("cube_errmap_true.png"), img("cube_errmap_pred_mlp.png"), img("cube_errmap_err_mlp.png"), img("cube_errmap_err_gp.png"),
    align(center, text(8.5pt)[true (74 – 101 °C)]), align(center, text(8.5pt)[MLP prediction]),
    align(center, text(8.5pt)[MLP error (°C)]), align(center, text(8.5pt)[POD + GP error (*m*°C)])),
  caption: [The worst test simulation of the MLP ($Phi_0 = 26 space 560$, $h = 60$). The MLP error
  is an almost uniform offset of −0.44 °C: the _shape_ of the field is right, its _level_ is not.
  POD + GP on the same case: 0.07 m°C on average, 0.4 m°C at worst.],
) <fig-errmap>

#idea[@fig-errmap was the key observation. The MLP gets the shape right and the level wrong. Its
560 outputs are 560 separate numbers, and it must learn on its own that they move together. If the
fields really are "one shape whose level changes", we should give the model that structure instead
of asking it to discover it. This led to POD.]

#limit[
- Needs many simulations: 720 runs to reach 0.06 °C, and still improving.
- Tied to one mesh: 560 outputs means 560 fixed nodes; a new mesh or a new geometry needs a new model.
- Extrapolates badly (@sec-extrap): its error grows 57× outside the training range of $h$.
- Training is noisy; the result depends on the learning-rate schedule.
]

== Coordinate MLP <sec-coord>

#idea(title: "Why")[The field MLP has 560 separate outputs. Instead, we can ask a network for the
temperature _at one point_: $(T_0, Phi_0, h, x, y, z) -> T$. The same weights then serve every node,
so 360 simulations give 201 600 training examples instead of 360, and the model is not tied to a
mesh: it can be queried at any point.]

*How it works.* An MLP $6 -> 256 -> 256 -> 256 -> 256 -> 1$ (199 k parameters) on the `pointwise`
dataset, batch 4096 points, cosine schedule over 300 epochs (about 4 minutes on the GPU).

*Results.* Trained on 360 simulations, it reaches *0.093 °C* (CI 0.087 – 0.099) on the 100 independent
simulations, better than the field MLP with the same data (0.160 °C) and about as good as the field
MLP with twice the data.

#limit[
- Still a fixed geometry: $(x, y, z)$ tells _where_ the point is, but nothing about _what is around
  it_. A hole elsewhere in the part changes the temperature at a point; this model cannot know it.
- 560 network calls per field instead of one (still fast on a GPU).
]

== POD + regression

#idea(title: "Why")[If all fields share a few shapes, we can write
$T(bold(x)) approx overline(T)(bold(x)) + sum_(i=1)^k c_i phi_i (bold(x))$:
a mean field plus $k$ fixed shapes (_modes_) with weights $c_i$. Then the learning problem shrinks
from "3 inputs $->$ 560 outputs" to "3 inputs $->$ $k$ numbers", which needs far less data.]

*How it works.* _Proper Orthogonal Decomposition_ (POD) finds the best such shapes: it is a singular
value decomposition (SVD) of the centred training fields. @fig-pod shows how fast the energy decays:
the first mode alone carries 99.5 % of the variance, three modes 99.9998 %, five modes all but
$10^(-9)$. We keep $k = 5$ modes. A regressor then predicts $(c_1, ..., c_5)$ from $(T_0, Phi_0, h)$.
We tried two:

- *Ridge regression* (regularisation chosen by cross-validation) on 9 hand-made features suggested by
  the physics: $T_0, Phi_0, h, Phi_0\/h, T_0\/h, 1\/h, Phi_0 h, T_0 h, h^2$;
- *Gaussian process* (GP) regression, one per mode, with an anisotropic RBF kernel (one length scale
  per input) and a small noise term, hyper-parameters fitted by maximum likelihood.

#figure(
  grid(columns: (1.5fr, 1fr), gutter: 6pt, align: horizon,
    img("pod_spectrum.png"),
    grid(columns: 3, gutter: 2pt, img("pod_mode_1.png"), img("pod_mode_2.png"), img("pod_mode_3.png"),
      align(center, text(8pt)[mode 1]), align(center, text(8pt)[mode 2]), align(center, text(8pt)[mode 3]))),
  caption: [Left: share of the variance per POD mode and what is left after keeping $k$ modes.
  Right: the first three modes (red positive, blue negative). Mode 1 is "hot near the channel",
  mode 2 changes the contrast between channel and corners.],
) <fig-pod>

*Results.* POD + GP is the clear winner: *0.0001 °C* with 720 simulations, and already 0.0003 °C
with 90. That is 600 times better than the best MLP trained on the same data; even with only 90
simulations it is 190 times better than the MLP trained on 720. Fitting takes a few
seconds on a CPU, without a GPU. POD + ridge stops at about 0.08 °C whatever the data size: its
error comes from the limited list of features, not from a lack of data.

Why is the GP so good here? The map $(T_0, Phi_0, h) -> (c_1, ..., c_5)$ is very smooth and has only
3 inputs, and the FEM data has no noise. This is exactly the case where a GP is at its best: it
interpolates between training points with a smooth function whose smoothness it learns. A neural
network, trained by stochastic gradient descent, never gets the last digits right.

#limit[
- *Needs a minimum of data* (@fig-learning). With 10, 20 and 45 simulations the GP gives
  #fmt(limits.pod_gp_10, d: 2), #fmt(limits.pod_gp_20, d: 2) and #fmt(limits.pod_gp_45, d: 2) °C;
  below about 50 simulations ridge is as good or better. The GP needs enough points to estimate its
  length scales, then the error drops by 3 orders of magnitude between 45 and 90 simulations.
- *Fixed mesh and fixed geometry*: the modes are vectors on the 560 nodes. For the sponges, where
  every sample has another mesh, POD cannot be used directly.
- *Few inputs*: a GP costs $O(n^3)$ in the number of training points and loses its edge when the
  number of inputs grows (tens of geometric parameters).
- *Extrapolation* is still a weak spot (next section).
]

== Limit test: extrapolation <sec-extrap>

#idea(title: "Why")[A surrogate will be used for parameters a little outside what it saw. We trained
each model again on the 482 training simulations with $h >= 100$ only, and tested on the 100
independent simulations, 39 of which have $h < 100$, i.e. were never seen.]

#figure(
  grid(columns: (1.55fr, 1fr), gutter: 8pt, align: horizon,
    img("cube_extrapolation.png"),
    table(columns: 4, align: (left, right, right, right),
      [MAE (°C)], [$h >= 100$], [$h < 100$], [$h < 70$],
      [MLP (cos.)], fmt(limits.extrap_mlp.inside), fmt(limits.extrap_mlp.outside, d: 2), fmt(limits.extrap_mlp.far_h_lt_70, d: 2),
      [POD + ridge], fmt(limits.extrap_pod_ridge.inside), fmt(limits.extrap_pod_ridge.outside, d: 2), fmt(limits.extrap_pod_ridge.far_h_lt_70, d: 2),
      [POD + GP], fmt(limits.extrap_pod_gp.inside), fmt(limits.extrap_pod_gp.outside), fmt(limits.extrap_pod_gp.far_h_lt_70),
    )),
  caption: [Models trained only on $h >= 100$. Grey band: values of $h$ never seen. Every model
  degrades smoothly as $h$ moves away from the training range.],
) <fig-extrap>

All models degrade outside the training range, and the further, the worse. The MLP goes from
0.04 °C to 4.2 °C for $h < 70$ (×100); POD + GP from $3 times 10^(-5)$ to 0.065 °C (×2000, yet still
the most accurate). Note also that inside the range, all three models are 2 to 11 times more accurate
than in @tab-cube: the region $h < 100$, where parts are hot and still heating up, is where most of the
error of every model comes from.

== Speed

On a laptop CPU, one prediction takes about 2 ms for the MLP and the POD + GP model and 0.5 ms for
POD + ridge, against about 1 s for one FEM run: a speed-up of 500 to 2000. A batch of 100 predictions
takes 5 to 20 ms.

