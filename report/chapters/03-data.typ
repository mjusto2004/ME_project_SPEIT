#import "../common.typ": *

= Physics and datasets <sec-data>

== The physical model

All simulations solve the transient heat equation in aluminium
($rho = 2700 "kg/m"^3$, $c = 896 "J/(kg K)"$, $k = 237 "W/(m K)"$) over $t_"max" = 10 space 000 "s"$:

$ rho c (partial T) / (partial t) = k Delta T, quad
  -k (partial T) / (partial n) = h (T - T_"ext") "on cooled faces", quad
  -k (partial T) / (partial n) = -Phi_0 "on the heated surface". $

The part starts at a uniform $T_0$, the air is at $T_"ext" = 20 °"C"$. Time integration uses a
Crank–Nicolson scheme; the convection term is applied _explicitly_ (computed from the temperature of
the previous step), because EasyFEA has no built-in Robin condition. The output is the temperature
at every node at $t = 10 space 000 "s"$, in °C.

The sampled ranges are the same for both problems:
$T_0 in [273, 303] "K"$, $Phi_0 in [15 space 000, 30 space 000] "W/m"^2$,
$h in [50, 200] "W/(m"^2"K)"$.

#idea[A quick order of magnitude tells us what the data will look like. The cooling time of the
cube is $tau approx rho c V \/ (h A) approx 2700 dot 896 \/ (6 h)$, i.e. about 1.1 h for
$h = 100$ and 2.2 h for $h = 50$. After $10 space 000 "s" approx 2.8 "h"$, parts with strong
cooling are close to steady state (temperature set by $Phi_0 \/ h$), while parts with weak cooling
are still heating up (temperature also depends on $T_0$ and on time). We expect low-$h$ cases to be
the hardest to learn, and we will see this again and again.]

== Dataset 1: the fixed cube (`field` and `pointwise` kinds) <sec-cube-data>

A unit cube (1 m) contains a cylindrical channel of diameter 1/3 m along $z$. The heat flux
enters through the wall of this channel; the six outer faces are cooled by convection. The mesh is
always the same: 560 nodes, 2160 tetrahedra (@fig-cube-mesh). One simulation takes about one second
on one CPU core.

#figure(grid(columns: (1fr, 1.25fr), gutter: 10pt, align: horizon,
    img("cube_mesh.png"),
    [#set text(9pt)
     #table(columns: 2, align: (left, left),
      [Property], [Value],
      [geometry], [1 m cube, channel $diameter$ 1/3 m],
      [mesh], [560 nodes, 2160 tetrahedra],
      [inputs], [$T_0$, $Phi_0$, $h$],
      [output], [560 temperatures (°C)],
      [training set], [720 simulations (seed 1)],
      [independent test], [100 simulations (seed 2)],
      [time to generate], [≈ 2 min for 720 on 6 cores])
    ]),
  caption: [The cube mesh, cut in half at $y = 0.5$. Red: the wall of the heated channel.],
) <fig-cube-mesh>

@fig-cube-fields shows three simulations of the independent test set. The field is always the same
_shape_: hot around the channel, cooler at the corners. What changes is mostly the _level_ and the
contrast. @fig-cube-stats confirms that the mean temperature is driven mainly by the ratio
$Phi_0 \/ h$ (heating against cooling), with a spread caused by $T_0$ and by the transient effect
predicted above.

#figure(
  grid(columns: (1fr, 1fr, 1fr), gutter: 4pt,
    img("cube_field_0.png"), img("cube_field_1.png"), img("cube_field_2.png"),
    [#set text(8.5pt); #align(center)[$Phi_0 = 26 space 560$, $h = 60$ \ 74 – 101 °C]],
    [#set text(8.5pt); #align(center)[$Phi_0 = 15 space 892$, $h = 81$ \ 47 – 64 °C]],
    [#set text(8.5pt); #align(center)[$Phi_0 = 15 space 660$, $h = 182$ \ 31 – 48 °C]],
  ) + img("cube_field_colorbar.png", w: 55%),
  caption: [Three cube simulations (half cube, same colour scale). Hot (weak cooling, strong flux),
  median and cold cases.],
) <fig-cube-fields>

#figure(img("cube_stats.png"),
  caption: [The 720 training simulations. Left: sampling of $(h, Phi_0)$ (uniform), coloured by mean
  temperature. Middle: mean temperatures range from 37 to 98 °C. Right: the mean temperature is
  almost a function of $Phi_0 \/ h$; the spread at high $Phi_0\/h$ (low $h$) comes from $T_0$ and from
  parts still heating up.],
) <fig-cube-stats>

The same simulations can be stored in two ways, which matter for the models:

- `field`: one row per simulation, `[T0, Phi0, h, T_1 ... T_560]`. The model predicts the 560 values
  at once.
- `pointwise`: one row per (simulation, node), `[T0, Phi0, h, x, y, z, T]`. The model predicts one
  temperature at one point. 360 simulations give 201 600 rows.

#idea(title: "History of the cube dataset")[The first dataset (90 simulations, $h in [150, 300]$,
$T_0 in [146, 293] "K"$) was easy: with 360 simulations an MLP reached 0.008 °C. We then moved to
the current, more realistic ranges (lower $h$, $T_0$ around room temperature). Lower $h$ means
hotter parts that are further from steady state, and the same MLP fell to 0.18 °C. All results
below use the current ranges. We also tried a "smart" sampling: at fixed $h$ the solver is linear
in $T_0$ and $Phi_0$, so 40 values of $h$ with 3 runs each (120 runs) should be enough. Linearity
was confirmed (to $10^(-12)$ °C), but the MLP did _worse_ (0.079 °C vs 0.039 °C with 90 random
runs): a neural network does not know about this linearity and needs the $(T_0, Phi_0)$ plane to be
covered. We went back to plain random sampling.]

== Dataset 2: random sponges (`graph` kind) <sec-sponge-data>

To move towards real parts, we made the geometry random. Each sample is the unit cube minus 1 to 20
spherical holes of the same radius $r in [0.05, 0.3] "m"$, with random centres; a draw that cuts the
cube in several pieces is rejected and redrawn. Heat enters through one of the six outer faces,
chosen at random; convection acts on every other boundary, *including the walls of the holes*.
gmsh meshes each sponge with an element size of $1 \/ "res"$, with `res` drawn between 5 and 10.

#figure(
  grid(columns: (1fr, 1fr, 1fr), gutter: 4pt,
    img("sponge_example_0.png"), img("sponge_example_2.png"), img("sponge_example_7.png"),
    [#set text(8.5pt); #align(center)[19 holes, $r = 0.17$, $h = 156$ \ heated face $x = 1$]],
    [#set text(8.5pt); #align(center)[19 holes, $r = 0.22$, $h = 69$ \ heated face $x = 0$]],
    [#set text(8.5pt); #align(center)[1 hole, $r = 0.12$, $h = 87$ \ heated face $z = 1$]],
  ),
  caption: [Three sponges of the training set, cut at $y = 0.5$ (each with its own colour scale).
  Green dots: nodes of the heated face. Big holes and weak cooling create hot, poorly connected
  regions (middle).],
) <fig-sponges>

A sample is now a *graph*: the nodes carry the inputs $(T_0, Phi_0, h, r, "res", x, y, z,
"heated")$, where `heated` marks the nodes of the heated face, and two nodes are linked when they
share a tetrahedron. The dataset has three files: the node table (CSV), the links (`_edges.npz`) and
the spec (JSON).

#figure(
  table(columns: 4, align: (left, right, left, right),
    [Set], [Sponges], [mesh `res`], [nodes per sponge],
    [training (seed 10)], [1000], [5 – 10], [271 – 2977 (median 1047)],
    [test "same resolution" (seed 11)], [50], [7], [≈ 940 (mean)],
    [test "finer" (seeds 12, 13)], [20 + 20], [12, 18], [≈ 2 350, ≈ 5 830],
    [test "much finer" (seed 14)], [12], [25], [≈ 14 000],
  ),
  caption: [Sponge datasets. The training set has 1 098 804 node values and 12 071 168 directed
  links (81 MB of CSV); it took 7 minutes on 6 cores.],
) <tab-sponge-sets>

#figure(img("sponge_stats.png"),
  caption: [The 1000 training sponges. Left: mesh sizes. Middle: the hottest node of each sponge
  (log scale); a few sponges exceed 200 °C. Right: big holes ($r$ large) and weak cooling ($h$
  small) give the hottest parts.],
) <fig-sponge-stats>

#idea(title: "Checking the data before trusting it")[Two checks on the sponge data changed how we read
every result that follows.

*1. Rare numerical artefacts.* No node can be colder than both $T_0$ and the air (heat only comes in),
yet one training sponge out of 1000 has 4 nodes at up to 76 °C _below_ that bound. In an older
sponge dataset, one sample had 7 nodes at up to 1691 °C while 99 % of its nodes stayed below
87 °C. Both come from the explicit convection term on tiny elements that gmsh creates where holes
almost touch. They are rare, but a single such node dominates an RMSE, so we report MAE per
simulation.

*2. The "truth" depends on the mesh.* We meshed the _same_ sponge at three resolutions:
#align(center, table(columns: 4, align: center,
  [`res`], [nodes], [mean T (°C)], [max T (°C)],
  [6], [556], [56.4], [105.2],
  [12], [2184], [55.7], [94.4],
  [25], [13 434], [55.3], [87.5]))
The mean barely moves (1 °C), but the hot spot moves by 18 °C between `res` 6 and 25. Our training
data (`res` 5–10) is therefore _not converged_: its hot spots are wrong by several degrees compared
to a fine mesh. A model can never be more accurate than its data, and comparing predictions with
coarse-mesh "truth" has an error floor of this size.]

#figure(grid(columns: 3, gutter: 4pt, img("sponge_res_6.png"), img("sponge_res_12.png"), img("sponge_res_25.png")),
  caption: [The same sponge meshed with `res` = 6, 12 and 25 (556, 2184 and 13 434 nodes).],
) <fig-sponge-res>

