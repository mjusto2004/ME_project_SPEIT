#import "../common.typ": *

= Models for random sponges <sec-sponge>

#idea(title: "Why new models")[None of the cube models can be used on sponges. The MLP and POD need a
fixed list of nodes. The coordinate MLP could predict at any point, but it only sees $(x, y, z)$: it
cannot know that a hole between the point and the heated face blocks the heat. A model for sponges
must look at the *neighbourhood* of each node and pass information through the part, the way heat
does. Graphs, and neural networks on graphs, do exactly that.]

== Baseline

Predicting the mean training temperature (#calc.round(sstats.T_mean, digits: 1) °C) everywhere gives
an MAE of #sm("mean", "split") °C on the 150 test sponges. Sponges vary much more than the cube: the
hottest node ranges from 53 to 270 °C (@fig-sponge-stats).

== Graph neural network (GNN)

#idea(title: "Why")[Heat flows from node to neighbouring node. A _message-passing_ network copies this:
each node holds a vector of numbers (its _state_); at every round it receives a message from each
neighbour and updates its state. After $L$ rounds, a node has information from every node up to $L$
links away. The same small networks are used at every node, so one model works on any mesh.]

*How it works* (`models/gnn.py`, plain PyTorch).
+ An MLP _encodes_ the 9 inputs of each node into a state of 64 numbers.
+ $L$ times: for every link $j -> i$, an MLP computes a message from the states of $i$ and $j$; node
  $i$ averages its messages, and another MLP updates its state ($h_i <- h_i + "update"(h_i, "mean message")$).
+ A linear layer reads the temperature from the final state.

We trained two versions, with $L = 4$ rounds (142 k parameters) and $L = 8$ rounds (274 k), with
batches of 4 sponges and Adam with a cosine schedule over 100 epochs (about 15 and 25 minutes on the GPU).

*Results.* On the 150 test sponges (resolutions 5–10, like training), the 4-round GNN reaches
*#sm("gnn", "split") °C* (CI #ci("gnn", "split")), i.e. #nm("gnn", "split") of the temperature range of
each sponge, 4.5 times better than the baseline. With 8 rounds it improves to
*#sm("gnn8", "split") °C* (CI #ci("gnn8", "split"), #nm("gnn8", "split")), the best of all our sponge
models on meshes like the training ones. More rounds mean that each node sees further (8 links
instead of 4), and in a sponge the heat can come from far away along a winding path.

== Graph neural operator (GNO)

#idea(title: "Why")[A GNN counts in _links_, not in metres. With `res` = 5 a link is about 20 cm long,
with `res` = 25 about 4 cm. Four rounds therefore "see" 80 cm on a coarse mesh but only 16 cm on a fine
one, and an average over the neighbours changes meaning when a node has 40 neighbours instead of 10. We
expected the GNN to fail on finer meshes, and finer meshes are exactly what a user would give it to get
more detail. A _neural operator_ is designed to be independent of the mesh: it works with physical
distances.]

*How it works* (`models/gno.py`, after Li et al. 2020, "Neural Operator: Graph Kernel Network").
- The neighbours of a node are the nodes within a ball of radius 0.15 m, whatever the mesh. Six
  rounds therefore reach 0.9 m at any resolution.
- A message depends on the states of both nodes and on their _relative position_
  $(Delta x, Delta y, Delta z) \/ 0.15$ and distance. Averaging the messages estimates an integral
  over the ball, and an integral does not depend on how finely the ball is sampled.
- On a fine mesh a ball holds hundreds of nodes; at most 24 are drawn at random (a new draw at every
  training step, which also acts as a regulariser).
- At every round, the mean state of the whole sponge is given back to every node, so every node knows
  about the whole part (think of the global heat balance).
- The input `res` is ignored: the answer must not depend on the mesh.

It has 932 k parameters (states of 128 numbers, 6 rounds); 100 epochs take about 20 minutes on the
GPU (12 s per epoch, 6.8 GB of GPU memory with batches of 4 sponges).

*Results.* On the test split, the GNO reaches *#sm("gno", "split") °C* (CI #ci("gno", "split")),
#nm("gno", "split") of the range: between the two GNNs, and the confidence intervals overlap. An
earlier GNO run of the project, on an older sponge dataset, gave the same level (3.3 °C), so this
number is stable. On meshes like the training ones, the GNO buys nothing. Its value appears in the
next test.

#figure(img("sponge_training_curves.png", w: 82%),
  caption: [Training (solid) and validation (dashed) losses of the graph models. All three overfit:
  the training loss keeps falling while the validation loss stalls after 40–80 epochs.],
) <fig-sponge-train>

== Changing the mesh resolution

This is the experiment the GNO was built for. All models were trained on `res` 5–10 only, then
tested on new sponges meshed at `res` 7 (inside the training range), 12 (a bit finer), 18 and 25
(2.5 times finer than anything seen, about 14 000 nodes per sponge).

#figure(
  grid(columns: (1.45fr, 1fr), gutter: 8pt, align: horizon,
    img("sponge_resolution.png"),
    {
      set text(9pt)
      table(columns: 5, align: (left, right, right, right, right),
        [`res`], [mean], [GNN-4], [GNN-8], [GNO],
        ..("7", "12", "18", "25").map(k => (k, sm("mean", k), sm("gnn", k), sm("gnn8", k), [*#sm("gno", k)*])).flatten())
    }),
  caption: [MAE per simulation (°C) against the mesh resolution of the test sponges. Grey band:
  resolutions seen in training. Bars: 95 % confidence intervals.],
) <fig-sponge-resolution>

The result is clear (@fig-sponge-resolution):

- *Both GNNs break down on fine meshes*, and in opposite directions. At `res` 25 the 4-round GNN is
  on average #fmt(calc.abs(S("gnn", "25").bias), d: 1) °C too cold, and the 8-round GNN
  #fmt(S("gnn8", "25").bias, d: 1) °C too hot. Both are _worse than predicting a constant_. The
  better GNN on coarse meshes (8 rounds) is the worse one on fine meshes: it relies even more on
  "how many links", which is exactly what changes.
- *The GNO degrades gently*: from #sm("gno", "7") °C to #sm("gno", "25") °C at `res` 25, i.e. 4 times
  better than the 4-round GNN and 14 times better than the 8-round GNN there. Its error stays near 8 %
  of the temperature range.

@fig-sponge-map25 shows one sponge at `res` 25. The GNN errors cover the whole part (a uniform shift);
the GNO error is small almost everywhere, and concentrated next to the heated face, where the field
changes fastest.

#figure(
  grid(columns: 4, gutter: 2pt,
    img("sponge_map_res25_true.png"), img("sponge_map_res25_err_gnn.png"), img("sponge_map_res25_err_gnn8.png"), img("sponge_map_res25_err_gno.png"),
    align(center, text(8.5pt)[true T (°C)]), align(center, text(8.5pt)[GNN-4 error]),
    align(center, text(8.5pt)[GNN-8 error (saturated)]), align(center, text(8.5pt)[GNO error])),
  caption: [One test sponge at `res` 25 (11 197 nodes), the median case of the GNO. Left: true
  temperature, 20 °C (black) to 55 °C (yellow). Errors: blue = too cold, red = too hot, colour scale
  ±30 °C. MAE: GNN-4 15.2 °C, GNN-8 80.5 °C, GNO 3.1 °C.],
) <fig-sponge-map25>

#idea(title: "Why is the GNO not perfect at res 25?")[Two reasons, which we could separate.
First, the "truth" itself changes with the mesh (@sec-sponge-data): the model learned the
coarse-mesh answer. Second, the GNO is slightly _too cold_ on fine meshes (bias
#fmt(S("gno", "25").bias, d: 1) °C at `res` 25). Our explanation: its global state is an average over
_nodes_, not over _volume_. gmsh puts relatively more nodes inside a fine mesh than on its faces, so the
share of nodes on the heated face falls from 10.3 % (training) to 5.6 % (`res` 25). The global average
then "sees" less heating than there really is. Weighting each node by its volume would fix this; it is
the next thing to try.]

== Where the errors are

#figure(img("sponge_where_gno.png"),
  caption: [GNO errors on the 150 test sponges. Left: error against cooling $h$ and hole radius $r$.
  Right: error against the temperature range of the sponge.],
) <fig-sponge-where>

The same pattern as for the cube appears (@fig-sponge-where): the hardest sponges have *weak cooling and
big holes*. The correlation of the error with $r$ is +0.55 and with $h$ −0.44. Big holes make thin
walls and isolated regions that heat up a lot, and weak cooling keeps the part far from steady state.
The error grows with the temperature range of the sponge (correlation +0.67), so the relative error
stays about 5 %. The models struggle with the sharp, local hot spots, which are also where the
coarse-mesh data is least accurate.

#limit[
- *Accuracy*: 3 °C (about 5 % of the range) is 30 to 50 times worse than the MLP on the cube. Variable
  geometry is a much harder problem, and 1000 sponges are few for it (all models overfit).
- *Data quality caps the score*: the training data comes from non-converged coarse meshes, and contains
  rare numerical artefacts. A better model cannot beat the error of its data.
- *GNNs do not transfer across meshes*: they must be used on meshes like the training ones.
- *GNO*: a small cold bias on fine meshes (node average instead of volume average), a higher cost
  (neighbour search, 6.8 GB of GPU memory with batches of 4), and random neighbour sampling, which
  makes its predictions depend slightly on the draw.
- *Scope*: one material, one heated face, spherical holes, and one output time. Real parts would need
  more varied geometry, and probably the whole time history.
]

