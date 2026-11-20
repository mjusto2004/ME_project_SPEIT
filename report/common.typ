// Shared data, colours and helpers for the report. Imported by report.typ and every chapter.

#let cube = json("results/cube_learning_curve.json")
#let limits = json("results/cube_limits.json")
#let sponge = json("results/sponge_results.json")
#let sstats = json("results/sponge_stats.json")

// sober, LaTeX-like palette: everything is black
#let accent = black
#let soft = none

// helpers -----------------------------------------------------------------
#let fmt(x, d: 3) = {
  if x == none { "–" } else if calc.abs(x) >= 0.01 or x == 0 { str(calc.round(x, digits: d)) } else {
    // scientific notation for tiny values
    let e = calc.floor(calc.log(calc.abs(x)))
    let m = x / calc.pow(10.0, e)
    [#calc.round(m, digits: 1)#sym.times;10#super[#e]]
  }
}
#let get(model, n) = cube.find(r => r.model == model and r.n == n)
// framed remarks, in the spirit of a LaTeX `remark` environment inside a thin \fbox
#let remark(title, body) = block(width: 100%, inset: (x: 8pt, y: 7pt), stroke: 0.4pt + black,
  above: 1.1em, below: 1.1em)[
  #set par(first-line-indent: 0pt)
  #strong[#title.] #body
]
#let idea(title: "Remark", body) = remark(title, body)
#let limit(body) = remark(emph[Limits], body)
#let img(path, w: 100%) = image("figures/" + path, width: w)

// sponge results helpers
#let S(m, k) = sponge.at(m).at(k)
#let sm(m, k) = fmt(S(m, k).sim_MAE, d: 2)
#let ci(m, k) = [#fmt(S(m, k).sim_MAE_lo, d: 2) – #fmt(S(m, k).sim_MAE_hi, d: 2)]
#let nm(m, k) = [#calc.round(100 * S(m, k).NMAE, digits: 1) %]
