// Report of the ME project SPEIT: deep-learning surrogates for 3D heat transfer.
// Build: typst compile report/report.typ   (from the repository root, or with --root .)
// Content is split into chapters/*.typ; shared data and helpers live in common.typ.

#import "common.typ": *

#set document(title: "Learning 3D temperature fields with neural surrogates", author: "ME project SPEIT team")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1",
  header: context { if counter(page).get().first() > 1 [
    #set text(8pt, fill: gray.darken(30%))
    ME project SPEIT #h(1fr) Neural surrogates for 3D heat transfer
  ] })
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { pagebreak(weak: true); v(0.4em); text(16pt, fill: accent, it); v(0.3em) }
#show heading.where(level: 2): it => { v(0.5em); text(12pt, fill: accent, it); v(0.15em) }
#show heading.where(level: 3): it => { v(0.3em); text(10.5pt, style: "italic", it) }
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
#show figure.caption: set text(9pt)
#set figure(gap: 0.6em)
#show link: set text(fill: accent)
#show table: set par(justify: false)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { none }, inset: (x: 5pt, y: 3.5pt))
#show table.cell.where(y: 0): set text(weight: "bold")

// title page ---------------------------------------------------------------
#page(numbering: none, header: none)[
  #v(3cm)
  #text(11pt, fill: gray.darken(20%))[ME project · SPEIT]
  #v(0.3cm)
  #text(26pt, weight: "bold", fill: accent)[Learning 3D temperature fields \ with neural surrogates]
  #v(0.4cm)
  #text(13pt)[From a fixed cube to random porous parts: \ what we built, what we tried, what worked and why]
  #v(1.2cm)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 6pt,
    img("cube_field_0.png"), img("sponge_example_2.png"), img("sponge_example_6.png"))
  #v(1fr)
  #text(10pt)[Experiment report — #datetime(year: 2026, month: 11, day: 19).display("[month repr:long] [day], [year]")]
  #v(0.2cm)
  #text(9pt, fill: gray.darken(20%))[All numbers in this report were produced by the scripts in `report/scripts/`,
  from datasets that can be regenerated with the commands of the appendix.]
]

#outline(indent: auto, depth: 2)

#include "chapters/01-introduction.typ"
#include "chapters/02-product.typ"
#include "chapters/03-data.typ"
#include "chapters/04-protocol.typ"
#include "chapters/05-cube.typ"
#include "chapters/06-sponge.typ"
#include "chapters/07-conclusion.typ"
#include "chapters/08-appendix.typ"
