// Report of the ME project SPEIT: deep-learning surrogates for 3D heat transfer.
// Build: typst compile report/report.typ   (from the repository root, or with --root .)
// Content is split into chapters/*.typ; shared data and helpers live in common.typ.
// Layout mimics the LaTeX `report` class: Computer Modern, "Chapter N" headings, plain page style.

#import "common.typ": *

#let title = [Learning 3D temperature fields with neural surrogates]

#set document(title: "Learning 3D temperature fields with neural surrogates", author: "ME project SPEIT team")

// page: running header with the chapter name (LaTeX `headings` style), plain on chapter openings
#set page(paper: "a4", margin: (x: 2.5cm, top: 3cm, bottom: 2.8cm), numbering: "1",
  header: context {
    let here_ = here().page()
    let starts = query(heading.where(level: 1)).filter(h => h.location().page() == here_)
    if starts.len() > 0 { return }
    let prev = query(heading.where(level: 1).before(here()))
    if prev.len() == 0 { return }
    let hd = prev.last()
    set text(9pt)
    if hd.numbering != none [
      #smallcaps[Chapter #counter(heading).at(hd.location()).first(). #hd.body]
    ] else [#smallcaps(hd.body)]
    h(1fr)
    counter(page).display()
    v(-0.6em)
    line(length: 100%, stroke: 0.4pt)
  },
  footer: context {
    let starts = query(heading.where(level: 1)).filter(h => h.location().page() == here().page())
    if starts.len() > 0 { align(center, text(10pt, counter(page).display())) }
  },
)

#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.55em, spacing: 0.55em, first-line-indent: 1.5em)
#show math.equation: set text(font: "New Computer Modern Math")
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
#show link: set text(fill: black)
#show ref: set text(fill: black)
#set list(indent: 1em, spacing: 0.55em)
#set enum(indent: 1em, spacing: 0.55em)
#show list: set block(above: 0.8em, below: 0.8em)
#show enum: set block(above: 0.8em, below: 0.8em)
#show math.equation.where(block: true): set block(above: 1em, below: 1em)

// headings -----------------------------------------------------------------
#set heading(numbering: "1.1")
#show heading: set text(weight: "bold")
#show heading.where(level: 1): set heading(supplement: [Chapter])
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  block(above: 2.5cm, below: 2em, {
    set par(first-line-indent: 0pt)
    if it.numbering != none {
      text(17pt)[Chapter #counter(heading).display()]
      v(0.9em)
    }
    text(21pt, it.body)
  })
}
#show heading.where(level: 2): it => block(above: 1.6em, below: 1em, text(13pt, it))
#show heading.where(level: 3): it => block(above: 1.3em, below: 0.8em, text(11pt, it))

// figures and tables -------------------------------------------------------
#set figure(gap: 0.8em)
#show figure: set block(above: 1.4em, below: 1.4em)
#show figure.caption: it => {
  set text(10pt)
  set par(first-line-indent: 0pt, justify: true)
  block(width: 90%, [#it.supplement~#context it.counter.display(it.numbering)#it.separator#it.body])
}
#set figure.caption(separator: [: ])
#show figure.where(kind: table): set figure.caption(position: top)
#show table: set par(justify: false, first-line-indent: 0pt)
#show table: set text(10pt)
// booktabs: \toprule, \midrule, \bottomrule
#set table(
  stroke: (x, y) => (
    top: if y == 0 { 0.8pt } else if y == 1 { 0.5pt } else { none },
  ),
  inset: (x: 6pt, y: 4pt),
)
#show table: t => block(stroke: (bottom: 0.8pt), t)

// outline ------------------------------------------------------------------
#show outline.entry.where(level: 1): it => {
  v(0.9em, weak: true)
  strong(link(it.element.location(), it.indented(it.prefix(), [#it.body() #h(1fr) #it.page()])))
}

// title page (\maketitle with the titlepage option) ------------------------
// sjtubannerred.svg is a full A4 page with the SJTU logo in its upper third: use it as the background
#page(numbering: none, header: none, footer: none,
  background: align(top + center, pad(top: 0.5cm, image("sjtubannerred.svg", width: 60%))))[
  #set align(center)
  #set par(first-line-indent: 0pt)
  #v(6.5cm)
  #text(20pt)[Learning 3D temperature fields \ with neural surrogates]
  #v(1.2em)
  #text(13pt)[From a fixed cube to random porous parts: \ what we built, what we tried, what worked and why]
  #v(2.5em)
  #text(12pt)[ME project SPEIT]
  #v(1em)
  #text(12pt)[#datetime(year: 2026, month: 11, day: 19).display("[month repr:long] [day], [year]")]
  #v(1fr)
  #block(width: 80%)[
    #set text(10pt)
    #set par(justify: true)
    Mathieu Senart, Marc Justo, Tristan Jin, Amelia Soare, Anoj Nagarajah
  ]
  #v(2cm)
]

#outline(title: [Contents], indent: auto, depth: 2)

#include "chapters/01-introduction.typ"
#include "chapters/02-product.typ"
#include "chapters/03-data.typ"
#include "chapters/04-protocol.typ"
#include "chapters/05-cube.typ"
#include "chapters/06-sponge.typ"
#include "chapters/07-conclusion.typ"
#include "chapters/08-appendix.typ"
