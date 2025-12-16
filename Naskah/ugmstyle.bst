% ugmstyle.bst
% Custom style based on plainnat.bst
% Modified: 2025-09-18
% Features:
%  - Hyperlinked title (URL if available, else DOI)
%  - APA-like sorting (author, year)
%  - Compatible with natbib

ENTRY
  { address
    author
    booktitle
    chapter
    doi
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
    url
  }
  {}
  { label extra.label sort.label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

STRINGS  { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

% --- Hyperlinked title (url > doi > plain) ---
FUNCTION {format.title}
{ title empty$
    { "" }
    { url empty$
        { doi empty$
            { title "t" change.case$ }
            { "\href{\detokenize{https://doi.org/" doi * "}}{" * title * "}" * }
          if$
        }
        { "\href{\detokenize{" url * "}}{" * title * "}" * }
      if$
    }
  if$
}

FUNCTION {format.date}
{ year empty$
    { "" }
    { year }
  if$
}

% ---- ENTRY TYPES ----
FUNCTION {article}
{ output.bibitem
  author output
  format.title output
  new.block
  journal output
  volume output
  number output
  pages output
  format.date output
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author output
  editor output
  format.title output
  new.block
  publisher output
  address output
  format.date output
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author output
  editor output
  format.title output
  chapter output
  new.block
  publisher output
  address output
  format.date output
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  author output
  format.title output
  new.block
  booktitle output
  editor output
  pages output
  organization output
  publisher output
  address output
  format.date output
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  author output
  format.title output
  howpublished output
  format.date output
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  author output
  format.title output
  new.block
  institution output
  number output
  address output
  format.date output
  note output
  fin.entry
}

FUNCTION {phdthesis}
{ output.bibitem
  author output
  format.title output
  new.block
  school output
  address output
  format.date output
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  author output
  format.title output
  new.block
  school output
  address output
  format.date output
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  author output
  format.title output
  note output
  format.date output
  fin.entry
}

READ
EXECUTE {initialize.extra.label}
ITERATE {call.type$}
SORT
EXECUTE {finish.extra.label}
