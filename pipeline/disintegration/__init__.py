"""Disintegration (matrix-erosion) time, analysed against dissolution and the recipe.

The section is optional: a workbook without a ``Disintegration`` sheet runs
exactly as before. With one, the replicate disintegration times are summarised,
correlated with the dissolution responses, compared across HPMC grades at
identical composition, fitted with the same classical DoE as the dissolution
responses, and written out as a report, diagnostics, an audit section and
journal-style figures.

    python -m pipeline.disintegration --input <workbook.xlsx>
"""
