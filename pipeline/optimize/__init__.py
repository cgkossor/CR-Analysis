"""Goal-driven formulation search.

The inverse problem a formulator actually has is not "choose three components
that sum to 100". Dose fixes the drug load, so the real question is: given that
load, which HPMC grade and how much of it?

That reduces the search to two degrees of freedom -- grade and polymer content --
with lactose as the balance. The composition then sums correctly by construction
and there is nothing to balance by hand.
"""
