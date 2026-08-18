"""Classical design-of-experiments analysis on named responses.

The two-stage Weibull model elsewhere in this pipeline exists for one concrete
reason: three parameters reconstruct a whole predicted *curve*, which the
formulator tool needs to draw a profile and compute f2. It is a curve generator.

It is not what anyone should have to read. ``Td``, ``beta`` and ``F_inf`` are
latent parameters; a formulator thinks in "time to 50% released" and "% released
at 12 hours". This package fits ordinary response surfaces directly on those
named quantities and produces the analysis a Minitab user expects -- ANOVA,
effect estimates, main-effect and interaction plots, contour and 3D surfaces --
so that every headline number has units someone can act on.
"""
