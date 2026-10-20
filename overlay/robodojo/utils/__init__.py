"""Make the RoboDojo top-level ``utils`` package win over OpenCV's.

``utils`` used to be a PEP 420 namespace package here (no ``__init__.py``), and
that is fragile in a way that silently breaks the evaluation client:

  * Kit loads ``cv2`` during SimulationApp start-up. OpenCV's ``cv2/__init__.py``
    appends its own directory to ``sys.path`` -- that directory contains a
    *regular* package ``cv2/utils/``.
  * A regular package beats a namespace package's portions **regardless of
    ``sys.path`` order**, so ``import utils`` started resolving to
    ``<...>/pip_prebundle/cv2/utils`` and ``from utils.load_file import *``
    raised ``ModuleNotFoundError``. Reordering ``sys.path`` cannot fix that;
    ``sys.path.insert(0, ...)`` does not help.
  * Before kit reaches cv2 the import works, which is why this looks like a
    random "kit breaks Python imports" bug rather than a name collision.

Adding this file turns ``utils`` into a regular package, so it wins. The
``__path__`` extension below keeps the previous namespace behaviour intact --
``XPolicyLab/utils`` was previously merged in as a second portion, and code that
reaches for ``utils.<something>`` defined there still resolves (RoboDojo never
relies on this for the OpenDM evaluation path, but the policies vendored under
XPolicyLab/policy/* do, and there is no reason to break them).

Verified: ``cv2`` never imports ``utils`` itself (0 occurrences in the whole cv2
prebundle), and no Isaac Sim extension does either, so giving RoboDojo priority
cannot break them.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTRA_PORTION = os.path.join(_REPO_ROOT, "XPolicyLab", "utils")
if os.path.isdir(_EXTRA_PORTION) and _EXTRA_PORTION not in __path__:
    __path__.append(_EXTRA_PORTION)
