"""Build-time check for docker/patch/sitecustomize.py.

Runs without a GPU and without kit, which is the whole point: the real
``omni.physics.tensors.api`` only becomes importable after ``SimulationApp``
has booted, so the alias finder cannot be exercised against the real module
during an image build. Instead the minimal package chain kit would provide is
faked with stub modules, and the finder is asked to resolve the old path from
the new one.

The assertions below pin the contract Isaac Lab 2.3.2 depends on, not the
internal shape of the shim:

  * the old path imports and both attribute and from-import access reach the
    real module -- six Isaac Lab files do
    ``import omni.physics.tensors.impl.api as physx``;
  * the alias is a live view rather than a copy of the target. A duplicated
    module would hand Isaac Lab two distinct copies of every view class, which
    breaks ``isinstance`` checks in ways that are miserable to trace later;
  * the five names Isaac Sim 6.0 deleted are present through the alias, and
    instantiating one raises. Isaac Lab 2.3.2 evaluates them while importing --
    deformable_object_data.py annotates ``root_physx_view: physx.SoftBodyView``
    and does *not* enable postponed evaluation -- but never instantiates them;
  * the real ``omni.physics.tensors.api`` is left untouched, so any
    ``hasattr``-based feature detection inside Isaac Sim still sees 6.0's
    actual surface rather than a padded one;
  * the alias finder sits behind ``PathFinder``, so a real ``impl`` package
    wins again if a future Isaac Sim restores it.

Exits non-zero on failure so a broken shim fails the build rather than the
first evaluation run.
"""

import importlib
import importlib.machinery
import sys
import types

STUB_SPEC = "stub-spec-of-omni.physics.tensors.api"
OLD_API = "omni.physics.tensors.impl.api"
OLD_PACKAGE = "omni.physics.tensors.impl"
NEW_API = "omni.physics.tensors.api"

# Deleted in Isaac Sim 6.0; Isaac Lab 2.3.2 still names them in annotations.
REMOVED = (
    "ParticleClothView",
    "ParticleMaterialView",
    "ParticleSystemView",
    "SoftBodyView",
    "SoftBodyMaterialView",
)
STUB_PROBE = "SoftBodyView"


def _install_stub_chain():
    for name in ("omni", "omni.physics", "omni.physics.tensors"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    api = types.ModuleType(NEW_API)
    api.MARKER = 42
    api.__spec__ = STUB_SPEC
    sys.modules[NEW_API] = api
    sys.modules["omni.physics.tensors"].api = api


def main():
    _install_stub_chain()

    import sitecustomize  # noqa: F401  (imported for its side effect)

    import omni.physics.tensors.impl.api as aliased
    from omni.physics.tensors.impl.api import MARKER

    package = importlib.import_module(OLD_PACKAGE)
    canonical = sys.modules[NEW_API]

    # A snapshot copy would answer 42 here too, so change the target underneath
    # and require the alias to follow.
    canonical.MARKER = 99
    alias_is_live = getattr(aliased, "MARKER", None) == 99
    canonical.MARKER = 42

    try:
        getattr(aliased, STUB_PROBE)()
        stub_raises = False
    except NotImplementedError:
        stub_raises = True
    except Exception:  # noqa: BLE001 - any other error is also a failure
        stub_raises = False

    # Note: sys.meta_path mixes instances (ours) with bare classes (PathFinder
    # and friends are classes of classmethods and are placed there as-is), so
    # compare the entries themselves rather than type(finder).
    finders = list(sys.meta_path)
    alias_finder = getattr(sitecustomize, "_AliasFinder", None)
    alias_at = next(
        (i for i, f in enumerate(finders) if alias_finder is not None and isinstance(f, alias_finder)),
        None,
    )
    path_finder_at = next(
        (i for i, f in enumerate(finders) if f is importlib.machinery.PathFinder),
        None,
    )
    alias_behind_path_finder = (
        alias_at is not None and path_finder_at is not None and alias_at > path_finder_at
    )

    checks = [
        ("attribute reached through the alias", aliased.MARKER == 42),
        ("from-import through the alias", MARKER == 42),
        (
            "alias registered under the old name in sys.modules",
            sys.modules.get(OLD_API) is aliased and aliased.__name__ == OLD_API,
        ),
        (
            "impl package resolves and exposes .api",
            sys.modules.get(OLD_PACKAGE) is package and getattr(package, "api", None) is aliased,
        ),
        ("alias is a live view, not a snapshot", alias_is_live),
        ("deleted 6.0 names are present through the alias", all(hasattr(aliased, n) for n in REMOVED)),
        (f"instantiating a deleted name ({STUB_PROBE}) raises", stub_raises),
        (
            "canonical module untouched (no stubs injected)",
            canonical.__spec__ == STUB_SPEC
            and canonical.MARKER == 42
            and not any(hasattr(canonical, n) for n in REMOVED),
        ),
        ("alias finder loses to a real module", alias_behind_path_finder),
    ]

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    if failed:
        print(f"alias selftest FAILED: {', '.join(failed)}")
        return 1
    print("alias selftest PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
