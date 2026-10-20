"""Compatibility shims for running the RoboDojo stack on Isaac Sim 6.0.

Loaded automatically by CPython at interpreter start-up because this directory
sits on ``PYTHONPATH`` (see ``docker/robodojo-env.sh``).

Two independent incompatibilities are bridged here. Both were reproduced on this
cluster; neither is hypothetical.

1. ``omni.physics.tensors.impl`` does not exist in Isaac Sim 6.0
---------------------------------------------------------------
Isaac Sim 6.0.1 removed the ``impl`` level from the physics-tensors bindings:

    Isaac Sim 5.1.0 : omni/physics/tensors/impl/api.py
    Isaac Sim 6.0.1 : omni/physics/tensors/api.py

The file set is otherwise identical and every class/method signature RoboDojo
touches compares equal. Isaac Lab 2.3.2 still imports the old path at module
scope in six files, so without a bridge ``import isaaclab`` dies.

A meta-path finder resolves the old name lazily. Laziness is required, not
stylistic: ``omni.physics.tensors.api`` lives in
``/isaac-sim/extscache/omni.physics.tensors-<ver>/`` and only reaches
``sys.path`` once ``SimulationApp`` has booted kit and enabled that extension.

Six of the names in the old module were deleted outright in 6.0
(``ParticleClothView``, ``ParticleMaterialView``, ``ParticleSystemView``,
``SoftBodyView``, ``SoftBodyMaterialView``). Isaac Lab 2.3.2 imports them, so the
alias hands back loudly-failing placeholders instead of missing attributes --
``import isaaclab_tasks`` otherwise aborts extension start-up with
"module ... has no attribute 'SoftBodyView'". Anything that actually
*instantiates* one raises immediately; nothing in the RoboDojo evaluation path
does (the cloth/fluid tasks go through isaacsim.core.prims.SingleClothPrim and
SingleParticleSystem, which still exist in 6.0 under extsDeprecated).

2. OpenCV hijacks the top-level name ``utils``
----------------------------------------------
``cv2/__init__.py`` inserts its own package directory into ``sys.path``:

    sys.path.insert(1 if not applySysPathWorkaround else 0, p)

That directory contains a *regular* package ``cv2/utils/``. RoboDojo's
``utils/`` is a PEP 420 namespace package, and a regular package wins over a
namespace portion **regardless of ordering and regardless of where each sits on
``sys.path``**. So ``from utils.load_file import *`` -- used by
``src/eval_client/main.py``, ``src/eval_client/eval_env.py`` and several
``env/*_manager`` modules -- fails with ``ModuleNotFoundError`` once kit has
loaded cv2, which it does during SimulationApp start-up.

``sys.path.insert(0, ...)`` does not fix this and neither does adding an
``__init__.py`` to ``utils/``: cv2 inserts at index 1, ahead of the project root,
and a later regular package still beats an earlier namespace portion. The only
order-independent fix is to claim the name before the path-based finder runs.
``_ProjectTreeFinder`` below is prepended to ``sys.meta_path`` and does exactly
that, for the two top-level names the RoboDojo tree owns.

Verified to be safe: cv2 never imports ``utils`` itself (0 occurrences across
the whole cv2 prebundle) and no Isaac Sim extension does either.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import functools
import inspect
import os
import sys
import types

# ─────────────────────────────────────────────────────────────────────────────
# 1. omni.physics.tensors.impl -> omni.physics.tensors
# ─────────────────────────────────────────────────────────────────────────────

_ALIASES = {
    "omni.physics.tensors.impl": ("omni.physics.tensors", True),
    "omni.physics.tensors.impl.api": ("omni.physics.tensors.api", False),
}

# Deleted in Isaac Sim 6.0; Isaac Lab 2.3.2 still imports them. Placeholders keep
# the imports working and fail loudly if one is ever really used.
_REMOVED_IN_6_0 = (
    "ParticleClothView",
    "ParticleMaterialView",
    "ParticleSystemView",
    "SoftBodyView",
    "SoftBodyMaterialView",
)


def _make_removed_stub(name):
    def _init(self, *args, **kwargs):
        raise NotImplementedError(
            f"{name} was removed in Isaac Sim 6.0 and is not available in this "
            f"image. It is stubbed only so that Isaac Lab 2.3.2 can be imported; "
            f"the RoboDojo evaluation path does not use deformables or particles "
            f"(those tasks go through isaacsim.core.prims instead)."
        )

    return type(name, (), {"__init__": _init, "__doc__": f"Removed in Isaac Sim 6.0 ({name})."})


class _CompatProxy(types.ModuleType):
    """Alias module that forwards to the real one and stubs the deleted names."""

    def __init__(self, name, target_name):
        super().__init__(name)
        self.__dict__["_target_name"] = target_name
        self.__dict__["_stubs"] = {}

    def _target(self):
        return importlib.import_module(self.__dict__["_target_name"])

    def __getattr__(self, item):
        try:
            return getattr(self._target(), item)
        except AttributeError:
            if item in _REMOVED_IN_6_0:
                stubs = self.__dict__["_stubs"]
                if item not in stubs:
                    stubs[item] = _make_removed_stub(item)
                return stubs[item]
            raise


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, fullname, target_name, is_package):
        self._fullname = fullname
        self._target_name = target_name
        self._is_package = is_package

    def create_module(self, spec):
        module = _CompatProxy(self._fullname, self._target_name)
        if self._is_package:
            module.__path__ = []
        return module

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        entry = _ALIASES.get(fullname)
        if entry is None:
            return None
        target_name, is_package = entry
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, target_name, is_package),
            is_package=is_package,
        )



# ─────────────────────────────────────────────────────────────────────────────
# 2. Isaac Sim 6.0 renames inside modules RoboDojo imports
# ─────────────────────────────────────────────────────────────────────────────

# RoboDojo was written against Isaac Sim 5.1 and reaches for these names, which
# 6.0 moved or renamed. Unlike the two cases above, the fix has to be applied
# *after* the real module has been executed -- ``omni.physx`` is a normal
# extension package whose contents are produced by its own loader, so there is
# nothing to synthesise ahead of time, only a missing attribute to add.
#
# Mapping is ``module -> {old name: new name}``; the old name is only added when
# it is genuinely absent, so a future Isaac Sim that restores it still wins.
_RENAMED_IN_6_0 = {
    # env/environment/base_env.py:10, used at line 196 as acquire_physx_interface()
    "omni.physx": {"acquire_physx_interface": "get_physx_interface"},
}


# Names Isaac Sim 6.0 does not provide at all. Separate from the renames above
# because there is nothing to resolve *to*: the physx USD deformable schema was
# replaced by the OmniPhysics* family registered by the
# ``omniUsdPhysicsDeformableSchema`` plugin, and the legacy classes are simply
# gone from ``pxr.PhysxSchema``.
#
# Mapping is ``module -> {absent name: why}``.
_ABSENT_IN_6_0 = {
    "pxr.PhysxSchema": {
        "PhysxDeformableBodyAPI": "6.0 replaced the deformable schema with the OmniPhysics* family",
        "PhysxDeformableBodyMaterialAPI": "6.0 replaced the deformable schema with the OmniPhysics* family",
        "PhysxDeformableAPI": "6.0 replaced the deformable schema with the OmniPhysics* family",
    },
}


class _AbsentSchemaAPI(str):
    """Stand-in for a USD API schema that Isaac Sim 6.0 no longer binds.

    Subclasses ``str`` because that is exactly what ``Usd.Prim.HasAPI`` accepts
    for a lookup by name, and because name lookup is the *only* thing the
    evaluation path does with these classes::

        prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)   # -> False

    False is the truthful answer here: the schema is not registered in 6.0, so
    no prim can carry it. What is *not* truthful is silently pretending an
    operation succeeded, so every other entry point (constructing the schema
    wrapper for a prim, ``Apply``) fails loudly with the reason. Those paths are
    reached only by the deformable spawners, which this benchmark does not use.
    """

    def __new__(cls, name, reason):
        instance = super().__new__(cls, name)
        instance._reason = reason
        return instance

    def _unsupported(self, action):
        raise NotImplementedError(
            f"Isaac Sim 6.0 does not provide the PhysX USD schema {str(self)!r} "
            f"({self._reason}). '{action}' cannot be performed. If a task really "
            f"needs deformable bodies, it has to be ported to the OmniPhysics* "
            f"schema names instead."
        )

    def __call__(self, *args, **kwargs):
        self._unsupported("constructing the schema wrapper")

    def Apply(self, *args, **kwargs):  # noqa: N802 - USD API naming
        self._unsupported("Apply")


def _install_absent_members(module, members):
    """Install the stand-ins, without ever shadowing a name that exists."""
    for name, reason in members.items():
        if name in module.__dict__:
            continue
        setattr(module, name, _AbsentSchemaAPI(name, reason))
        sys.stderr.write(
            f"[robodojo-compat] {module.__name__}.{name}: 6.0 已移除，装入缺席替身（{reason}）\n"
        )


def _install_lazy_renames(module, repairs):
    """Answer to the old names through a module-level ``__getattr__``.

    Deliberately lazy rather than a one-shot ``setattr``. An extension's
    ``__init__.py`` is executed by kit *before* its ``startup()`` runs, and
    ``omni.physx.get_physx_interface`` only appears once that startup has
    happened -- so copying it during ``exec_module`` silently finds nothing and
    can never retry. A ``__getattr__`` fallback is only consulted when normal
    lookup fails, so it costs nothing when the name exists and resolves the
    rename whenever the target finally shows up.
    """
    existing = module.__dict__.get("__getattr__")

    def __getattr__(name, _repairs=repairs, _existing=existing, _module=module):
        replacement = _repairs.get(name)
        if replacement is not None:
            try:
                resolved = getattr(_module, replacement)
            except AttributeError:
                pass
            else:
                sys.stderr.write(
                    f"[robodojo-compat] {_module.__name__}.{name} -> {replacement} "
                    f"(renamed in Isaac Sim 6.0)\n"
                )
                # Materialise the result so later lookups take the normal path
                # instead of re-entering this fallback (and re-logging). If the
                # lookup failed above nothing is cached, so a name that appears
                # even later is still picked up.
                setattr(_module, name, resolved)
                return resolved
        if _existing is not None:
            return _existing(name)
        raise AttributeError(f"module {_module.__name__!r} has no attribute {name!r}")

    module.__dict__["__getattr__"] = __getattr__


class _AfterImportLoader(importlib.abc.Loader):
    """Run the real loader, then install the rename fallbacks."""

    def __init__(self, loader, repairs, module_name, absent=None):
        self._loader = loader
        self._repairs = repairs
        self._module_name = module_name
        self._absent = absent or {}

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _install_lazy_renames(module, self._repairs)
        _install_absent_members(module, self._absent)
        _RENAMES_INSTALLED.add(self._module_name)

    def __getattr__(self, item):
        # Different loaders expose extra hooks (get_filename, is_package...);
        # forward anything we do not override.
        return getattr(self._loader, item)


class _RenameRepairFinder(importlib.abc.MetaPathFinder):
    """Wrap the loader of the modules listed in _RENAMED_IN_6_0.

    Has to sit in front of PathFinder, because only the finder that resolves the
    module can wrap its loader; once a module is in sys.modules there is no
    second chance. The search itself is delegated to PathFinder with the same
    ``path`` the import machinery passed in, so the module picked is exactly the
    one a plain import would have picked -- no second-guessing of sys.path.
    """

    def find_spec(self, fullname, path=None, target=None):
        repairs = _RENAMED_IN_6_0.get(fullname)
        absent = _ABSENT_IN_6_0.get(fullname)
        if repairs is None and absent is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _AfterImportLoader(spec.loader, repairs or {}, fullname, absent)
        return spec


# ─────────────────────────────────────────────────────────────────────────────
# 2b. particleUtils.add_pbd_particle_material lost `drag` / `lift` in 6.0
# ─────────────────────────────────────────────────────────────────────────────
#
# Isaac Sim 5.1's helper took `drag` and `lift` and wrote the matching
# PhysxPBDMaterialAPI attributes. 6.0's signature stops at `cfl_coefficient`:
#
#   5.1: ..., friction, particle_friction_scale, drag, lift, damping, ...
#   6.0: ..., friction, particle_friction_scale, damping, viscosity, ...,
#         cfl_coefficient
#
# and the schema attributes went with the parameters. 6.0's `physxPBDMaterial:*`
# token list (omni.usd.schema.physx 110.1.13) has thirteen entries --
# adhesion, adhesionOffsetScale, cflCoefficient, cohesion, damping, density,
# friction, gravityScale, particleAdhesionScale, particleFrictionScale,
# surfaceTension, viscosity, vorticityConfinement -- and neither `drag` nor
# `lift` is among them. So there is nothing to rename onto and nothing to set
# after the fact: the knob no longer exists in the physics engine at all.
#
# env/scene_manager/objects/fluid.py:204 passes both unconditionally, so every
# fluid/particle task dies on its first reset with
#
#   TypeError: add_pbd_particle_material() got an unexpected keyword argument 'drag'
#
# The failure lands inside reset, which the evaluator catches per episode: the
# run then ends "cleanly" with rc=0 and zero episodes evaluated
# (`Success nums: 0, Fail nums: 0`), which is far more expensive to notice than
# a crash would have been.
#
# Why dropping the arguments is *exactly* equivalent here: 5.1's helper wrapped
# every schema write in `if value is not None`, so a None argument was already a
# no-op. All four affected layouts (pour_by_language, pour_liquid_into_cup,
# pour_liquid_into_cup_random, pour_balls_into_vase) carry `"drag": null` and
# `"lift": null`, so removing them cannot change the simulation. A *non*-None
# value is different -- 6.0 has no way to honour it -- and is reported loudly
# rather than dropped in silence.

_PARTICLE_UTILS_MODULE = "omni.physx.scripts.particleUtils"

#: Named explicitly instead of "whatever the signature rejects": a *renamed*
#: parameter must keep raising, or the benchmark silently loses physics.
_PBD_MATERIAL_DROPPED_IN_6_0 = ("drag", "lift")


def _patch_pbd_particle_material(module):
    """Make the particle-material helper tolerant of 6.0's slimmer signature.

    Idempotent, because it is reached twice: once from the loader hook that owns
    the import, and again from the top-up that runs before every project module
    (whichever happens first wins; the guard attribute makes the second a no-op).
    """
    original = getattr(module, "add_pbd_particle_material", None)
    if original is None or getattr(original, "_robodojo_pbd_compat", False):
        return original

    try:
        accepted = set(inspect.signature(original).parameters)
    except (TypeError, ValueError):  # pragma: no cover - C callables only
        accepted = None

    reported = set()

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        dropped = {}
        for name in _PBD_MATERIAL_DROPPED_IN_6_0:
            if name in kwargs and (accepted is None or name not in accepted):
                dropped[name] = kwargs.pop(name)
        for name, value in dropped.items():
            # Two reports per name at most: "the None case is harmless" and
            # "the non-None case is a real difference" must not be conflated,
            # and neither should be repeated for every env in the scene.
            key = (name, value is not None)
            if key in reported:
                continue
            reported.add(key)
            if value is None:
                sys.stderr.write(
                    f"[robodojo-compat] particleUtils.add_pbd_particle_material: "
                    f"{name}=None 省略（6.0 已移除该参数；5.1 对 None 同样不写属性，等价）\n"
                )
            else:
                sys.stderr.write(
                    f"[robodojo-compat] particleUtils.add_pbd_particle_material: "
                    f"{name}={value!r} 被丢弃 —— 6.0 的 physxPBDMaterial schema 不再提供该属性，"
                    f"本次运行的 {name} 力场缺失（行为差异，非等价复现）\n"
                )
        return original(*args, **kwargs)

    wrapper._robodojo_pbd_compat = True
    module.add_pbd_particle_material = wrapper
    sys.stderr.write(
        f"[robodojo-compat] {_PARTICLE_UTILS_MODULE}.add_pbd_particle_material: "
        f"已按 6.0 签名过滤 {', '.join(_PBD_MATERIAL_DROPPED_IN_6_0)}"
        f"（该版本接受 {len(accepted) if accepted is not None else '?'} 个参数）\n"
    )
    return wrapper


def _patch_loaded_particle_utils():
    """Top up the patch on a particleUtils that kit already imported.

    The finder below can only wrap an import it gets to resolve itself. The
    evaluation client reaches this module through ``from omni.physx.scripts
    import particleUtils`` inside env/scene_manager/objects/fluid.py, i.e.
    normal import machinery, so the finder normally wins -- but kit boots a
    long chain of extensions first and may have pulled the submodule in already.
    """
    module = sys.modules.get(_PARTICLE_UTILS_MODULE)
    if module is not None:
        _patch_pbd_particle_material(module)


class _PbdMaterialLoader(importlib.abc.Loader):
    """Run the real loader, then filter the helper's arguments."""

    def __init__(self, loader, module_name):
        self._loader = loader
        self._module_name = module_name

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_pbd_particle_material(module)

    def __getattr__(self, item):
        return getattr(self._loader, item)


class _PbdMaterialFinder(importlib.abc.MetaPathFinder):
    """Wrap the loader of omni.physx.scripts.particleUtils.

    Prepended for the same reason as _RenameRepairFinder: only the finder that
    resolves a module can wrap its loader.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PARTICLE_UTILS_MODULE or fullname in sys.modules:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PbdMaterialLoader(spec.loader, fullname)
        return spec


# ─────────────────────────────────────────────────────────────────────────────
# 3. top-level names owned by the RoboDojo tree
# ─────────────────────────────────────────────────────────────────────────────

# Resolved through the environment so the shim stays testable off-cluster: the
# image's robodojo-env.sh exports ROBODOJO_ROOT=/workspace/RoboDojo.
_ROOT = os.environ.get("ROBODOJO_ROOT", "/workspace/RoboDojo")

# name -> directories, in priority order. Kept claimable only while they exist,
# so the shim is inert when the tree is not present.
_PROJECT_TREES = {
    "utils": (os.path.join(_ROOT, "utils"), os.path.join(_ROOT, "XPolicyLab", "utils")),
    "env": (os.path.join(_ROOT, "env"),),
}

# Top-level names whose modules are part of the project. Anything imported under
# them is a place where an Isaac Sim rename has to be repaired first.
_REPAIR_CARRIERS = frozenset(_PROJECT_TREES) | {"task", "src"}

_RENAMES_INSTALLED = set()


def _apply_renames_to_loaded_modules():
    """Top up renamed names on Isaac Sim modules that are already imported.

    This exists because intercepting ``omni.physx``'s own import does not work
    in practice: kit loads the extension packages during boot through a path
    that never consults sys.meta_path, so by the time RoboDojo imports
    ``omni.physx`` it is already in sys.modules and no finder is asked about it.

    What the shim *can* rely on is the consumer. ``base_env.py`` -- the file that
    wants ``acquire_physx_interface`` -- is a project module, and every project
    module goes through _ProjectTreeFinder. Repairing the already-loaded module
    right before a project module executes puts the name in place before any
    project code can ask for it.
    """
    for module_name, repairs in _RENAMED_IN_6_0.items():
        if module_name in _RENAMES_INSTALLED:
            continue
        module = sys.modules.get(module_name)
        if module is None:
            continue
        _install_lazy_renames(module, repairs)
        _RENAMES_INSTALLED.add(module_name)

    for module_name, absent in _ABSENT_IN_6_0.items():
        if module_name in _RENAMES_INSTALLED:
            continue
        module = sys.modules.get(module_name)
        if module is None:
            continue
        _install_absent_members(module, absent)
        _RENAMES_INSTALLED.add(module_name)

    _patch_loaded_particle_utils()


class _ProjectLoader(importlib.abc.Loader):
    """Apply the pending API repairs, then run the real project module."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        _apply_renames_to_loaded_modules()
        self._loader.exec_module(module)

    def __getattr__(self, item):
        return getattr(self._loader, item)


class _ProjectTreeFinder(importlib.abc.MetaPathFinder):
    """Resolve selected top-level names to the RoboDojo tree, ignoring sys.path."""

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top not in _REPAIR_CARRIERS:
            return None

        if path is not None or fullname not in _PROJECT_TREES:
            # A submodule of a project package, or a plain project package such
            # as ``task``/``src`` that needs no claiming. Resolve it normally,
            # but wrap the loader: executing a project module is the last moment
            # at which the API repairs below can still be applied in time.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _ProjectLoader(spec.loader)
            return spec

        paths = [p for p in _PROJECT_TREES[fullname] if os.path.isdir(p)]
        if not paths:
            return None
        # loader=None + submodule_search_locations gives PEP 420 semantics:
        # the name resolves to a package assembled from these directories and
        # nothing else, whatever sys.path happens to look like at the time.
        spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
        spec.submodule_search_locations = paths
        return spec


# ─────────────────────────────────────────────────────────────────────────────

if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    # Appended: a real omni.physics.tensors.impl always wins if it reappears.
    sys.meta_path.append(_AliasFinder())

if not any(isinstance(finder, _ProjectTreeFinder) for finder in sys.meta_path):
    # Prepended: these names belong to the project tree, so no contribution from
    # sys.path -- the pip prebundle directories kit adds, cv2's own directory --
    # may win them.
    sys.meta_path.insert(0, _ProjectTreeFinder())

if not any(isinstance(finder, _RenameRepairFinder) for finder in sys.meta_path):
    # Prepended: it has to resolve the module itself in order to wrap its loader.
    sys.meta_path.insert(0, _RenameRepairFinder())

if not any(isinstance(finder, _PbdMaterialFinder) for finder in sys.meta_path):
    # Same reason, for the PBD particle-material helper's dropped parameters.
    sys.meta_path.insert(0, _PbdMaterialFinder())

# A meta-path finder only ever sees names that are not yet in sys.modules, and
# kit boots a long chain of extensions before the evaluation client runs. One of
# those extensions imports ``utils`` while OpenCV's directory is on sys.path, and
# from that point on the finder above is never consulted again: ``utils`` is
# already cached, pointing at ``<...>/pip_prebundle/cv2/utils``, and
# ``import utils.load_file`` dies with ModuleNotFoundError.
#
# So claim the project-tree names here instead. sitecustomize runs before kit,
# before cv2 and before any project code, so the claim cannot lose a race. Both
# packages are cheap to import: ``utils/__init__.py`` imports ``os`` and nothing
# else, and ``env`` has no ``__init__.py`` at all.
for _name in _PROJECT_TREES:
    if _name not in sys.modules:
        try:
            importlib.import_module(_name)
        except Exception:
            # Never fatal: the finder above still covers the plain import case.
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics -- off by default. Set ROBODOJO_TRACE_SIMCTX=1 to print the call
# stack of every SimulationContext construction, which answers "who allocated
# the singleton before RoboDojo's launch_sim() could". Nothing else depends on
# this block, so it is inert on every normal run.

_SIMCTX_MODULES = frozenset(
    {
        "isaacsim.core.api.simulation_context",
        "isaacsim.core.api.simulation_context.simulation_context",
    }
)


def _trace_log(message):
    """Diagnostics must never touch stdout.

    ``scripts/eval_policy.sh`` reads a sim-config name out of
    ``$(python3 -c ...)``, so anything this shim prints to stdout would be
    spliced into that value and turn a valid run into
    "Sim config file not found". stderr is outside every such protocol.
    """
    print(f"[TRACE-SIMCTX] {message}", file=sys.stderr, flush=True)


def _trace_patch_simctx(module):
    """Wrap the SimulationContext singleton hooks so constructions are visible."""
    cls = getattr(module, "SimulationContext", None)
    if cls is None or getattr(cls, "_robodojo_traced", False):
        return

    original_new = cls.__new__
    original_init = cls.__init__

    def traced_new(subcls, *args, **kwargs):
        import traceback

        _trace_log(f"__new__ 被调用：{subcls.__module__}.{subcls.__name__}")
        traceback.print_stack()
        return original_new(subcls, *args, **kwargs)

    def traced_init(self, *args, **kwargs):
        import traceback

        _trace_log(f"__init__ 被调用：{type(self).__module__}.{type(self).__name__}")
        traceback.print_stack()
        result = original_init(self, *args, **kwargs)
        _trace_log(f"构造完成，instance() is self = {cls.instance() is self}")
        return result

    cls.__new__ = traced_new
    cls.__init__ = traced_init
    cls._robodojo_traced = True
    _trace_log(f"探针已挂在 {module.__name__}")


class _TraceLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _trace_patch_simctx(module)

    def __getattr__(self, item):
        return getattr(self._loader, item)


class _SimCtxTraceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _SIMCTX_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _TraceLoader(spec.loader)
        return spec


if os.environ.get("ROBODOJO_TRACE_SIMCTX") == "1":
    sys.meta_path.insert(0, _SimCtxTraceFinder())
    _trace_log("已启用 SimulationContext 构造探针")
