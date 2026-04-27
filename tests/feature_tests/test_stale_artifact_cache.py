# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression for stale / inconsistent piecewise Inductor artifact cache.

Two failure modes:

1. ``subgraph_indices.py`` references a path that is not a directory (manual
   cleanup, partial cache, or never-written artifact).  ``CompilerManager.load``
   must not call ``CompiledArtifact.load``.

2. Rehydrated handles from disk are not overwritten when ``piecewise_compiler``
   returns ``cache_handle is None`` (save failed).  We must drop the stale
   entry so ``save_to_file`` does not re-serialize ghost paths — this is not a
   ``graph_index`` mismatch; index and ``artifact_shape_*_subgraph_{i}`` keys
   are always built from the same ``graph_index`` in ``compile()``.
"""

from __future__ import annotations

import pprint
from pathlib import Path

import torch
import torch.fx as fx

from magi_compiler.config import CompileConfig
from magi_compiler.magi_backend._cache_data_cls import CacheEntry
from magi_compiler.magi_backend.magi_backend import CompilerManager


def test_compiler_manager_drops_missing_artifact_dir_restart_zero(tmp_path: Path) -> None:
    cache_dir = tmp_path / "model_x"
    cache_dir.mkdir(parents=True)
    bogus = cache_dir / "artifact_shape_None_subgraph_0"
    assert not bogus.exists()

    data = {(None, 0, "inductor_standalone"): ("artifact_shape_None_subgraph_0", str(bogus), 0)}
    (cache_dir / "subgraph_indices.py").write_text(pprint.pformat(data))

    cfg = CompileConfig(cache_root_dir=str(tmp_path))
    cm = CompilerManager(cfg)
    cm.initialize_cache(cache_dir)

    entry = CacheEntry(None, 0, "inductor_standalone")
    assert entry in cm.cache

    gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
    assert cm.load(gm, [], entry) is None
    assert entry not in cm.cache


def test_compiler_manager_drops_missing_artifact_dir_restart_nonzero(tmp_path: Path) -> None:
    """Same as production SR cache: restart_analysis_count=1 but no on-disk artifact."""
    cache_dir = tmp_path / "model_2"
    cache_dir.mkdir(parents=True)
    bogus = cache_dir / "artifact_shape_None_subgraph_0"
    assert not bogus.exists()

    data = {(None, 0, "inductor_standalone"): ("artifact_shape_None_subgraph_0", str(bogus), 1)}
    (cache_dir / "subgraph_indices.py").write_text(pprint.pformat(data))

    cfg = CompileConfig(cache_root_dir=str(tmp_path))
    cm = CompilerManager(cfg)
    cm.initialize_cache(cache_dir)

    entry = CacheEntry(None, 0, "inductor_standalone")
    assert entry in cm.cache

    gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
    assert cm.load(gm, [], entry) is None
    assert entry not in cm.cache
    assert 0 not in cm._remaining_restart_skips


def test_maybe_store_cache_entry_removes_stale_when_handle_is_none(tmp_path: Path) -> None:
    """Simulate: disk had a handle; compile finished but artifact was not persisted."""
    cache_dir = tmp_path / "model_y"
    cache_dir.mkdir(parents=True)
    bogus = cache_dir / "artifact_shape_None_subgraph_0"
    data = {(None, 0, "inductor_standalone"): ("artifact_shape_None_subgraph_0", str(bogus), 1)}
    (cache_dir / "subgraph_indices.py").write_text(pprint.pformat(data))

    cfg = CompileConfig(cache_root_dir=str(tmp_path))
    cm = CompilerManager(cfg)
    cm.initialize_cache(cache_dir)

    entry = CacheEntry(None, 0, "inductor_standalone")
    assert entry in cm.cache

    cm._maybe_store_cache_entry(entry, None, None, "artifact_shape_None_subgraph_0")

    assert entry not in cm.cache
    assert 0 not in cm._remaining_restart_skips
