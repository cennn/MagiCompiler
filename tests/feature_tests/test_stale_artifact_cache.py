# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression: subgraph_indices points at a missing Inductor artifact directory.

SR integration could persist handles with restart_analysis_count>0 while the
artifact directory was never created; a later load must not call
CompiledArtifact.load on a non-directory path.
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
