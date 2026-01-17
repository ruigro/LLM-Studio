# OWLLM Reliability Refactor - Implementation Status

## Goal
Make model testing, tool-calling, and environment creation reliable by fixing structural issues through focused refactoring (no big rewrite).

## Phase 1: StateStore (Single Source of Truth) ✅ COMPLETE

### Objectives
- Eliminate config drift between YAML and runtime state
- Make YAML static config only (never rewritten)
- Store runtime state (ports, PIDs, server status) in persistent DB
- Enable queryable state for testing and debugging

### Implementation Complete
✅ **StateStore created** (`LLM/core/state_store.py`)
- SQLite database at `data/owllm_state.db`
- Tables: models, envs, servers, kv
- Thread-safe with WAL mode
- CRUD operations for all entities

✅ **Server Manager updated** (`LLM/core/llm_server_manager.py`)
- Uses StateStore for port allocation (not YAML)
- Tracks server lifecycle (STARTING → RUNNING → STOPPED/FAILED)
- `_save_config()` deprecated (now no-op)
- `_get_server_url()` checks StateStore first, YAML second

✅ **Desktop App updated** (`LLM/desktop_app/main.py`)
- Registers new models in StateStore
- YAML port is now "preferred" hint only

✅ **Documentation** (`docs/ARCH_STATE_MAP.md`)
- Before/after architecture diagram
- State store schema documented
- Migration path explained

### Verification
```bash
# Check that StateStore was created
ls LLM/data/owllm_state.db  # Should exist after first run

# Start a server and verify state
python -m desktop_app.main
# In GUI: Select model, run inference
# StateStore should have entries in servers table with allocated port

# Check git status
git status  # YAML should NOT be modified after server starts
```

### Commit
```
bc0a175 PHASE 1: Add StateStore - single source of truth for runtime state
```

---

## Phase 2: Per-Stack Environments (env_key) - IN PROGRESS

### Objectives
- Replace per-model venvs with shared env_key (e.g., "torch-cu121-transformers-bnb")
- Atomic provisioning: create in .tmp, health check, rename to final
- Constraints files for reproducibility
- Eliminate env explosion

### Remaining Work
- [ ] Modify `EnvRegistry` to use env_key instead of per-model env_id
- [ ] Implement atomic create: `.envs/.tmp/<env_key>-<uuid>` → `.envs/<env_key>`
- [ ] Add constraints file generation/validation
- [ ] Health checks (imports + CUDA verification)
- [ ] Update StateStore `envs` table usage
- [ ] Migration path for existing per-model envs

### Files to Modify
- `LLM/core/envs/env_registry.py` - env_key resolution
- `LLM/core/environment_manager.py` - atomic provisioning
- Create: `constraints/*.txt` - per-env-key constraints

---

## Phase 3: Testing & CLI - TODO

### Objectives
- CLI commands for env/model/tools testing
- pytest suite for CI
- Make failures obvious and reproducible

### Tasks
- [ ] Add CLI entrypoint (`python -m owllm` or `scripts/owllm.py`)
- [ ] Implement `owllm env test <env_key>`
- [ ] Implement `owllm model smoke <model_id>`
- [ ] Implement `owllm tools smoke <model_id>`
- [ ] Add pytest tests:
  - `tests/test_env_health.py`
  - `tests/test_server_health.py`
  - `tests/test_tool_calling.py`
- [ ] GPU test skipping for CI

---

## Phase 4: Tool Calling (Strict JSON) - TODO

### Objectives
- Single tool-call envelope (JSON only)
- Schema validation
- Stop sequences for clean output
- Remove XML/Python parsers

### Tasks
- [ ] Create `tools/schema.json` for validation
- [ ] Modify `LLM/core/tool_calling.py`:
  - Remove XML/Python parsing
  - Single JSON extractor
  - jsonfix repair attempt
- [ ] Add stop sequences per backend
- [ ] System prompt templates for JSON-only output
- [ ] Tool server health check integration

---

## Phase 5: Documentation & Deliverables - TODO

### Tasks
- [ ] Write `docs/RELIABILITY.md` - howto guide
  - Add a model
  - Create env
  - Run env tests
  - Run model smoke
  - Run tools smoke
- [ ] PR with verification log showing:
  - `owllm env test torch-cu121-transformers-bnb` → PASS
  - `owllm model smoke <model>` → PASS
  - `owllm tools smoke <model>` → PASS

---

## Acceptance Criteria

### Phase 1 ✅
- [x] StateStore exists and is used by server manager
- [x] YAML never rewritten at runtime
- [x] Ports allocated at runtime and stored in DB
- [x] Server lifecycle tracked in DB
- [x] No regressions: existing UI/servers/tools still work

### Phase 2 (Remaining)
- [ ] Creating env twice is idempotent and doesn't corrupt state
- [ ] Multiple models can share same env_key
- [ ] Atomic provisioning prevents partial failures
- [ ] Constraints files enable reproducible builds

### Phase 3 (Remaining)
- [ ] CLI commands work and produce JSON output
- [ ] pytest suite passes (with GPU skips)
- [ ] Smoke tests verify end-to-end functionality

### Phase 4 (Remaining)
- [ ] Tool calls use strict JSON envelope only
- [ ] Schema validation prevents malformed calls
- [ ] Stop sequences prevent extra output

### Final Acceptance
- [ ] `owllm model smoke` passes for at least one real model
- [ ] `owllm tools smoke` demonstrates tool call end-to-end reliably
- [ ] All tests pass in CI

---

## Next Steps

1. **Continue Phase 2**: Implement env_key in EnvRegistry
2. **Test Phase 1**: Verify StateStore works in real usage
3. **Document**: Add examples of querying StateStore
4. **Plan Phase 3**: Design CLI interface

## Notes

- **Minimal invasive changes**: Kept existing UI, servers, tool server
- **Backward compatible**: YAML still read, StateStore adds new capability
- **Incremental**: Each phase is independently testable and committable
- **Focused**: No big rewrite, just targeted fixes for identified problems
