# INTEGRATION_PLAN.md  
Comprehensive Plan to Merge the Enhanced “Skyscope Sentinel Intelligence AI” System  
into the existing GitHub repo  
`skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI`

---

## 1 ▪ Objectives
1. Merge all new modules (enhanced chat, autonomous business engine, launcher, docs) without breaking current master.
2. Consolidate overlapping scripts / folders, deleting true duplicates.
3. Standardise package layout → flatter, clearer import paths.
4. Preserve git history; use PR-based workflow.
5. Ship a repo that can be cloned and run with **one command**.

---

## 2 ▪ High-Level Phases

| Phase | Goal | Branch | Owner | Duration |
|------|------|--------|-------|----------|
| 0 | Repo backup & issue tracker creation | `backup/*` | Maintainer | 0.5 d |
| 1 | Inventory & diff mapping | `integration/inventory` | Integrator | 1 d |
| 2 | Code consolidation & refactor | `integration/refactor` | Dev team | 3 d |
| 3 | CI/CD, tests, lint fixes | `integration/qa` | QA lead | 1 d |
| 4 | Documentation merge | `integration/docs` | Tech writer | 0.5 d |
| 5 | Final review & squash-merge | PR to `main` | Reviewer | 0.5 d |

_Total calendar time: ≈ 1 week._

---

## 3 ▪ Detailed Steps

### 3.1 Inventory & Mapping
1. `git clone` current repo ➜ `origin`.
2. Create folder `~/audit` and run:
   ```
   find . -type f \( -name "*.py" -o -name "*.md" \) > ../audit/filelist_origin.txt
   ```
3. Place new system export in `~/new_system` → run same listing.
4. Build **Diff Matrix** (`audit/diff.xlsx`) with columns  
   `file`, `in_origin`, `in_new`, `action`.

### 3.2 Consolidation Rules
| Condition | Action |
|-----------|--------|
| File identical (hash) | Keep one copy, preserve older path if referenced. |
| Same name, divergent code | Rename legacy to `<name>_legacy.py`, mark for review. |
| Overlapping functionality (e.g., two `agent_manager.py`) | Keep _enhanced_ version, refactor calls, delete legacy. |
| Config duplication (`config.json`, `.env`) | Merge keys → produce single source of truth in `config/`. |
| Unused assets / notebooks older than 30 d | Archive to `archive/` branch, delete from main. |

### 3.3 Target Folder Structure
```
skyscope/
├── skyscope/                 # main package
│   ├── __init__.py
│   ├── core/                 # agents, orchestration
│   ├── ui/                   # Streamlit + assets
│   ├── business/
│   ├── crypto/
│   ├── infra/                # deploy, docker, k8s
│   └── utils/
├── tests/
├── scripts/                  # CLI helpers
├── assets/                   # fonts, css, js, images
├── config/
└── docs/
```
_All new code to be moved accordingly; add `skyscope/` namespace to imports._

### 3.4 Migration of Key Files

| Action | Source | Destination |
|--------|--------|-------------|
| ADD | `enhanced_chat_interface.py` | `skyscope/ui/chat.py` |
| ADD | `autonomous_business_operations.py` | `skyscope/business/operations.py` |
| ADD | `main_launcher.py` | `scripts/launch.py` |
| REPLACE | existing `agent_manager.py` | `skyscope/core/agent_manager.py` |
| MERGE | `crypto_manager.py` + legacy `crypto.py` | `skyscope/crypto/manager.py` |
| REMOVE | old `app.py` (thin wrapper) | — (replaced by chat.py) |
| ADD | `install.py` | root |
| ADD | `SYSTEM_OVERVIEW.md`, `README.md`, `INTEGRATION_PLAN.md` | `/docs` + root symlink |
| ARCHIVE | `notebooks/*`, `legacy_scripts/*` | new `archive/` branch |

### 3.5 Dependency Hygiene
1. Combine `requirements.txt` files → dedupe → pin versions (`~=`)
2. Add `pip-compile` workflow; generate `requirements.lock`.
3. Introduce `pyproject.toml` (setuptools) for installable package.

### 3.6 CI/CD Enhancements
- GitHub Actions:
  1. `lint`: black, isort, flake8, mypy.
  2. `test`: pytest, coverage ≥ 90 %.
  3. `build-docker`: push image on tag.
  4. `release-docs`: deploy `/docs` to GitHub Pages.

### 3.7 Testing & QA
- Move legacy tests to `tests/legacy/`; write new unit tests for:
  - AgentManager (spawning, queue)
  - Wallet prompt logic
  - Perplexica search wrapper
  - Business plan generator
- Add integration test: `pytest -q tests/e2e/test_launch.py` (launch Streamlit in headless mode and assert 200 OK).

### 3.8 Documentation Merge
- Root `README.md` stays concise (≤400 lines).
- Move deep dive docs to `docs/` (MkDocs ready).
- Auto-generate API docs via `pdoc`.

---

## 4 ▪ Redundancy Removal Checklist

- [ ] Identify duplicate model checkpoints (>50 MB)  
- [ ] Delete `.DS_Store`, `Thumbs.db`, cache folders  
- [ ] Consolidate multiple `.env*` examples → one template  
- [ ] Remove any hard-coded API keys in history (`git filter-repo`)  

---

## 5 ▪ Capability Enhancement Items

| Enhancement | Issue # | File(s) | Notes |
|-------------|---------|---------|-------|
| RAG memory persistence | #45 | `core/rag.py` | Use SQLite backend behind asyncio cache |
| Multi-wallet support | #46 | `crypto/manager.py` | Extend schema; UI toggle |
| Theme hot-swap | #47 | `ui/theme_manager.py` | Use session state + dynamic CSS inject |
| K8s auto-scaler | #48 | `infra/k8s/` | HPA manifests generated from config |

---

## 6 ▪ Branch & Commit Policy
1. **Conventional Commits** (`feat:`, `fix:`, `refactor:` …).  
2. One logical change per commit; no mixed formatting+logic.  
3. Always open PRs against `main`; require **one approval + green CI**.  
4. Use “Squash and merge” to keep history linear.  

---

## 7 ▪ Timeline Recap
1. **Day 1** – Inventory complete, diff matrix ready.  
2. **Day 2-4** – Consolidation & refactor, unit tests updated.  
3. **Day 5** – CI passes, docs merged.  
4. **Day 6** – Final PR review, squash-merge, tag `v1.1.0`.  

---

## 8 ▪ Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Merge conflicts with long-lived branches | Medium | Freeze feature branches during Phase 2 |
| Breaking imports after folder move | High | Run `pytest` + static `vulture` scan |
| Large binary files in git history | Low | Use `git lfs` or purge with `filter-repo` |
| Sensitive keys committed | High | Add `gitleaks` pre-commit hook |

---

## 9 ▪ Next Steps
1. Approve this integration plan in an Issue/PR.
2. Allocate roles & timeline owners.
3. Begin **Phase 0** (repo backup) immediately.
4. Track progress via GitHub project board “Integration v1.1”.

---

*Prepared for Skyscope Sentinel Technologies – July 2025.*  
