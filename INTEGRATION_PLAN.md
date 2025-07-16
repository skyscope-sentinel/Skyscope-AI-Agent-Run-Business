# INTEGRATION_PLAN.md  
**Skyscope Sentinel Intelligence AI – Repository Integration Blueprint**  

_Last updated: 2025-07-16_

---

## 1. Objectives
1. Merge the new **Enhanced Skyscope Sentinel Intelligence AI system** (v1.0) with the pre-existing GitHub repository without data loss.  
2. Standardise file-tree layout, eliminate legacy duplication and dead code.  
3. Establish a repeatable workflow (branches → PR → CI) so future iterations ship safely.  

---

## 2. High-Level Workflow
| Phase | Owner | Description | Done? |
|-------|-------|-------------|-------|
| 0. Backup  | Repo admin | Tag current `main` as `pre-enhancement-backup` | ☐ |
| 1. Audit   | Integrator | Inventory current repo vs. new package | ☐ |
| 2. Staging | Integrator | Create `integration/v1.0` branch, copy in new tree | ☐ |
| 3. De-dupe | Integrator | Remove/merge duplicate modules, pick canonical version | ☐ |
| 4. Refactor Map | Lead dev | Align imports & paths to new structure | ☐ |
| 5. Automated Tests | QA | Ensure all tests pass locally | ☐ |
| 6. CI Pipeline Update | DevOps | Update GitHub Actions & Docker build | ☐ |
| 7. Code-Review PR | Team | Review, approve, squash-merge to `main` | ☐ |
| 8. Tag & Release | Maintainer | Tag `v1.0.0`, create release notes | ☐ |

---

## 3. Target Repository Structure
```
/
├─ .github/
│  └─ workflows/          # CI definitions
├─ docs/                  # MkDocs / Sphinx sources
├─ scripts/               # helper & maintenance scripts
├─ skyscope/              # **new** Python package root
│  ├─ __init__.py
│  ├─ core/               # agent_manager, crypto_manager, …
│  ├─ ui/                 # Streamlit apps, themes, assets
│  ├─ ops/                # autonomous_business_operations, business_manager
│  ├─ data/               # embedded small reference data
│  └─ tests/              # unit tests colocated for package
├─ assets/                # fonts, css, images, animations
├─ config/                # json/yaml configs, .env template
├─ data/                  # runtime-generated DB & stores (git-ignored)
├─ requirements.txt
├─ install.py
├─ start.sh / start.bat
└─ README.md
```
Key principles  
• All importable code lives inside `skyscope/` to avoid path hacks.  
• No runtime artefacts (logs, .db, wallets) stored in Git – ensure `.gitignore` is strict.  

---

## 4. Redundancy & Conflict Resolution Guide
1. **File duplicates** – compare checksum & git history. Keep the newest fully-featured module, delete older alias files (e.g. `app-2.py`, `agent_manager-2.py`).  
2. **Naming collisions** – prefer snake_case, singular module names. Update internal imports via IDE multi-rename or `sed`.  
3. **Legacy build specs** (`*.spec`, old shell scripts) – archive under `legacy/` then remove once CI is green.  
4. **Obsolete requirements** – consolidate `requirements*.txt` into single root `requirements.txt`; annotate extras via markers.  
5. **Multiple README variants** – merge content into one authoritative `README.md`, move detailed docs to `/docs`.  

---

## 5. Integration Steps in Detail

### 5.1 Local Audit
```bash
# inside repo root
git checkout -b integration/v1.0
python scripts/repo_audit.py  # optional helper to list duplicates
```

### 5.2 Bring in New System
1. Run `generate_skyscope_system.sh --output temp_build/` if not already present.  
2. Copy / move generated directories & files into their target paths (see section 3).  
3. Commit with message “chore: scaffold enhanced system (unwired)”.

### 5.3 De-duplication Pass
Use the following cheat-sheet:  

| Legacy file | New canonical file | Action |
|-------------|-------------------|--------|
| `main_app.py`, `app.py` | `skyscope/ui/app.py` | Delete old, re-export entry in `start.sh` |
| `business_generator.py`, `business_manager.py` | `skyscope/ops/business_manager.py` | Merge functions then delete duplicates |
| `crypto_manager.py` (two copies) | `skyscope/core/crypto_manager.py` | Keep one |

Continue until `git ls-files | sort | uniq -d` returns none.

### 5.4 Align Imports
Run `ruff` or `isort` autofix, then `pytest -q`. Fix any broken relative imports.

### 5.5 Update CI/CD
1. Modify `.github/workflows/ci.yml` to  
   • setup Python 3.11  
   • cache `~/.cache/pip`  
   • run `pytest`, `black --check`, `ruff`.  
2. Add Docker build stage that copies only `skyscope/`, `install.py`, `requirements.txt` for lean image.

### 5.6 Documentation
• Auto-generate API docs via `pdoc skyscope -o docs/api`.  
• Update `/docs/index.md` with new architecture diagram.  

### 5.7 Tag & Release
```bash
git checkout main
git merge --no-ff integration/v1.0
git tag -a v1.0.0 -m "Skyscope Sentinel Intelligence AI – first unified release"
git push origin main --tags
```

---

## 6. Versioning & Branch Strategy
* **main** – stable, deployable  
* **dev** – day-to-day feature integration  
* **integration/\*** – ad-hoc large merges (like this plan)  
SemVer rules: bump **MAJOR** for breaking API or data migrations, **MINOR** for feature, **PATCH** for bugfix.

---

## 7. Post-Merge Validation Checklist
- [ ] `pip install -e . && pytest -q` passes locally and in CI.  
- [ ] `./start.sh --no-autonomous` starts UI on localhost.  
- [ ] Wallet prompt appears and persists config.  
- [ ] Sample autonomous business cycle runs for ≥1 iteration without error.  
- [ ] Docker image builds (`docker build -t skyscope-ai .`).  
- [ ] Release notes published under GitHub Releases.

---

## 8. Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Hidden circular imports | enforce package structure & run `python -m pip install flake8-import-order` |
| Large binary assets bloat | track via Git LFS or exclude; keep models outside repo |
| Windows path length issues | enable `core.longpaths=true` and avoid deeply nested dirs |
| Environment drift | lock dependencies with `requirements.lock`, pin in CI |

---

## 9. Contacts
• **Integration Lead:** your.name@skyscope.ai  
• **DevOps:** devops@skyscope.ai  
• **QA:** qa@skyscope.ai  

---

_This document is living: update checkboxes and notes as integration progresses._  
