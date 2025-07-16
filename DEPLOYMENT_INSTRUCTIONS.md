# DEPLOYMENT_INSTRUCTIONS.md  
Deploying the **Enhanced Skyscope Sentinel Intelligence AI** system to  
`skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI`

---

## 1 ▪ Overview  
This guide walks you through **cloning, integrating, committing and publishing** the new enhanced code-base (10 000-agent swarm, glass UI, Perplexica search, autonomous business engine) into the existing public GitHub repository.

The process uses:

• a **temporary working clone** (`temp_repo/`)  
• the **enhanced system export** (`enhanced_system/`) – the folder containing all new files you generated locally  
• an **automated deploy script** `deploy_to_github.py` – placed at project root during export

> ⚠️ The script never touches your original repo directly; it works on a clone and creates an integration branch (`integration/enhanced-system`) plus an optional pull-request.

---

## 2 ▪ Prerequisites  

1. **Git 2.30+** and **Python 3.9+**  
2. **GitHub CLI** (`gh`) – optional but enables automatic PR creation  
3. A **GitHub Personal Access Token (PAT)** with `repo` scope  
   - Save the token to a file called `.github_token` in project root or provide path via `--token`.
4. Enhanced export folder `enhanced_system/` containing files produced by the AI.

---

## 3 ▪ One-Command Quick Deploy  

```bash
# From the directory that contains deploy_to_github.py and enhanced_system/
python deploy_to_github.py
```

You will be asked to confirm before changes are pushed.  
The script will:

1. Clone the remote repo into `temp_repo/`  
2. Create backup in `backup_repo/`  
3. Generate the new clean package layout (`skyscope/…`)  
4. Copy/rename enhanced files (see Integration Plan)  
5. Delete redundant legacy files  
6. Update all import statements  
7. Commit with a single **Conventional Commit** message  
8. Push to `integration/enhanced-system`  
9. Open a pull-request (if `gh` is installed)

---

## 4 ▪ Step-by-Step (Manual or Audit)

### 4.1 Dry-Run Audit  

```bash
python deploy_to_github.py --dry-run
```

Nothing is written to disk except log `deploy_to_github.log`.  
Inspect:

* new folder structure preview  
* list of files to remove / copy  
* commit message

If everything looks good remove `--dry-run`.

### 4.2 No-Push Test  

Generate commits locally but keep them un-pushed:

```bash
python deploy_to_github.py --no-push
# Review with:
cd temp_repo
git log --stat
pytest            # run tests
```

Push manually when satisfied:

```bash
git push origin integration/enhanced-system
```

---

## 5 ▪ Manual Token Configuration  

If you prefer not to store the PAT in a file:

```bash
export GITHUB_TOKEN=ghp_XXXXXX
python deploy_to_github.py --token /dev/null
```

The script will read `GITHUB_TOKEN` env-var when file is blank.

---

## 6 ▪ Reviewing the Pull-Request  

1. Open the PR URL printed by the script or run  
   `gh pr view --web integration/enhanced-system`  
2. **Checklist** before merging:  
   - [ ] CI lints & tests pass  
   - [ ] Folder structure matches `INTEGRATION_PLAN.md`  
   - [ ] No secrets in diff (`gitleaks` action should be green)  
   - [ ] Docs render correctly on GitHub  
3. Use **“Squash & merge”** to keep history linear.  
4. Tag release once merged:

```bash
git checkout main
git pull
git tag v1.1.0 -m "Enhanced Skyscope Sentinel AI"
git push --tags
```

---

## 7 ▪ Post-Merge Clean-Up  

```bash
rm -rf temp_repo backup_repo
```

Optional: delete integration branch after PR merge:

```bash
git push origin --delete integration/enhanced-system
```

---

## 8 ▪ Troubleshooting  

| Issue | Resolution |
|-------|------------|
| `fatal: could not read Username` | PAT missing or invalid. Re-create `.github_token`. |
| `gh: command not found` | Install GitHub CLI or run script with `--no-push` and create PR manually. |
| Import errors after deploy | Run `pytest`; check `deploy_to_github.py` log for missed mappings, then fix manually and recommit. |
| CI failing on black/isort | Run `black . && isort .`, commit and push. |

---

## 9 ▪ Useful Script Flags  

| Flag | Description |
|------|-------------|
| `--dry-run` | Perform all steps but **no writes**, good for audit |
| `--no-push` | Commit locally but don’t push or open PR |
| `--branch BR` | Set custom branch name |
| `--token FILE` | Path to PAT file (default `.github_token`) |

---

## 10 ▪ Estimated Timeline  

Entire automated run (clone → PR) ≈ **2–4 minutes**  
Manual review & merge ─ depends on CI (≈5 min)

---

## 11 ▪ Logs  

All actions are logged to `deploy_to_github.log`.  
Check the file if something goes wrong.

---

Happy deploying!  
_– Skyscope Sentinel DevOps Team_  
