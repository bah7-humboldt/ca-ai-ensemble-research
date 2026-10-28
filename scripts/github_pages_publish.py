"""Publishes scripts/nfdrs_ovf_web_export.py's output tree to a GitHub Pages site --
added 2026-09-15 as the interactive OVF web viewer's delivery path, chosen over
scripts/webdav_publish.py's gacc.nifc.gov target specifically because that target is a good
fit for a small, bounded PNG set (see psa_danger_map_prep.py), not a dataset that needs
arbitrary client-side querying against month-chunked JSON that grows every day. See
scripts/nfdrs_ovf_web_export.py's own module docstring for why GitHub Pages + month-chunking
was chosen over shipping the whole archive at once or building a real backend.

ONE-TIME MANUAL SETUP THIS MODULE ASSUMES HAS ALREADY BEEN DONE (not scriptable -- there is
no `gh` CLI on this machine and none is required):
  1. Create a NEW, separate, PUBLIC GitHub repo dedicated to this viewer (e.g.
     "oncc-nfdrs-ovf-viewer") via github.com's own web UI. Kept SEPARATE from this pipeline's
     own source repo on purpose -- it should contain nothing but the exported JSON + the
     self-contained viewer HTML (scripts/templates/nfdrs_ovf_web_viewer.html), never pipeline
     source/credentials, so there's no reason for it to be anything but public (GitHub Pages'
     free tier requires a public repo anyway; the data itself is FEMS-derived and already
     public via fems.fs2c.usda.gov).
  2. `git clone` that repo ONCE, by hand, in an interactive terminal, to whatever local path
     config/github_pages_config.json's clone_path will point at. This is also what PRIMES
     Windows Git Credential Manager (bundled with Git for Windows) -- it triggers GCM's
     browser-based GitHub OAuth login and caches the resulting credential in Windows
     Credential Manager. Every subsequent `git push` this module runs (including unattended,
     via Windows Task Scheduler) reuses that same cached credential -- NO personal access
     token is ever stored in any file this pipeline writes.
  3. Enable Pages on that repo: Settings -> Pages -> Source -> Deploy from a branch -> the
     branch config/github_pages_config.json names (default "main") -> "/ (root)". Confirm the
     resulting https://<username>.github.io/<repo>/ URL loads once, then publish_dir() below
     is what keeps it updated.

AUTH CAVEAT, VERIFY LIVE BEFORE TRUSTING THIS UNATTENDED: setup_nfdrs_ovf_task.ps1 registers
its Windows Scheduled Task with -User $env:USERNAME and no explicit password/logon type,
which defaults to an S4U logon. S4U logons do generally have access to the real user's
DPAPI-protected Credential Manager store, so GCM's cached credential SHOULD still work when
this runs unattended off that scheduled task -- but this has NOT yet been confirmed by
actually triggering the task (vs. only testing from an interactive terminal), and should be
before relying on it. If it turns out not to work unattended, the documented fallback is
embedding a PAT directly in the clone's own remote URL (`git remote set-url origin
https://<token>@github.com/<user>/<repo>.git`, done by hand, once) -- no worse a risk tier
than config/ftp_config.json's own already-plaintext password, but only fall back to it if the
GCM path is confirmed not to work.

CONFIG: config/github_pages_config.json (gitignored, template
config/github_pages_config.json.example) -- clone_path (the local clone from step 2 above),
branch (default "main"), optional commit_author_name/commit_author_email (falls back to
whatever this machine's own global git config already has if blank). A second site gets its own
clone-path key -- rx_window_clone_path (added 2026-09-29) for the Rx window forecast's separate
rxwindowexplorer repo, passed as publish_dir(clone_path_key=...) -- sharing branch and author. Same
"gitignored-JSON-soft-skip-if-missing/blank" convention as config/ftp_config.json/
config/email_config.json/config/teams_config.json.

COMMIT WORKFLOW (publish_dir(), below):
  1. `git pull --ff-only` first -- soft-skip (never force-overwrite) if that fails, both to
     never clobber anything added by hand on github.com between runs, and because a
     non-fast-forward pull failing is itself a loud signal something needs a human to look.
  2. Merge-copy local_dir's contents into the clone (shutil.copytree(..., dirs_exist_ok=True))
     -- NEVER wipes the clone directory first, so anything outside this pipeline's own known
     paths (a hand-added README, a custom domain's CNAME file) survives untouched.
  3. `git add -A`, then explicitly check `git diff --cached --quiet` BEFORE committing --
     `git commit` itself errors on a no-op commit, and once a run of settled months produces
     no changes at all (the common case most days), skipping straight to "nothing to publish"
     avoids both that error and an empty commit cluttering history.
  4. Only if something is actually staged: commit, then push. A push failure is logged and
     returned as False with NO retry/rebase/force-push -- the local commit stays in place, and
     the next run's own pull+push naturally picks it up along with that day's new changes. Any
     real divergence (e.g. someone edited files directly on github.com) should surface loudly
     for a human to resolve, not be silently auto-resolved by automation.

The local output_dir scripts/nfdrs_ovf_web_export.py writes into stays the actual source of
truth throughout -- this module is a thin, always-safe-to-retry mirror of it into git, not a
second place any real state lives. A failed push today just gets picked up again next run.
"""
import datetime
import json
import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "github_pages_config.json"
# Also appended to a file (added 2026-09-29): under the nightly scheduled task stdout goes
# nowhere, so three nights of silent push failures (2026-09-27..29) left no trace of the error.
LOG_PATH = ROOT / "data" / "outputs" / "github_pages_publish.log"


def log(msg):
    print(f"[github_pages_publish] {msg}", flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except OSError:
        pass


def _load_config(clone_path_key="clone_path"):
    """Returns {"clone_path": Path, "branch": str, "author_name": str|None,
    "author_email": str|None} or None (already logged) if config is missing/blank/the
    clone_path doesn't exist -- same soft-skip convention as webdav_publish._load_config().
    clone_path_key picks which site's clone (see CONFIG in the module docstring)."""
    if not CONFIG_PATH.exists():
        log(f"GitHub Pages publish not configured ({CONFIG_PATH} not found) -- skipping. "
            f"Copy config/github_pages_config.json.example to set it up (see this module's "
            f"own docstring for the one-time manual repo/clone/Pages setup it assumes).")
        return None
    try:
        config = json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        log(f"WARNING: could not parse {CONFIG_PATH}: {e}")
        return None

    clone_path_str = (config.get(clone_path_key) or "").strip()
    if not clone_path_str:
        log(f"GitHub Pages publish not configured ({clone_path_key} blank in "
            f"{CONFIG_PATH}) -- skipping")
        return None
    clone_path = pathlib.Path(clone_path_str)
    if not (clone_path / ".git").is_dir():
        log(f"WARNING: {clone_path_key} ({clone_path}) is not a git clone (no .git/ found) -- "
            f"skipping. See this module's docstring: clone the Pages repo there by hand first.")
        return None

    return {
        "clone_path": clone_path, "branch": (config.get("branch") or "main").strip(),
        "author_name": (config.get("commit_author_name") or "").strip() or None,
        "author_email": (config.get("commit_author_email") or "").strip() or None,
    }


def _run_git(args, cwd, env_overrides=None):
    """Runs a git subcommand in cwd, returns (success, combined_output). Never raises --
    every caller here checks the returncode/output itself."""
    env = {**os.environ, **(env_overrides or {})}
    try:
        result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True,
                                 timeout=120)
    except subprocess.TimeoutExpired:
        return False, (f"git {args[0]} timed out after 120s (likely a credential prompt with no "
                       f"one to answer it)")
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def publish_dir(local_dir, commit_message=None, clone_path_key="clone_path", replace_history=False,
                clean_subdirs=()):
    """Mirrors local_dir (scripts/nfdrs_ovf_web_export.py's own output_dir) into the
    configured GitHub Pages clone and pushes it. Returns True for "nothing to do" and
    "successfully published", False for any config/git failure -- never raises, same
    contract as webdav_publish.publish_files(). See module docstring for the full workflow
    and the manual setup it assumes. replace_history, clean_subdirs: see _publish_to_clone."""
    loaded = _load_config(clone_path_key)
    if loaded is None:
        return False
    return _publish_to_clone(local_dir, loaded["clone_path"], loaded["branch"], commit_message,
                             loaded["author_name"], loaded["author_email"], replace_history, clean_subdirs)


def _publish_to_clone(local_dir, clone_path, branch, commit_message=None, author_name=None,
                      author_email=None, replace_history=False, clean_subdirs=()):
    """The git side of publish_dir, given a clone. replace_history=False is the workflow in the
    module docstring. replace_history=True (added 2026-10-04 for the Rx window site, whose data
    file is rewritten in full every day) publishes each update as the branch's ONLY commit, so
    the repo stays one day's size instead of growing by every day's data (rxwindowexplorer was
    30 MB after 6 days at the coarsest grid; GitHub caps a Pages site at ~1 GB):
      1. fetch, and reset the clone to the remote branch -- so anything added by hand on
         github.com is there to carry forward, and a previous run's unpushed commit (local_dir
         regenerates it anyway) doesn't get in the way;
      2. merge-copy + .nojekyll + stage exactly as the normal workflow; skip if nothing changed;
      3. commit the staged tree with NO parent, and push it with --force-with-lease pinned to the
         remote commit fetched in step 1 -- so if anything reached the remote since (a hand edit
         mid-run), the push is refused rather than overwriting it;
      4. expire the reflog and gc the clone so the old objects leave the local disk too (GitHub
         drops its now-unreachable copies on its own schedule).
    clean_subdirs (added 2026-10-10 for the AI ensemble research site): sub-folders of the clone
    that local_dir regenerates in full; they're deleted after the pull/reset and before the
    merge-copy, so pages that dropped out of local_dir disappear from the site too (the merge-copy
    otherwise never deletes anything). Only names directly under the clone are accepted."""
    clone_path = pathlib.Path(clone_path)
    local_dir = pathlib.Path(local_dir)
    if not local_dir.exists():
        log(f"GitHub Pages publish: local_dir ({local_dir}) does not exist -- skipping")
        return False

    if replace_history:
        ok, output = _run_git(["fetch", "origin", branch], cwd=clone_path)
        if not ok:
            log(f"WARNING: git fetch failed, refusing to publish this run: {output}")
            return False
        ok, remote_sha = _run_git(["rev-parse", f"origin/{branch}"], cwd=clone_path)
        if not ok:
            log(f"WARNING: no origin/{branch} to publish over: {remote_sha}")
            return False
        ok, output = _run_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=clone_path)
        if ok:
            ok, output = _run_git(["reset", "--hard", f"origin/{branch}"], cwd=clone_path)
        if not ok:
            log(f"WARNING: could not reset the clone to origin/{branch}: {output}")
            return False
    else:
        ok, output = _run_git(["pull", "--ff-only", "origin", branch], cwd=clone_path)
        if not ok:
            log(f"WARNING: git pull --ff-only failed, refusing to publish this run: {output}")
            return False

    for sub in clean_subdirs:
        target = clone_path / sub
        if sub and "/" not in sub and "\\" not in sub and sub not in (".", "..", ".git") and target.is_dir():
            shutil.rmtree(target)

    shutil.copytree(local_dir, clone_path, dirs_exist_ok=True,
                     ignore=shutil.ignore_patterns(".git"))

    # GitHub Pages runs its default Jekyll build unless told not to -- which silently DROPS any
    # file/directory whose name starts with an underscore from the published output (Jekyll's
    # own "don't touch my internals" convention, e.g. _config.yml/_layouts/). Found live
    # 2026-09-19: scripts/fdra_charts_web_export.py's own data/_gacc.json would otherwise 404 on
    # the real site despite committing/pushing successfully. Touching this once, idempotently,
    # for every publish (not just that caller) protects any current or future underscore-led
    # path either exporter writes, and costs nothing on runs where it's already there.
    (clone_path / ".nojekyll").touch()

    ok, output = _run_git(["add", "-A"], cwd=clone_path)
    if not ok:
        log(f"WARNING: git add -A failed: {output}")
        return False

    ok, _ = _run_git(["diff", "--cached", "--quiet"], cwd=clone_path)
    if ok:  # exit 0 == nothing staged
        log("No changes to publish (all touched months already match the remote) -- skipping")
        return True

    commit_message = commit_message or "OVF web data update"
    env_overrides = {}
    if author_name:
        env_overrides["GIT_AUTHOR_NAME"] = env_overrides["GIT_COMMITTER_NAME"] = author_name
    if author_email:
        env_overrides["GIT_AUTHOR_EMAIL"] = env_overrides["GIT_COMMITTER_EMAIL"] = author_email

    if replace_history:
        ok, output = _run_git(["write-tree"], cwd=clone_path)
        if ok:
            tree = output
            ok, output = _run_git(["commit-tree", tree, "-m", commit_message], cwd=clone_path,
                                  env_overrides=env_overrides)
        if ok:
            new_sha = output
            ok, output = _run_git(["reset", "--soft", new_sha], cwd=clone_path)
        if not ok:
            log(f"WARNING: could not build the single replacement commit: {output}")
            return False
        ok, output = _run_git(["push", f"--force-with-lease={branch}:{remote_sha}", "origin", branch],
                              cwd=clone_path)
        if not ok:
            log(f"WARNING: git push (history replace) failed -- the remote changed since this run's "
                f"fetch, or the push itself failed; next run starts over from the remote: {output}")
            return False
        _run_git(["reflog", "expire", "--expire=now", "--all"], cwd=clone_path)
        _run_git(["gc", "--prune=now", "--quiet"], cwd=clone_path)
        log(f"Published to {clone_path} (branch {branch}, history replaced -- single commit): {commit_message}")
        return True

    ok, output = _run_git(["commit", "-m", commit_message], cwd=clone_path, env_overrides=env_overrides)
    if not ok:
        log(f"WARNING: git commit failed: {output}")
        return False

    ok, output = _run_git(["push", "origin", branch], cwd=clone_path)
    if not ok:
        log(f"WARNING: git push failed (commit is saved locally, will be included in the "
            f"next successful publish): {output}")
        return False

    log(f"Published to {clone_path} (branch {branch}): {commit_message}")
    return True
