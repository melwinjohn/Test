# AGENTS.md

## Project overview

This repository (`melwinjohn/Test`) is currently a **placeholder scaffold**. It contains only `README.md` with no application code, dependency manifests, or service definitions.

When application code is added, update this file with product-specific run/lint/test instructions.

## Cursor Cloud specific instructions

### Current state

- **No runnable application** — there is nothing to build, lint, or test yet.
- **No services** — no docker-compose, dev servers, or databases are defined.
- **No dependency manifests** — no `package.json`, `requirements.txt`, `pyproject.toml`, etc.

### VM toolchain (available when code is added)

The Cloud Agent VM includes:

| Tool | Notes |
|------|-------|
| Git | Repo cloned at `/workspace` |
| Node.js | v22.x via nvm (`/home/ubuntu/.nvm`) |
| npm / pnpm | Available alongside Node |
| Python 3.12 | `/usr/bin/python3` |
| pip | `/usr/bin/pip` |

### Startup update script

The VM update script conditionally installs dependencies **only when** matching manifest files exist (`package.json`, `pnpm-lock.yaml`, `requirements.txt`, `pyproject.toml`). On the current `main` branch, the script is a no-op.

### When application code is added

1. Add the appropriate manifest(s) and document run commands in `README.md`.
2. Update this section with:
   - Required services and how to start them
   - Lint, test, and dev-server commands
   - Non-obvious gotchas (ports, env vars, migrations, etc.)
3. Extend the update script if additional install steps are needed beyond the conditional defaults.
