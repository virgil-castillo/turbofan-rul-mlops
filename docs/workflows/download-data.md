# Workflow: `turbofan-download-data`

What happens when downloading or verifying the NASA C-MAPSS dataset. One of two
mutually exclusive modes: `--check` (verify files) or `--kaggle` (download then
verify).

```mermaid
flowchart TD
    START(["$ turbofan-download-data --kaggle | --check"])

    subgraph SETUP["Setup"]
        direction TB
        ARGS["Parse args (mutually exclusive --kaggle / --check)"]
    end

    MODE{"Mode?"}

    subgraph CHECK["Verify"]
        direction TB
        VER["Check 12 expected files in data/raw<br/>(train/test/RUL x FD001-FD004)"]
    end

    subgraph DOWNLOAD["Download (Kaggle)"]
        direction TB
        KEY{"~/.kaggle/kaggle.json<br/>present?"}
        PULL["kaggle datasets download --unzip"]
        FLATTEN["Flatten CMaps/ subdir into data/raw"]
        MANUAL["Print manual download instructions"]
        KEY -->|yes| PULL --> FLATTEN
        KEY -->|no| MANUAL
    end

    START --> ARGS --> MODE
    MODE -->|--check| VER
    MODE -->|--kaggle| KEY
    FLATTEN --> VER
    VER --> DONE(["exit 0 if all present, else 1"])
    MANUAL --> FAIL(["exit 1"])
```

## Step reference

| Step | Function | Module |
|---|---|---|
| Verify files | `check` | `cli/download_data.py` |
| Download | `download_kaggle` | `cli/download_data.py` |

Note: this command is foundation-only — it touches just `utils` (logging) and
the filesystem; no config, models, or registry are involved.
