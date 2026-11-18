# Environments

| File | For | Notes |
|---|---|---|
| `requirements-verify.txt` | level 1 (`scripts/verify_paper_numbers.py`) | numpy + pandas only |
| `../tracks/night/requirement.txt` | night tracking (HiFT/pysot) | tested on CUDA (NVIDIA A40) |
| `../tracks/cpu_int8/requirements.txt` | CPU int8 track | exact installed versions: `../tracks/cpu_int8/environment-tested.txt` (Python 3.11.17, torch 2.8.0+cu128) |
| `../tracks/domain_ft/pyproject.toml`, `../tracks/domain_ft/requirements/` | domain fine-tuning | `pip install -e tracks/domain_ft` |

The three tracks were run in separate environments; install the one you need.
