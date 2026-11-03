# Security policy

## Scope

This repository holds study implementations of pricing, risk and trading
models. It runs on synthetic data, has no network services and handles no
credentials. The code has not been used on a live book.

Security issues still matter here. Examples: a dependency with a known
vulnerability, a CI workflow that could be abused to run untrusted code with
write access, or a script that writes outside its output directory.

## Supported versions

Only the latest release and `main` receive fixes.

## Reporting a vulnerability

Report it privately through GitHub: open the repository's **Security** tab
and choose **Report a vulnerability**. Do not open a public issue.

Include the affected file and commit, the steps to reproduce, and the impact
you expect. You should get a first reply within 7 days. A fix or a decision
not to fix follows within 30 days, and the reporter is credited in the
CHANGELOG unless they ask not to be.

## Dependencies

Dependabot checks pip, cargo and GitHub Actions dependencies weekly. CI
installs every third-party Python package from `uv.lock` with its sha256
hash checked (`pip install --require-hashes`, wheels only, through
`.github/scripts/hashed_install.sh`), so a changed or substituted wheel fails
the job. Every third-party action is pinned to a full commit SHA.
