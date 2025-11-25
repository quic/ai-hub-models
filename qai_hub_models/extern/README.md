This package contains:

1.  Copies of external packages that are modified to remove incompatible dependencies.

2.  Namespace duplicates of optional package dependencies.

    These modules can be imported without error if the underlying package is not installed.
    Imported symbols may resolve to None in that case. It is the implementer's responsibility to check the symbols are available when used.

    This is useful when:
        - shared utilties use dependencies that are required by only a subset of models.
        - shared utilities must be imported at the module level instead of inside the function / class in which they are used

    If packages are defined here, imports of the original package (eg. 'import numba') should be banned by the linter (via pyproject.toml settings), with a message to import from extern instead.
