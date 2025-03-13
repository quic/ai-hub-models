# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

NO_LICENSE = "This model's original implementation does not provide a LICENSE."


def _get_usage_and_limitations(include_gen_ai_terms: bool):
    return (
        """
## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation

"""
        if include_gen_ai_terms
        else ""
    )


def _get_references(
    research_paper_title: str | None,
    research_paper_url: str | None,
    source_repo: str | None,
) -> str:
    smi = ""
    if source_repo:
        smi = f"\n* [Source Model Implementation]({source_repo})"
    rp = ""
    if research_paper_title and research_paper_url:
        rp = f"\n* [{research_paper_title}]({research_paper_url})"
    elif (not research_paper_title) ^ (not research_paper_url):
        raise ValueError(
            "research_paper_title and research_paper_url should both be set or both be None."
        )

    if not smi and not rp:
        return ""

    return f"""
## References{rp}{smi}

"""


def _get_licenses(
    model_name: str, license_url: str | None, deploy_license_url: str | None
):
    if deploy_license_url is None and license_url is None:
        return ""
    license_url = license_url if license_url is not None else NO_LICENSE
    return f"""
## License
* The license for the original implementation of {model_name} can be found
  [here]({license_url}).
* The license for the compiled assets for on-device deployment can be found [here]({deploy_license_url})

"""


def _get_package_instructions(
    model_id: str, pip_install_flags: str | None, model_has_reqs: bool
):
    # Use dashes in model name to avoid an issue where older pip versions may not install modules correctly.
    install_pkg = "qai-hub-models" + (
        f'[{model_id.replace("_", "-")}]' if model_has_reqs else ""
    )
    if model_has_reqs:
        # Package extras include brackets in the package name, which confuses
        # shells like zsh unless contained within quotes.
        install_pkg = f'"{install_pkg}"'

    return f"""
Install the package via pip:
```bash
pip install {install_pkg}{f' {pip_install_flags}' if pip_install_flags else ''}
```
"""
