#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""Parse test results (JUnit XML or unittest text) into a consolidated table row.

Output format (one line per invocation):
  STATUS|PLATFORM|LANG|TOTAL|PASSED|FAILED|ERRORS|SKIPPED|FAILED_DETAILS

The test-report job collects all rows and builds a single markdown table.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit(xml_path: str) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    suites = root.findall(".//testsuite") if root.tag != "testsuite" else [root]

    total = failures = errors = skipped = 0
    failed_tests = []

    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", suite.get("disabled", 0)))

        for case in suite.findall("testcase"):
            failure = case.find("failure")
            error = case.find("error")
            if failure is not None or error is not None:
                name = f"{case.get('classname', '')}.{case.get('name', '')}"
                msg = (failure if failure is not None else error).get("message", "")
                failed_tests.append((name.lstrip("."), msg[:120]))

    passed = total - failures - errors - skipped
    return {
        "total": total,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "failed_tests": failed_tests,
    }


def parse_unittest_text(text_path: str) -> dict:
    """Parse 'python -m unittest -v' output for the summary line."""
    text = Path(text_path).read_text()

    total = failures = errors = skipped = 0
    failed_tests = []

    m = re.search(r"Ran (\d+) tests? in", text)
    if m:
        total = int(m.group(1))

    m = re.search(r"FAILED \(([^)]+)\)", text)
    if m:
        counts = m.group(1)
        fm = re.search(r"failures=(\d+)", counts)
        if fm:
            failures = int(fm.group(1))
        em = re.search(r"errors=(\d+)", counts)
        if em:
            errors = int(em.group(1))
        sm = re.search(r"skipped=(\d+)", counts)
        if sm:
            skipped = int(sm.group(1))
    else:
        m = re.search(r"OK(?: \(([^)]+)\))?", text)
        if m and m.group(1):
            sm = re.search(r"skipped=(\d+)", m.group(1))
            if sm:
                skipped = int(sm.group(1))

    for line in text.splitlines():
        if "... FAIL" in line or "... ERROR" in line:
            name = line.split(" ... ")[0].strip()
            failed_tests.append((name, "FAIL" if "FAIL" in line else "ERROR"))

    passed = total - failures - errors - skipped
    return {
        "total": total,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "failed_tests": failed_tests,
    }


def is_xml(path: str) -> bool:
    try:
        with open(path) as f:
            return f.read(5).lstrip().startswith("<")
    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <platform> <language> <results_file>", file=sys.stderr)
        sys.exit(1)

    platform = sys.argv[1]
    language = sys.argv[2]
    results_file = sys.argv[3]

    if not Path(results_file).exists():
        print(f"no_results|{platform}|{language}|0|0|0|0|0|")
        sys.exit(0)

    if is_xml(results_file):
        r = parse_junit(results_file)
    else:
        r = parse_unittest_text(results_file)

    status = "pass" if r["failures"] + r["errors"] == 0 else "fail"
    if r["total"] == 0:
        status = "no_results"

    details = json.dumps(r["failed_tests"]) if r["failed_tests"] else ""
    print(f"{status}|{platform}|{language}|{r['total']}|{r['passed']}|{r['failures']}|{r['errors']}|{r['skipped']}|{details}")
