import pytest
import os
import sys
import re

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__))
    test_dir = os.path.join(root_dir, "tests")

    # Run pytest with coverage, capture output
    args = [
        "-q",               # quiet (only test results, no extra info)
        "--disable-warnings",
        "--maxfail=1",      # stop if too many errors (optional)
        f"--cov={root_dir}/src",   # measure coverage on src/
        "--cov-report=term",       # print coverage to stdout
        test_dir,
    ]

    # pytest returns exit code, but we also need its output
    from io import StringIO
    from contextlib import redirect_stdout

    buf = StringIO()
    with redirect_stdout(buf):
        exit_code = pytest.main(args)

    output = buf.getvalue()

    # Parse test summary line (e.g., "=== 3 passed, 1 failed in 0.12s ===")
    passed = total = 0
    for line in output.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            # Example: "collected 4 items"
            if line.strip().startswith("collected"):
                m = re.search(r"collected (\d+)", line)
                if m:
                    total = int(m.group(1))
            # Example: "3 passed, 1 failed in ..."
            if "passed" in line or "failed" in line:
                nums = re.findall(r"(\d+) (\w+)", line)
                for n, word in nums:
                    n = int(n)
                    if word == "passed":
                        passed = n
                    if total == 0:  # fallback if "collected" missing
                        total += n

    # Parse coverage percentage (last line like "TOTAL  123  4  96%")
    coverage = "N/A"
    for line in output.splitlines():
        if re.search(r"\d+%$", line.strip()):
            coverage = line.strip().split()[-1]

    # Print summary
    print(f"{passed}/{total} test cases passed. " +
          f"{coverage} line coverage achieved.")

    sys.exit(exit_code)
