"""Measure local Streamlit dashboard startup and response time."""

import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_PATH = PROJECT_ROOT / "dashboard" / "app.py"

HOST = "127.0.0.1"
PORT = 8502
URL = f"http://{HOST}:{PORT}"

STARTUP_TIMEOUT_SECONDS = 60
WARM_REQUEST_COUNT = 10


def wait_for_dashboard(
    process: subprocess.Popen,
) -> float:
    """Wait until Streamlit responds and return startup time."""

    start_time = time.perf_counter()

    while True:
        if process.poll() is not None:
            raise RuntimeError(
                "Streamlit stopped before the dashboard became available."
            )

        elapsed = time.perf_counter() - start_time

        if elapsed > STARTUP_TIMEOUT_SECONDS:
            raise TimeoutError(
                "Dashboard did not start within "
                f"{STARTUP_TIMEOUT_SECONDS} seconds."
            )

        try:
            with urlopen(URL, timeout=1) as response:
                if response.status == 200:
                    return elapsed
        except (URLError, TimeoutError):
            time.sleep(0.1)


def measure_response_time() -> float:
    """Measure one HTTP response time in milliseconds."""

    start_time = time.perf_counter()

    with urlopen(URL, timeout=10) as response:
        response.read()

    elapsed = time.perf_counter() - start_time

    return elapsed * 1000


def main() -> None:
    """Run the local dashboard benchmark."""

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--server.headless=true",
        f"--server.address={HOST}",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
    ]

    print("Starting dashboard benchmark...")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print(f"URL: {URL}")

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        startup_seconds = wait_for_dashboard(process)

        warm_times = [
            measure_response_time()
            for _ in range(WARM_REQUEST_COUNT)
        ]

        average_ms = statistics.mean(warm_times)
        median_ms = statistics.median(warm_times)
        minimum_ms = min(warm_times)
        maximum_ms = max(warm_times)

        print("\nDashboard performance results")
        print("-" * 38)
        print(
            f"Cold startup time: {startup_seconds:.3f} seconds"
        )
        print(
            f"Warm requests measured: {WARM_REQUEST_COUNT}"
        )
        print(
            f"Average warm response: {average_ms:.2f} ms"
        )
        print(
            f"Median warm response: {median_ms:.2f} ms"
        )
        print(
            f"Minimum warm response: {minimum_ms:.2f} ms"
        )
        print(
            f"Maximum warm response: {maximum_ms:.2f} ms"
        )

        target_met = startup_seconds < 3

        print(
            "Under-three-second startup target: "
            f"{'PASS' if target_met else 'NOT MET'}"
        )

    finally:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()