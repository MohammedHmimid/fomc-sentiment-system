"""Launch all 5 services as local subprocesses (Colab-friendly, no Docker)."""
import subprocess
import sys
import time

SERVICES = [
    ("fusion-service.main", 8004),
    ("nlp-service.main", 8001),
    ("audio-service.main", 8002),
    ("vision-service.main", 8003),
    ("gateway-service.main", 8000),
]


def main():
    procs = []
    for module, port in SERVICES:
        # module dirs use hyphens; uvicorn needs the file via --app-dir + import path.
        service_dir = module.split(".")[0]
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--app-dir", service_dir, "--host", "0.0.0.0", "--port", str(port)]))
        print(f"started {service_dir} on :{port}")
        time.sleep(2)
    print("all services up; Ctrl-C to stop")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
