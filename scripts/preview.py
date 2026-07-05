import os
import sys
import subprocess

def main():
    """Start a local HTTP server to preview the generated Khmer text images."""
    out_dir = "generated/training_100k"
    port = 8888

    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid port number: {sys.argv[2]}", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(out_dir):
        print(f"Error: Directory '{out_dir}' does not exist.", file=sys.stderr)
        print("Please generate the dataset first using scripts/generate.py", file=sys.stderr)
        sys.exit(1)

    print(f"============================================================")
    print(f"Starting Khmer Text Preview Server")
    print(f"Target Directory : {out_dir}")
    print(f"Local URL        : http://localhost:{port}")
    print(f"Network URL      : http://0.0.0.0:{port}")
    print(f"============================================================")
    print("Press Ctrl+C to stop the server.")

    try:
        # Use python -m http.server with --directory flag (supported in python 3.7+)
        subprocess.run([sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0", "--directory", out_dir])
    except KeyboardInterrupt:
        print("\nPreview server stopped.")

if __name__ == "__main__":
    main()
