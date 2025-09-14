import os
import sys
import subprocess
import argparse
from datetime import datetime
import shutil

def run_script(script_name, *args):
    """Runs a Python script using the same interpreter and waits for it to complete."""
    # Use sys.executable to ensure we're using the python from the current venv
    command = [sys.executable, script_name] + list(args)
    print(f"--- Running: {script_name} ---")
    
    try:
        # By removing `capture_output=True`, the subprocess can write directly to the terminal,
        # allowing progress bars like tqdm to be visible.
        subprocess.run(command, check=True)
        print(f"--- Successfully finished {script_name} ---\n")
        return True
    except subprocess.CalledProcessError as e:
        # The error output from the script will have already been printed to the console.
        print(f"\n--- Error running {script_name} ---")
        print(f"Return Code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Error: The script '{script_name}' was not found.")
        return False

def main():
    parser = argparse.ArgumentParser(description="End-to-end PDF statement processing pipeline.")
    parser.add_argument("pdf_file_path", type=str, help="The path to the PDF file to process.")
    args = parser.parse_args()

    pdf_path = args.pdf_file_path

    if not os.path.exists(pdf_path):
        print(f"Error: Input PDF file not found at '{pdf_path}'")
        return

    # --- Step 1: Convert PDF to images ---
    print(">>> Step 1: Converting PDF to enhanced images...")
    if not run_script("process_pdf.py", pdf_path):
        print(">>> PDF processing failed. Aborting.")
        return

    # --- Step 2: Process images with Llama to generate CSV ---
    print(">>> Step 2: Processing images with AI to extract transactions...")
    if not run_script("llama_process.py"):
        print(">>> AI processing failed. Aborting.")
        return

    # --- Step 3: Rename the output file ---
    print(">>> Step 3: Renaming the output file...")
    original_output_file = "transactions_from_local_dynamic.csv"
    
    if not os.path.exists(original_output_file):
        print(f"Error: Expected output file '{original_output_file}' not found. Cannot rename.")
        return

    # Generate the new filename
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    new_filename = f"{pdf_basename}-{timestamp}.csv"

    try:
        os.rename(original_output_file, new_filename)
        print(f"Successfully renamed output to: {new_filename}")
    except OSError as e:
        print(f"Error renaming file: {e}")
        return

    # --- Step 4: Clean up temp folder ---
    print(">>> Step 4: Cleaning up temporary files...")
    temp_subdir = os.path.join("temp", pdf_basename)
    if os.path.isdir(temp_subdir):
        try:
            shutil.rmtree(temp_subdir)
            print(f"Successfully removed temporary directory: {temp_subdir}")
        except OSError as e:
            print(f"Error cleaning up temp folder: {e}")
    else:
        print("Temp directory not found, skipping cleanup.")

    print("\n>>> Pipeline finished successfully! <<<")


if __name__ == "__main__":
    main()
