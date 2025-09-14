import os
import json
import requests
from dotenv import load_dotenv
import google.auth
import google.auth.transport.requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Prompts ---

HEADER_PROMPT = """You are an expert financial document analyst performing a pure OCR and data extraction task. Your task is to find the main transaction header row in the provided image. Be silent and do not add any commentary.

Instructions:
1.  Analyze the image to find the table containing transaction details.
2.  If you find the header row (e.g., "Data", "Histórico", "Valor", "Docto", "Crédito", "Débito", "Saldo"), return it as a single line with columns separated by a semicolon (;).
3.  If this page does NOT contain a transaction header, you MUST return the exact string "NO_HEADER_FOUND".
4.  Do not return any other text, explanations, or comments.

Example of a found header:
Data;Histórico;Valor;Docto;Crédito;Débito;Saldo
"""

DATA_PROMPT_TEMPLATE = """You are an expert financial document analyst performing a pure OCR and data extraction task. Your task is to extract all transaction rows from the provided bank statement image that match a specific header. Be silent and do not add any commentary.

The header for this document is: "{header}"

Instructions:
1.  Analyze the image to find all rows that belong to the transaction table with the header provided above.
2.  For each transaction row you find, format it as a single line with values separated by a semicolon (;), matching the order of the header.
3.  Return one line for each transaction found.
4.  If the page does not contain any transaction rows, you must return an empty response.
5.  Do NOT include the header row in your output. Only return the data rows. Do not add any other text, explanations, or comments.
"""

# --- Utility Functions ---

def get_access_token():
    """Gets the access token from the gcloud credentials."""
    credentials, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token

def image_to_base64(image_path):
    """Converts a local image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- API Call and Worker Functions ---

def make_api_call(project_id, location, access_token, image_path, prompt_text):
    """Generic function to make the API call."""
    url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    base64_image = image_to_base64(image_path)
    
    data = {
        "model": "meta/llama-3.2-90b-vision-instruct-maas",
        "stream": False,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}, {"type": "text", "text": prompt_text}]}],
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 32,
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=300)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("choices"):
            return response_data["choices"][0]["message"]["content"].strip()
        return "Error: No content in response."
    except requests.exceptions.RequestException as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return f"Error: {e}"

def find_header_sequentially(project_id, location, image_paths, access_token):
    """Sequentially processes images to find the first valid header."""
    for i, image_path in enumerate(image_paths):
        print(f"Searching for header in: {os.path.basename(image_path)}...")
        header = make_api_call(project_id, location, access_token, image_path, HEADER_PROMPT)
        if header and header != "NO_HEADER_FOUND":
            print(f"Header found on page {i+1}: {header}")
            return header, i  # Return header and the index of the page it was found on
    return None, -1

def data_extraction_worker(project_id, location, image_path, access_token, data_prompt):
    """Worker for parallel data extraction."""
    content = make_api_call(project_id, location, access_token, image_path, data_prompt)
    return image_path, content

# --- Main Execution ---

if __name__ == "__main__":
    load_dotenv()
    
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    LOCATION = os.getenv("GCP_LOCATION")

    if not PROJECT_ID or not LOCATION:
        print("Error: Please make sure GCP_PROJECT_ID and GCP_LOCATION are set in your .env file.")
        exit(1)

    # --- Find Image Directory ---
    try:
        temp_dir = "temp"
        subdirs = [os.path.join(temp_dir, d) for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if not subdirs:
            raise FileNotFoundError("No subdirectories found in the 'temp' folder.")
        latest_subdir = max(subdirs, key=os.path.getmtime)
        IMAGE_DIR = latest_subdir
        print(f"Processing images from: {IMAGE_DIR}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    image_files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg'))],
        key=lambda x: int(os.path.basename(x).split('_')[1])
    )

    print("Getting access token...")
    access_token = get_access_token()

    # --- Phase 1: Find Header ---
    found_header, header_page_index = find_header_sequentially(PROJECT_ID, LOCATION, image_files, access_token)

    if not found_header:
        print("Could not find a transaction header in any of the pages. Exiting.")
        exit(1)

    # --- Phase 2: Extract Data in Parallel ---
    data_prompt = DATA_PROMPT_TEMPLATE.format(header=found_header)
    pages_to_process = image_files[header_page_index:]
    results = {}

    # Wrap the as_completed iterator with tqdm for a progress bar
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_image = {executor.submit(data_extraction_worker, PROJECT_ID, LOCATION, img_path, access_token, data_prompt): img_path for img_path in pages_to_process}
        
        for future in tqdm(as_completed(future_to_image), total=len(pages_to_process), desc="Extracting Transactions"):
            try:
                image_path, content = future.result()
                results[image_path] = content
            except Exception as exc:
                image_path = future_to_image[future]
                print(f'{os.path.basename(image_path)} generated an exception: {exc}')
                results[image_path] = f"Error: {exc}"

    # --- Combine Results ---
    # Start with the header, then add the data from each page in the correct order
    data_rows = [results[path] for path in pages_to_process if results.get(path)]
    final_result = found_header + "\n" + "\n".join(filter(None, data_rows)) # Filter out empty results

    # No longer need to print the full result here as the progress is visible
    # print("\n--- All Images Processed. Concatenated Results: ---")
    # print(final_result)
    
    output_filename = "transactions_from_local_dynamic.csv"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_result)
    print(f"\nResult saved to {output_filename}")

