import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import csv
import io

def setup_openai():
    """Initialize OpenAI client with API key from environment variables."""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("\
        OPENAI_API_KEY not found in environment variables. \
        Set it up with: export OPENAI_APIKEY=....your_key....")
    return OpenAI(api_key=api_key)

def parse_csv_response(content: str) -> List[Dict]:
    """Parse the CSV response from OpenAI into a list of dictionaries."""
    try:
        # Create a CSV reader from the string content
        csv_file = io.StringIO(content)
        reader = csv.DictReader(csv_file)
        return list(reader)
    except Exception as e:
        print(f"Error parsing CSV response: {str(e)}")
        return []

def query_papers(protein_name: str) -> List[Dict]:
    """
    Query OpenAI to find neuroscience-related papers about a specific protein from PubMed.
    
    Parameters
    ----------
    protein_name : str
        Name of the protein to search for
        
    Returns
    -------
    List[Dict]
        List of dictionaries containing paper information
    """
    date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""Find all relevant neuroscience papers from PubMed about 
    the protein {protein_name} that can answer the following questions:
    1. Has {protein_name} been linked to dementia(s), in humans and/or animal models?
    2. Has {protein_name} been linked to other neuro pathologies, in humans and/or animal models?
    
    Organize the papers into a table, where each row is a paper and the columns are:
    - AuthorYEAR
    - Title
    - Journal
    - Human or Animal (which one)
    - If dementia, which dementia (AD, FTD, else)?
    - If other neuro pathologies, which ones?
    - Brief summary of findings related to {protein_name}

    Format the response as a CSV table with headers. Do not include any additional text or formatting.
    """

    try:
        client = setup_openai()
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better research capabilities
            messages=[
                {"role": "system", "content": "You are a research assistant specialized in neuroscience. Your task is to find and summarize relevant scientific papers from Pubmed. Respond only with the CSV data, no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused and consistent results
            max_tokens=2000
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        return parse_csv_response(content)
        
    except Exception as e:
        print(f"Error querying OpenAI: {str(e)}")
        return []

def ensure_queries_directory():
    """Ensure the queries_llm directory exists."""
    queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries_llm')
    os.makedirs(queries_dir, exist_ok=True)
    return queries_dir

def save_results_to_csv(papers: List[Dict], protein_name: str, queries_dir: str):
    """Save the results to a CSV file in the queries_llm directory."""
    if not papers:
        return None
        
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"{protein_name}_{date}.csv"
    filepath = os.path.join(queries_dir, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        if papers:
            writer = csv.DictWriter(csvfile, fieldnames=papers[0].keys())
            writer.writeheader()
            writer.writerows(papers)
    return filepath

def main():
    """Main function to run the protein paper search."""
    try:
        setup_openai()
        
        # Get protein name from user
        protein_name = input("Enter the name of the protein: ").strip()
        if not protein_name:
            raise ValueError("Protein name cannot be empty")
            
        print(f"\nSearching for papers about {protein_name} in neuroscience...\n")
        
        # Query papers
        papers = query_papers(protein_name)
        
        # Ensure queries directory exists and save results
        queries_dir = ensure_queries_directory()
        saved_file = save_results_to_csv(papers, protein_name, queries_dir)
        
        # Display results
        if papers:
            print(f"Found {len(papers)} relevant papers:\n")
            for i, paper in enumerate(papers, 1):
                print(f"Paper {i}:")
                for key, value in paper.items():
                    print(f"{key}: {value}")
                print("\n" + "-"*80 + "\n")
            if saved_file:
                print(f"\nResults have been saved to: {saved_file}")
        else:
            print("No papers found or an error occurred during the search.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()