import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import csv
import io
import requests
from Bio import Entrez
import time

def setup_openai():
    """Initialize OpenAI client with API key from environment variables."""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key)

def query_pubmed(protein_name: str) -> List[Dict]:
    """
    Query PubMed directly for recent papers about a specific protein.
    
    Parameters
    ----------
    protein_name : str
        Name of the protein to search for
        
    Returns
    -------
    List[Dict]
        List of dictionaries containing paper information
    """
    # Set your email for Entrez
    Entrez.email = "your.email@example.com"  # Replace with your email
    
    # Construct the search query
    query = f"{protein_name} AND (neuroscience OR brain OR dementia OR neuropathology) AND (2020:3000[Date - Publication])"
    
    try:
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=50, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            return []
            
        # Fetch details for each paper
        papers = []
        for pmid in record["IdList"]:
            # Add delay to respect NCBI's rate limits
            time.sleep(0.5)
            
            # Fetch paper details
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
            paper_record = handle.read()
            handle.close()
            
            # Parse the paper record
            paper_info = parse_pubmed_record(paper_record)
            if paper_info:
                papers.append(paper_info)
                
        return papers
        
    except Exception as e:
        print(f"Error querying PubMed: {str(e)}")
        return []

def parse_pubmed_record(record: str) -> Dict:
    """Parse a PubMed record into a dictionary."""
    try:
        lines = record.split('\n')
        paper_info = {
            "PMID": "",
            "Title": "",
            "Authors": "",
            "Journal": "",
            "Year": "",
            "Abstract": ""
        }
        
        current_field = ""
        for line in lines:
            if line.startswith("PMID- "):
                paper_info["PMID"] = line[6:].strip()
            elif line.startswith("TI  - "):
                paper_info["Title"] = line[6:].strip()
            elif line.startswith("AU  - "):
                if paper_info["Authors"]:
                    paper_info["Authors"] += "; "
                paper_info["Authors"] += line[6:].strip()
            elif line.startswith("JT  - "):
                paper_info["Journal"] = line[6:].strip()
            elif line.startswith("DP  - "):
                year = line[6:].strip().split()[0]
                if year.isdigit():
                    paper_info["Year"] = year
            elif line.startswith("AB  - "):
                paper_info["Abstract"] = line[6:].strip()
                
        return paper_info
    except Exception as e:
        print(f"Error parsing PubMed record: {str(e)}")
        return None

def summarize_papers_with_llm(papers: List[Dict], protein_name: str) -> List[Dict]:
    """
    Use OpenAI to summarize the papers and extract relevant information.
    
    Parameters
    ----------
    papers : List[Dict]
        List of paper information from PubMed
    protein_name : str
        Name of the protein being researched
        
    Returns
    -------
    List[Dict]
        List of summarized paper information
    """
    if not papers:
        return []
        
    # Prepare the papers for the LLM
    papers_text = "\n\n".join([
        f"PMID: {p['PMID']}\nTitle: {p['Title']}\nAuthors: {p['Authors']}\n"
        f"Journal: {p['Journal']}\nYear: {p['Year']}\nAbstract: {p['Abstract']}"
        for p in papers
    ])
    
    prompt = f"""You are a research assistant specialized in neuroscience. Analyze these papers about {protein_name} and create a summary table.

    Papers to analyze:
    {papers_text}

    Create a table with the following columns:
    - AuthorYEAR
    - Title
    - Journal
    - PMID
    - Human or Animal (which one)
    - Dementia or neuro pathologies (which one)
    - Brief summary of findings related to {protein_name}

    Format the response as a CSV table with headers. Do not include any additional text or formatting.
    Only include information that is explicitly stated in the papers.
    """

    try:
        client = setup_openai()
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",  # Using OpenAI's deep research model
            messages=[
                {"role": "system", "content": "You are a research assistant specialized in neuroscience. Your task is to analyze and summarize scientific papers. Only include information that is explicitly stated in the papers. Respond only with the CSV data, no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            top_p=0.1
        )
        
        # Parse the CSV response
        content = response.choices[0].message.content
        return parse_csv_response(content)
        
    except Exception as e:
        print(f"Error querying OpenAI: {str(e)}")
        return []

def parse_csv_response(content: str) -> List[Dict]:
    """Parse the CSV response from OpenAI into a list of dictionaries."""
    try:
        csv_file = io.StringIO(content)
        reader = csv.DictReader(csv_file)
        return list(reader)
    except Exception as e:
        print(f"Error parsing CSV response: {str(e)}")
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
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        if papers:
            writer = csv.DictWriter(csvfile, fieldnames=papers[0].keys())
            writer.writeheader()
            writer.writerows(papers)
    return filepath

def main():
    """Main function to run the protein paper search."""
    try:
        # Get protein name from user
        protein_name = input("Enter the name of the protein: ").strip()
        if not protein_name:
            raise ValueError("Protein name cannot be empty")
            
        print(f"\nSearching for recent papers about {protein_name} in PubMed...\n")
        
        # Query PubMed for papers
        papers = query_pubmed(protein_name)
        
        if not papers:
            print("No papers found in PubMed.")
            return
            
        print(f"Found {len(papers)} papers. Summarizing with AI...\n")
        
        # Summarize papers using LLM
        summarized_papers = summarize_papers_with_llm(papers, protein_name)
        
        # Ensure queries directory exists and save results
        queries_dir = ensure_queries_directory()
        saved_file = save_results_to_csv(summarized_papers, protein_name, queries_dir)
        
        # Display results
        if summarized_papers:
            print(f"Summarized {len(summarized_papers)} papers:\n")
            for i, paper in enumerate(summarized_papers, 1):
                print(f"Paper {i}:")
                for key, value in paper.items():
                    print(f"{key}: {value}")
                print("\n" + "-"*80 + "\n")
            if saved_file:
                print(f"\nResults have been saved to: {saved_file}")
        else:
            print("No papers were summarized or an error occurred during summarization.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 