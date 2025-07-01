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
import tiktoken

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
            "Abstract": "",
            "DOI": ""
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
            elif line.startswith("LID - "):
                doi_text = line[6:].strip()
                if "10." in doi_text:
                    doi_parts = doi_text.split("10.")[1]
                    doi = "10." + doi_parts.split()[0].split("[")[0].split("]")[0].strip()
                    paper_info["DOI"] = doi
                
        return paper_info
    except Exception as e:
        print(f"Error parsing PubMed record: {str(e)}")
        return None

def count_tokens(text: str, model: str = "gpt-4-0125-preview") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

def summarize_papers_with_llm(papers: List[Dict], protein_name: str) -> str:
    """
    Use OpenAI to summarize the papers and create a narrative summary with citations.
    
    Parameters
    ----------
    papers : List[Dict]
        List of paper information from PubMed
    protein_name : str
        Name of the protein being researched
        
    Returns
    -------
    str
        Narrative summary with citations
    """
    if not papers:
        return ""
        
    # Prepare the papers for the LLM
    papers_text = "\n\n".join([
        f"PMID: {p['PMID']}\nTitle: {p['Title']}\nAuthors: {p['Authors']}\n"
        f"Journal: {p['Journal']}\nYear: {p['Year']}\nAbstract: {p['Abstract']}\nDOI: {p['DOI']}"
        for p in papers
    ])
    
    prompt = f"""You are a research assistant specialized in neuroscience. 
    Analyze these papers about {protein_name} and create a comprehensive summary.

    Papers to analyze:
    {papers_text}

    Create a half-page summary that answers these specific questions:
    1. Has {protein_name} been linked to dementia in humans? If yes, which types of dementia?
    2. Has {protein_name} been linked to dementia in animal models? If yes, which types of dementia and which animal models?
    3. Has {protein_name} been linked to other neuropathologies in humans? If yes, which ones?
    4. Has {protein_name} been linked to other neuropathologies in animal models? If yes, which ones and which animal models?

    Important requirements:
    - Write in clear, academic language
    - If no link has been found for any question, explicitly state that
    - For each finding, cite the specific paper(s) it comes from
    - At the end, provide a References section with all cited papers in APA format
    - Each reference MUST include the DOI if available
    - Format references as: Author(s). (Year). Title. Journal, Volume(Issue), Pages. https://doi.org/DOI
    - Only include information that is explicitly stated in the papers
    - Keep the summary focused and concise (half-page)
    - Use proper scientific terminology

    Format the response as a well-structured text with clear paragraphs and a References section at the end.
    """

    # Count tokens in the prompt
    total_tokens = count_tokens(prompt)
    max_tokens = 120000  # Setting a safe limit below the model's 128K limit
    
    if total_tokens > max_tokens:
        print(f"\nWarning: Input exceeds token limit ({total_tokens} tokens). Reducing number of papers...")
        # Reduce papers until we're under the limit
        while total_tokens > max_tokens and len(papers) > 1:
            papers = papers[:-1]  # Remove the last paper
            papers_text = "\n\n".join([
                f"PMID: {p['PMID']}\nTitle: {p['Title']}\nAuthors: {p['Authors']}\n"
                f"Journal: {p['Journal']}\nYear: {p['Year']}\nAbstract: {p['Abstract']}\nDOI: {p['DOI']}"
                for p in papers
            ])
            prompt = f"""You are a research assistant specialized in neuroscience. 
            Analyze these papers about {protein_name} and create a comprehensive summary.

            Papers to analyze:
            {papers_text}

            Create a half-page summary that answers these specific questions:
            1. Has {protein_name} been linked to dementia in humans? If yes, which types of dementia?
            2. Has {protein_name} been linked to dementia in animal models? If yes, which types of dementia and which animal models?
            3. Has {protein_name} been linked to other neuropathologies in humans? If yes, which ones?
            4. Has {protein_name} been linked to other neuropathologies in animal models? If yes, which ones and which animal models?

            Important requirements:
            - Write in clear, academic language
            - If no link has been found for any question, explicitly state that
            - For each finding, cite the specific paper(s) it comes from
            - At the end, provide a References section with all cited papers in APA format
            - Each reference MUST include the DOI if available
            - Format references as: Author(s). (Year). Title. Journal, Volume(Issue), Pages. https://doi.org/DOI
            - Only include information that is explicitly stated in the papers
            - Keep the summary focused and concise (half-page)
            - Use proper scientific terminology

            Format the response as a well-structured text with clear paragraphs and a References section at the end.
            """
            total_tokens = count_tokens(prompt)
        
        print(f"Reduced to {len(papers)} papers to stay within token limit.")

    try:
        client = setup_openai()
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",  # Using OpenAI's deep research model
            messages=[
                {"role": "system", "content": "You are a research assistant specialized in neuroscience. Your task is to analyze and summarize scientific papers, providing clear citations and a comprehensive overview of the findings. Only include information that is explicitly stated in the papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            top_p=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error querying OpenAI: {str(e)}")
        return ""

def save_results_to_txt(summary: str, protein_name: str, queries_dir: str):
    """Save the summary to a text file in the queries_llm directory."""
    if not summary:
        return None
        
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"{protein_name}_summary_{date}.txt"
    filepath = os.path.join(queries_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)
    return filepath

def ensure_queries_directory():
    """Ensure the queries_llm directory exists."""
    queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries_llm')
    os.makedirs(queries_dir, exist_ok=True)
    return queries_dir

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
        summary = summarize_papers_with_llm(papers, protein_name)
        
        # Ensure queries directory exists and save results
        queries_dir = ensure_queries_directory()
        saved_file = save_results_to_txt(summary, protein_name, queries_dir)
        
        # Display results
        if summary:
            print("\nSummary of findings:\n")
            print(summary)
            if saved_file:
                print(f"\nResults have been saved to: {saved_file}")
        else:
            print("No summary was generated or an error occurred during summarization.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 