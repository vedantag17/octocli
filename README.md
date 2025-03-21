# 🐙 Octo — Your AI-Powered GitHub Companion 

**Octo** is a command-line tool with commands similar to git. It is powered by Large Language Models it allows users to QA the GitHub repository with native git commands and add a tell command that answers all your questions regarding the repository.
 
Just provide a GitHub issue link or issue number — Octo will:  
- Fetch the repo's file structure  
- Display and summarize the README  
- Index everything into a vector database  
- Let you ask questions interactively via an LLM-powered assistant  

Perfect for contributors, QAs, and developers of all skill levels — Octo ensures a smooth, productive, and beginner-friendly experience.  

---

## Key Features  
- **Fetch by Issue Link or Issue Number**  
- **Display Repository File Structure in CLI**  
- **Read and Summarize README**  
- **Index Data Using PyMilvus for Smart Retrieval**  
- **LLM-Powered Chat for Q&A about Issues/Repo**  
- **Designed for Beginners & Experts**  

---

## Tech Stack  
- **Vector Database**: PyMilvus  
- **GitHub API**: GitHub Python package  
- **CLI Framework**: Typer, argparse  
- **LLM Integration**: OpenAI or Hugging Face  

---

## Installation  

```bash
git clone https://github.com/yourusername/octo.git
cd octo
pip install -r requirements.txt
```


---
# Start the CLI
python octo.py

# Example: Provide an issue URL
octo --issue-link https://github.com/username/repo/issues/42

# Example: Provide repo and issue number
octo --repo username/repo --issue-number 42

# Start asking questions about the repo or issue!
---

