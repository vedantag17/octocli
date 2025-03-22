# 🐙 Octo – The AI GitHub Assistant
**Octo** is a command-line tool with commands similar to git. It is powered by Large Language Models it allows users to QA the GitHub repository with native git commands and add a tell command that answers all your questions regarding the repository.

## Description
When contributing to a project on GitHub, a developer needs to adapt/learn the Project goals, specifications. Commercially available LLMs do not read a whole repository, a developer needs to manually browse through each folder, sub-folder and its contents etc. Each folder varies in size and structural complexity. Manual browsing takes much time of developer, decreasing productivity.

Unlike other commercially available LLMs, **Octo**, A CLI application can directly ingest a GitHub repository when given the link. It fetches all file structures, issues, codebases etc. It creates a Readme markdown file in memory which acts as a knowledge base of the dedicated repository for fast retrieval, for user interaction. Octo saves much time of developers i.e switching between tabs, manual code reviewing. Octo understands the Repository and the relationships present in it, providing better insights, and accurate explainations.

---
## Key Features   
- **Display Repository File Structure in CLI**  
- **Read and Summarize README**    
- **LLM-Powered Chat for Q&A about Issues/Repo**   
---

## Tech Stack   
- **PyGitHub**: GitHub Python package
- **CLI Framework**: Typer, argparse  
- **LLM Integration**: OpenAI or Hugging Face  
---

## Installation  

```bash
pip install -r requirements.txt
pip install .
```


---
# Start the CLI
![WhatsApp Image 2025-03-22 at 05 23 34_8c0d46fe](https://github.com/user-attachments/assets/11139736-79c4-47b9-bcc1-bc5368e5db06)

# Commands
![Uploading WhatsApp Image 2025-03-22 at 05.22.15_e4b42286.jpg…]()


---
