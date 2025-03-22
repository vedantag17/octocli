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
## Start the CLI
![WhatsApp Image 2025-03-22 at 05 23 34_7a94d29e](https://github.com/user-attachments/assets/b7f489ee-0c47-43ef-ada1-8e0166fb7091)

## Commands
![WhatsApp Image 2025-03-22 at 05 22 15_ba01756d](https://github.com/user-attachments/assets/135b4a9b-4abb-46d1-b510-c01f0127ff08)

---
