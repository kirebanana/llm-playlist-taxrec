# LLM Playlist Generator (Taxonomy Approach)

This project generates music playlists based on natural language prompts (like "chill study music"). It uses **GPT-4o** to understand the request and match it against a **structured taxonomy** (genre, mood, era, etc.) applied to a local song collection.

It's inspired by the **TAXREC** paper and. The main report and results are in `report.ipynb`.

---

### Setup and run
1.  **Clone:** `git clone https://github.com/kirebanana/llm-playlist-taxrec.git && cd llm-playlist-taxrec`
2.  **Make a virtual environment:**
    `python -m venv .venv`
3.  **Activate it:**
    * Windows: `.venv\Scripts\activate`
    * Mac/Linux: `source .venv/bin/activate`
4.  **Install requirements:**
    `pip install -r requirements.txt`
5.  **API Key:**
    * Create a `.env` file in the root folder.
    * Add your OpenAI key: `OPENAI_API_KEY="sk-..."`

