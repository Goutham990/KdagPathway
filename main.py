import pathway as pw
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm import OpenAIComplete
import pandas as pd
import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import asyncio
 
# Load environment variables
load_dotenv()
 
class PaperEvaluationSystem:
	def __init__(self):
		self.embedder = OpenAIEmbedder(model="text-embedding-ada-002")
		self.llm = OpenAIComplete(model="gpt-4")
		self.vector_store = self._init_vector_store()
		self.conference_profiles = self._init_conference_profiles()
 
	def _init_vector_store(self):
		# Read and preprocess reference papers
		papers = pw.io.gdrive.read("reference_papers/*", format="pdf")
		return papers + self.embedder.apply(
			pw.this.content, output_column="embedding"
		)
 
	def _init_conference_profiles(self):
		return {
			"CVPR": "Computer vision and pattern recognition research...",
			"NeurIPS": "Machine learning and computational neuroscience...",
			"EMNLP": "Natural language processing and computational linguistics...",
			"TMLR": "Machine learning theory and applications...",
			"KDD": "Data mining and knowledge discovery...",
			"DAA": "Data analysis and algorithms...",
		}
 
	def evaluate_publishability(self, paper_content: str) -> Dict[str, Any]:
		prompt = f"""
		Analyze this research paper and determine if it's publishable.
		Consider: Methodology, Coherence, Evidence, Technical validity, Innovation.
 
		Paper: {paper_content[:4000]}...
 
		Return JSON:
		{{
			"publishable": boolean,
			"confidence": float,
			"reasons": list[str],
			"suggestions": list[str]
		}}
		"""
		response = self.llm.complete(prompt)
		return json.loads(response)
 
	def select_conference(self, paper_content: str) -> Dict[str, Any]:
		paper_embedding = self.embedder.embed_text(paper_content)
		similar_papers = self.vector_store.select(
			similarity=pw.similarities.cosine(pw.this.embedding, paper_embedding)
		).sort(pw.this.similarity, reverse=True).limit(3)
		prompt = f"""
		Recommend a conference (CVPR, NeurIPS, etc.) for the paper.
 
		Paper: {paper_content[:3000]}...
		Similar papers: {similar_papers}
		Conference profiles: {json.dumps(self.conference_profiles, indent=2)}
 
		Return JSON:
		{{
			"conference": str,
			"confidence": float,
			"rationale": str,
			"alternative_conferences": list[str]
		}}
		"""
		response = self.llm.complete(prompt)
		return json.loads(response)
 
	def process_papers(self, input_dir: str) -> pd.DataFrame:
		results = []
		for paper_file in os.listdir(input_dir):
			if not paper_file.endswith('.txt'):
				continue
			with open(os.path.join(input_dir, paper_file), 'r') as f:
				content = f.read()
			publishability = self.evaluate_publishability(content)
			if publishability["publishable"]:
				conference = self.select_conference(content)
				results.append({
					"Paper ID": paper_file.split('.')[0],
					"Publishable": 1,
					"Conference": conference["conference"],
					"Rationale": conference["rationale"]
				})
			else:
				results.append({
					"Paper ID": paper_file.split('.')[0],
					"Publishable": 0,
					"Conference": "na",
					"Rationale": "na"
				})
		return pd.DataFrame(results)
 
# FastAPI Implementation
app = FastAPI()
evaluator = PaperEvaluationSystem()
 
class PaperSubmission(BaseModel):
	content: str
	metadata: Dict[str, Any] = None
 
@app.post("/evaluate")
async def evaluate_paper(submission: PaperSubmission):
	try:
		publishability = evaluator.evaluate_publishability(submission.content)
		if publishability["publishable"]:
			conference = evaluator.select_conference(submission.content)
			return {
				"publishable": True,
				"conference": conference["conference"],
				"rationale": conference["rationale"],
				"confidence": conference["confidence"]
			}
		return {
			"publishable": False,
			"conference": "na",
			"rationale": "na",
			"suggestions": publishability["suggestions"]
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
 
@app.post("/batch")
async def batch_process(files: list[UploadFile] = File(...)):
	tasks = [
		evaluate_paper(PaperSubmission(content=(await file.read()).decode()))
		for file in files
	]
	return await asyncio.gather(*tasks)
 
if __name__ == "__main__":
	# For batch processing
	results_df = evaluator.process_papers("./input_papers")
	results_df.to_csv("results.csv", index=False)


