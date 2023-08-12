Create Pyhton3 environment:
python3 -m venv /path/to/new/virtual/environment

Activate virtual environment:
source /path/to/new/virtual/environment/bin/activate

Install all dependencies:
pip install requirements.txt


**WORKFLOW:**
TECH STACK: Python3, PyTorch, Spark, PostgreSQL, GIT, Docker, Grafana why not 

**
1. Read all file samples names, store in list
2. Generate JSON files with metainformation about each
 radar sample : label, context, number of persons in the radar range, the radar sample's values,
 the shape of initial radar sample and its type
3. Store the JSON files in a PostgreSQL database - UWB_Radar_Samples.
4. Visualize radar samples from each scenario
5. Make a shallow analysis 
6. Clean data: filtering, clutter removal, extract the dc component 
7. Extract features: PCA 
**
8. Provide as input for CNN, CRNN or Transformers 
9. Analyse metrics 
10. Save model 
11. Create final pipeline
12. Deploy in Docker Container