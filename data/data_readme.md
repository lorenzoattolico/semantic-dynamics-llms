# Data Directory

This directory stores data files for the Semantic Dynamics in LLMs project.

## Files

### Generated Files

- `semantic_edges.csv`: Edge list of the semantic network (columns: `source`, `target`)
- `semantic_nodes.csv`: Node list of the semantic network (columns: `node`, `degree`)

### Experimental Results

- `spreading_results.csv`: Results of spreading activation calculations (intermediate file)
- `semantic_analysis_results.csv`: Final dataset with spreading activation and Likert similarity scores

## File Formats

### semantic_edges.csv

Edge list representing the semantic network:

```
source,target
dog,animal
car,vehicle
...
```

### semantic_nodes.csv

Node list with degree information:

```
node,degree
dog,15
car,8
...
```

### spreading_results.csv

Spreading activation results for different parameter configurations:

```
Concept1,Concept2,PathLength,r1_t1_AtoB,r1_t1_BtoA,r1_t1_AVG,...
dog,animal,1,45.2,42.8,44.0,...
car,vehicle,1,38.5,36.9,37.7,...
...
```

### semantic_analysis_results.csv

Final dataset with both spreading activation and Likert similarity scores:

```
Concept1,Concept2,PathLength,r1_t1_AtoB,r1_t1_BtoA,r1_t1_AVG,...,LikertScore
dog,animal,1,45.2,42.8,44.0,...,7
car,vehicle,1,38.5,36.9,37.7,...,6
...
```

## Note

Do not manually modify these files. They are generated and updated by the scripts in the `src` directory.
