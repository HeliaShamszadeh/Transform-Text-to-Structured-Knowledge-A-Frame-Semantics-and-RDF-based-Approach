# Transform-Text-to-Structured-Knowledge-A-Frame-Semantics-and-RDF-based-Approach

A comprehensive pipeline for extracting semantic knowledge from text using Semantic Role Labeling (SRL), Entity Linking, and RDF conversion with advanced evaluation capabilities.

## System Architecture

### Pipeline Components

1. **Semantic Role Labeling (SRL)** - Extracts semantic frames and roles from text
2. **Entity Linking (REL)** - Identifies and links entities to knowledge bases
3. **RDF Conversion** - Transforms frames and entities into RDF triples
4. **Evaluation System** - LLM-based accuracy assessment and comprehensive metrics

### Models and Technologies Used

#### SRL Module
- **Model**: `frame-semantic-transformer` (base model)
- **Framework**: PyTorch-based transformer architecture
- **Capabilities**: 50+ semantic frame types, 300+ semantic roles
- **Input**: Raw text files
- **Output**: JSON with frames and semantic elements

#### Entity Linking Module
- **Model**: REL (Robust Entity Linking) system
- **Confidence Threshold**: 0.35-0.38 (configurable)
- **Knowledge Base**: Wikipedia/DBpedia entities
- **Output**: JSON with linked entities and URIs

#### RDF Conversion Module
- **Framework**: `rdflib` (Python RDF library)
- **Pronoun Resolution**: 
  - **LLM-based**: DeepInfra API (Llama-2-70b-chat-hf)
  - **Heuristic-based**: Pattern matching algorithms
- **Output**: Turtle (.ttl) RDF files

#### Evaluation System
- **LLM Evaluator**: Deepseek v3 API
- **Metrics**: Confidence analysis
- **Batch Processing**: Multi-author evaluation
- **Visualization**: NetworkX + Matplotlib graphs

## Quick Start

### Prerequisites
- Python 3.11+ (for SRL module)
- Python 3.10+ (for other modules)
- Internet connection (for APIs)
- 8GB+ RAM (for transformer models)

### 1. Setup

```bash
# Clone and navigate to project
cd Transform-Text-to-Structured-Knowledge-A-Frame-Semantics-and-RDF-based-Approach

# Run simple setup (creates virtual environments)
python setup_simple.py
```

This creates three virtual environments:
- `.srl_env` - For SRL processing (Python 3.11)
- `.orchestrator_env` - For RDF conversion (Python 3.10+)
- `.rel_env` - For entity linking (Python 3.10+)

### 2. Run Basic Pipeline

```bash
# Process sample text through complete pipeline
python run_pipeline.py
```

This will:
- Create sample input text (if not provided by user)
- Run SRL → Entity Linking → RDF conversion
- Generate outputs in `outputs/` directory

## Detailed Usage

### Author Data Extraction

#### Method 1: Wikipedia Scraping
```bash
# Create authors list
echo "Agatha Christie" > authors.txt
echo "Albert Einstein" >> authors.txt
echo "William Shakespeare" >> authors.txt

# Scrape Wikipedia biographies
python data_collection/wikipedia_scraper.py
```

#### Method 2: Use Existing Data
```bash
# Copy existing author files
cp -r inputs/authors_reduced_more/* inputs/authors/
```

#### Method 3: Reduce Long Texts (Optional: I did it since it would have become so time-consuming!)
```bash
# Reduce text length to prevent SRL timeouts (25k chars max)
python data_collection/reduce_author_contents.py

# Or chunk very long texts into smaller pieces
python data_collection/text_chunker.py
```

#### Method 4: Custom Text Files
```bash
# Create your own text files
echo "Your biographical text here..." > inputs/authors/author_name.txt
```

### Batch Processing

#### Process All Authors
```bash
# Run complete pipeline on all authors
python batch_pipeline.py
```

#### Process Specific Authors
```bash
# Process only selected authors
python batch_pipeline.py --authors "Agatha_Christie" "Albert_Einstein"
```

#### Configuration Options
```bash
# Custom output directory
python batch_pipeline.py --output-dir "my_outputs"

# Custom confidence threshold
python batch_pipeline.py --confidence 0.4

# Limit number of authors
python batch_pipeline.py --limit 5

# Process authors sequentially
python batch_pipeline.py --sequential
```

### Individual Module Usage

#### SRL Only
```bash
# Activate SRL environment
.srl_env\Scripts\activate  # Windows
# .srl_env/bin/activate     # Linux/Mac

# Run SRL
python modules/framesrl/framesrl_runner.py \
    --infile "input.txt" \
    --outfile "output/frames.json" \
    --model base
```

#### Entity Linking Only
```bash
# Activate REL environment
.rel_env\Scripts\activate  # Windows

# Run entity linking
python modules/rel_linker/rel_runner_fixed.py \
    --infile "input.txt" \
    --outfile "output/entities.json" \
    --confidence 0.35
```

#### RDF Conversion Only
```bash
# Activate orchestrator environment
.orchestrator_env\Scripts\activate  # Windows

# Run RDF conversion
python modules/orchestrator/rdfify_improved.py \
    --frames "frames.json" \
    --entities "entities.json" \
    --outfile "output.rdf.ttl" \
    --deepinfra-api-key "your_key"
```

## Evaluation and Analysis

### Run Complete Evaluation
```bash
# Run full evaluation pipeline
python evaluation_pipeline/evaluation_orchestrator.py
```

### Generate Evaluation Summary
```bash
# Generate comprehensive evaluation report
python accurate_evaluation_summary.py
```

### Visualize Knowledge Graphs
```bash
# Generate PNG visualizations of all author graphs
python simple_graph_generator.py
```

### Custom Evaluation
```bash
# Evaluate single author
python evaluate_single_direct.py --author "Agatha_Christie"

# Test specific components
python evaluation_pipeline/test_evaluation.py
```

## Graph Visualization

### Generate All Graphs
```bash
# Create PNG visualizations for all authors
python simple_graph_generator.py
```

### Custom Graph Generation
```python
from simple_graph_generator import create_author_graph

# Generate graph for specific author
create_author_graph("Agatha_Christie", 
                   data_dir="evaluation_outputs", 
                   output_dir="my_graphs")
```

### Graph Features
- **Node Types**: Entities (red), Objects (blue)
- **Edge Labels**: Frame:Predicate relationships
- **Layout**: Spring layout with automatic sizing
- **Format**: High-resolution PNG files
- **Output**: Saved in `graphs/` directory

## Output Structure

```
evaluation_outputs/
├── Author_Name/
│   ├── srl/
│   │   └── Author_Name_frames.json
│   ├── rel/
│   │   └── Author_Name_entities.json
│   ├── rdf/
│   │   ├── Author_Name_rdf.ttl
│   │   └── Author_Name_triples.csv
│   └── graphs/
│       └── Author_Name_graph.png
├── processing_results.json
└── accuracy_report.txt
```

## Configuration

### API Keys
Set environment variables or update config files:

```bash
# DeepInfra API (for pronoun resolution)
export DEEPINFRA_API_KEY="your_deepinfra_key"

# Deepseek API (for evaluation)
export DEEPSEEK_API_KEY="your_deepseek_key"
```

### Model Parameters
- **SRL Model**: `base` (recommended) or `small`
- **Entity Confidence**: 0.35-0.40 (adjust based on precision needs)
- **Max Triples**: 50 per author (for evaluation)
- **Text Length**: 50,000 characters max per author (use `reduce_author_contents.py` for longer texts)

## Troubleshooting

### Common Issues

1. **Virtual Environment Errors**
   ```bash
   # Recreate environments
   rm -rf .srl_env .orchestrator_env .rel_env
   python setup_simple.py
   ```

2. **Memory Issues**
   ```bash
   # Use smaller SRL model
   python modules/framesrl/framesrl_runner.py --model small
   ```

3. **API Rate Limits**
   ```bash
   # Add delays between requests
   python batch_pipeline.py --delay 2
   ```

4. **Unicode Errors**
   - Fixed in latest version with proper encoding handling
   - Ensure UTF-8 encoding for all text files

### Performance Optimization

- **Parallel Processing**: Use `--workers` parameter for batch processing
- **Memory Management**: Process authors in smaller batches
- **Caching**: Reuse SRL/REL outputs when possible

## 📈 Evaluation Metrics

### Accuracy Metrics
- **Precision**: Correctly extracted triples / Total extracted triples
- **Recall**: Correctly extracted triples / Total possible triples
- **F1-Score**: Harmonic mean of precision and recall

### Confidence Analysis
- **High Confidence**: ≥0.8 (Very High)
- **Medium Confidence**: 0.5-0.8 (High)
- **Low Confidence**: <0.5 (Low/Medium)

### Coverage Metrics
- **Frame Coverage**: Number of different semantic frames detected
- **Entity Coverage**: Number of unique entities linked
- **Triple Density**: Average triples per sentence

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## icense

This project is for educational and research purposes.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review error logs in `evaluation_outputs/`
3. Test with sample data first
4. Verify API keys and dependencies

---

**Last Updated**: September 2025  
**Version**: 1.0  
**Python Compatibility**: 3.10+ (SRL: 3.11+)
