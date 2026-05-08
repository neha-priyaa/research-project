 # AGENTS.md

This guide provides essential information for AI agents working in this research project codebase.

## Project Overview

This is a Python-based academic research repository focused on machine learning, artificial intelligence, data science, and quantitative analysis. The project follows a clean, modular structure separating code, data, tests, notebooks, and documentation.

## Essential Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda (if environment.yml exists)
conda env create -f environment.yml
conda activate research_env
```

### Testing
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils.py

# Run with verbose output
pytest tests/ -v

# Run tests directly from test module
cd tests/
python __init__.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Analysis and Development
```bash
# Start Jupyter notebook
jupyter notebook

# Start JupyterLab
jupyter lab
```

## Project Structure

```
research_project/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── .gitignore              # Git ignore rules
├── papers/                 # Research papers and documentation
│   ├── drafts/             # Work-in-progress papers
│   ├── reviews/            # Literature reviews
│   ├── published/          # Published research
│   ├── resources/          # Research resources
│   └── notes/              # Research notes
├── src/                      # Source code (main package)
│   ├── __init__.py         # Package initialization (contains commented imports)
│   ├── models/             # Machine learning models
│   │   └── __init__.py     # BaseModel, ResearchModel classes
│   └── utils/              # Utility functions
│       └── __init__.py     # Data loading, cleaning, normalization
├── data/                     # Data storage
│   ├── raw/                # Raw data files (contains .gitkeep)
│   └── processed/          # Processed/cleaned data (contains .gitkeep)
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                   # Test suite
│   └── __init__.py         # Test file (adds src to sys.path)
└── docs/                    # Additional documentation
    └── SETUP_GUIDE.md      # Detailed setup and usage guide
```

## Code Organization and Patterns

### Module Structure
- **src/**: Main Python package containing research code
  - `models/`: ML model implementations (sklearn-based)
  - `utils/`: Data processing utilities
- **tests/**: Pytest test suite
- **papers/**: Research materials organized by stage (drafts, published, etc.)
- **data/**: Data storage with raw/processed separation

### Design Patterns

**Base Class Pattern (models/__init__.py)**:
- `BaseModel` is an abstract base class with `fit()`, `predict()`, `score()` methods
- `ResearchModel(BaseModel)` provides concrete implementations
- Models track `is_fitted` state to prevent predictions before training
- Supported model types: 'random_forest', 'logistic'

**Utility Functions (utils/__init__.py)**:
- Simple, functional design for data operations
- `load_data()`: CSV loading with pandas
- `save_data()`: CSV saving (no index)
- `clean_data()`: Drops duplicates and missing values
- `normalize_data()`: Z-score normalization

### Import Structure

**Note**: The `src/__init__.py` file has commented-out imports:
```python
# from .models import *
# from .utils import *
```

To use package modules, import explicitly:
```python
from src.utils import load_data, clean_data
from src.models import ResearchModel
```

**Testing import note**: `tests/__init__.py` modifies `sys.path` to add `src` to the import path:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

## Code Style and Conventions

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `load_data`, `model_type`)
- **Classes**: `PascalCase` (e.g., `BaseModel`, `ResearchModel`)
- **Constants**: `UPPER_CASE` (not commonly used in current codebase)
- **Private members**: `_leading_underscore` (minimal usage currently)

### Docstring Style
- Module-level docstrings describe purpose (triple-quoted)
- Function docstrings are brief, single-line descriptions
- Docstrings use triple quotes (`"""`)
- No complex parameter/return documentation in current implementation

### Code Style
- Follows PEP 8
- Uses Black for formatting (configured in requirements.txt)
- Uses flake8 for linting (configured in requirements.txt)
- Simple, readable code without complex abstractions
- Minimal inline comments

### Type Hints
- Not currently used in the codebase
- SETUP_GUIDE mentions adding them "where appropriate" for future work

## Testing Approach

### Test Framework
- Uses `pytest` (version 6.2.0+)
- Tests located in `tests/__init__.py`
- Tests can run standalone (`python tests/__init__.py`) or via pytest

### Test Patterns
Current tests are placeholders with basic structure:
```python
def test_feature():
    """Test specific feature"""
    assert True
```

Pattern shown in SETUP_GUIDE:
```python
def test_feature():
    """Test specific feature"""
    # Arrange
    input_data = ...

    # Act
    result = function(input_data)

    # Assert
    assert expected == result
```

### Testing Commands
- Standard pytest: `pytest tests/`
- Coverage: `pytest tests/ --cov=src --cov-report=html`

## Dependencies and Libraries

### Core Data Science
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`

### Machine Learning
- `scikit-learn>=0.24.0`
- `xgboost>=1.4.0`
- `lightgbm>=3.3.0`

### Visualization
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `plotly>=5.0.0`

### Jupyter
- `jupyter>=1.0.0`
- `jupyterlab>=3.0.0`
- `ipython>=7.25.0`

### Data Processing
- `requests>=2.26.0`
- `beautifulsoup4>=4.9.0`

### Testing & Code Quality
- `pytest>=6.2.0`
- `pytest-cov>=2.12.0`
- `black>=21.0.0`
- `flake8>=3.9.0`

## Important Gotchas and Non-Obvious Patterns

### Git and Version Control
1. **Data files are NOT committed** to git (except `.gitkeep` placeholders)
   - Extensions excluded: `.csv`, `.json`, `.pkl`, `.pickle`, `.h5`, `.hdf5`
   - Placeholders kept: `data/raw/.gitkeep`, `data/processed/.gitkeep`

2. **Trained models are NOT committed**:
   - Extensions excluded: `.model`, `.h5`, `.pth`

3. **Logs and outputs are excluded**:
   - Directories: `logs/`, `outputs/`, `results/`
   - Files: `*.log`

4. **Build artifacts excluded**: Standard Python build directories (`__pycache__`, `dist`, `*.egg-info/`, etc.)

### Import Path Manipulation
The test file manually adds `src` to Python path:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

This means tests can import modules directly (e.g., `import utils`) without the `src.` prefix. This is a testing-specific pattern.

### Package Initialization
The main `src/__init__.py` does not import submodules automatically. All imports must be explicit:
```python
from src.utils import load_data
from src.models import ResearchModel
```

### Data Management Practices
- Raw data: Never modify original files in `data/raw/`
- Processed data: Store in `data/processed/` with documentation
- Versioning: Use version numbers like `data_v1.csv`, `data_v2.csv`
- Maintain data lineage: Keep processing scripts documented

### Current Implementation State
- Many tests are placeholders with `assert True`
- Notebooks directory is currently empty
- No Makefile, pyproject.toml, or setup.py (only requirements.txt)
- No CI/CD workflows configured
- No environment.yml file exists (only mentioned in docs as optional)

### Code Style Enforcement
- Black and flake8 are in requirements.txt but no config files found
- Format with: `black src/ tests/`
- Lint with: `flake8 src/ tests/`

## Workflow Recommendations

### When Adding New Code
1. Place in appropriate `src/` subdirectory (models/, utils/, or create new module)
2. Write tests in `tests/` directory
3. Update documentation in `docs/` or `README.md`
4. Format with Black: `black <file>`
5. Run flake8: `flake8 <file>`
6. Run tests: `pytest tests/`

### When Adding Data Processing
1. Store raw data in `data/raw/`
2. Save processed data to `data/processed/`
3. Document cleaning steps and transformations
4. Keep processing scripts in `src/` or `notebooks/`

### When Adding Research Materials
- Drafts → `papers/drafts/`
- Literature reviews → `papers/reviews/`
- Published papers → `papers/published/`
- Resources/datasets → `papers/resources/`
- Notes/ideas → `papers/notes/`

## Common Tasks

### Running Analysis
1. Start Jupyter: `jupyter notebook`
2. Navigate to `notebooks/` directory
3. Create or open notebook
4. Import from src: `from src.utils import load_data`

### Training a Model
```python
from src.models import ResearchModel
from src.utils import load_data, clean_data

# Load and prepare data
data = load_data('data/raw/dataset.csv')
data = clean_data(data)

# Create and train model
model = ResearchModel(model_type='random_forest')
X_train, y_train = data.drop('target', axis=1), data['target']
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Adding a New Model
1. Create class in `src/models/__init__.py` or new file
2. Inherit from `BaseModel` or implement `fit()`, `predict()`, `score()`
3. Add tests in `tests/`
4. Update documentation

## Key Files Reference

| File | Purpose |
|------|---------|
| `requirements.txt` | All Python dependencies |
| `src/__init__.py` | Package init (note: commented imports) |
| `src/utils/__init__.py` | Data utilities |
| `src/models/__init__.py` | ML model classes |
| `tests/__init__.py` | Test suite (modifies sys.path) |
| `docs/SETUP_GUIDE.md` | Detailed usage guide |
| `README.md` | Project overview and quick start |

## Notes for AI Agents

- **Testing**: Many tests are placeholders - this is a work-in-progress project
- **Documentation**: Some references in docs mention files that don't yet exist (e.g., environment.yml)
- **Git**: The outer directory `/Users/nehapriya/research_project` contains a submodule `research-project/`. Work should happen inside `research-project/`
- **No CI/CD**: No automated testing or deployment configured
- **Minimal Config**: No pyproject.toml, Makefile, or build system - everything runs through standard Python tools
