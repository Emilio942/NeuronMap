# Sphinx Documentation Configuration
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'NeuronMap'
copyright = '2025, NeuronMap Team'
author = 'NeuronMap Team'
release = '1.0.0'
version = '1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',           # Automatic documentation from docstrings
    'sphinx.ext.napoleon',          # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.intersphinx',       # Link to other project's documentation
    'sphinx.ext.mathjax',           # Math support
    'sphinx_autodoc_typehints',     # Type hints in documentation
    'sphinx.ext.doctest',           # Test code snippets
    'sphinx.ext.coverage',          # Coverage statistics
    'sphinxcontrib.mermaid',        # Mermaid diagrams
    'myst_parser',                  # Markdown support
    'nbsphinx',                     # Jupyter notebook support
]

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': None,
    '.ipynb': None,
}

# Master document
master_doc = 'index'

# Language
language = 'en'

# Exclude patterns
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'style_external_links': True,
}

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'neuronmap',
    'github_repo': 'neuronmap',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# Static files
html_static_path = ['_static']
html_css_files = ['custom.css']

# HTML output options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# Napoleon configuration (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'transformers': ('https://huggingface.co/docs/transformers', None),
}

# Math configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# NBSphinx configuration (for Jupyter notebooks)
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 600

# Coverage configuration
coverage_show_missing_items = True

# Mermaid configuration
mermaid_version = "9.4.0"

# Custom CSS and JS
def setup(app):
    """Custom setup function."""
    app.add_css_file('custom.css')
