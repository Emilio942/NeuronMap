# Contributing to NeuronMap

Thank you for your interest in contributing to NeuronMap! This document provides guidelines for contributing to our neural network activation analysis toolkit.

## 🎯 Project Overview

NeuronMap is a comprehensive toolkit for analyzing neural network activations, providing:
- Multi-model activation extraction and analysis
- Advanced interpretability methods (CAVs, saliency, activation maximization)
- Experimental analysis techniques (RSA, CKA, probing)
- Domain-specific analysis for code, math, multilingual, and temporal data
- Ethics and bias analysis capabilities
- Scientific rigor with statistical testing and experiment logging

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of neural networks and PyTorch/Transformers

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/Emilio942/NeuronMap.git
   cd NeuronMap
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Run tests to verify setup:**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Run the validation command:**
   ```bash
   python main.py validate
   ```

## 📋 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug Reports**: Found a bug? Please report it!
- **✨ Feature Requests**: Have an idea for a new feature?
- **🔧 Code Contributions**: Fix bugs or implement new features
- **📚 Documentation**: Improve docs, add examples, write tutorials
- **🧪 Testing**: Add tests, improve test coverage
- **🎨 Visualization**: Enhance plots and interactive visualizations
- **🔬 Analysis Methods**: Implement new interpretability techniques

### 🐛 Reporting Bugs

Before reporting a bug:
1. Check if the issue already exists in [GitHub Issues](https://github.com/Emilio942/NeuronMap/issues)
2. Try to reproduce the bug with a minimal example
3. Check if you're using the latest version

When reporting a bug, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces
- Minimal code example if possible

**Use the bug report template:**
```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Use configuration '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- NeuronMap version: [e.g. 0.1.0]
- CUDA version (if applicable): [e.g. 11.8]

**Additional Context**
Any other context about the problem.
```

### ✨ Feature Requests

For feature requests:
1. Check existing issues and discussions
2. Describe the use case and motivation
3. Provide examples of how it would work
4. Consider implementation complexity

**Use the feature request template:**
```markdown
**Feature Description**
A clear description of what you want to achieve.

**Motivation**
Why is this feature useful? What problem does it solve?

**Proposed Solution**
How would you like this to work?

**Example Usage**
```python
# Example of how the feature would be used
```

**Additional Context**
Any other context or screenshots.
```

### 🔧 Code Contributions

#### Workflow

1. **Create an issue** (if one doesn't exist) to discuss the change
2. **Fork the repository** and create a feature branch
3. **Make your changes** following our coding standards
4. **Add or update tests** for your changes
5. **Update documentation** if needed
6. **Run the full test suite** and ensure all tests pass
7. **Submit a pull request** with a clear description

#### Branch Naming

Use descriptive branch names:
- `feature/add-bert-support` - New features
- `fix/activation-extraction-bug` - Bug fixes
- `docs/improve-api-docs` - Documentation updates
- `test/add-integration-tests` - Test improvements

#### Coding Standards

**Python Style:**
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use [mypy](https://mypy.readthedocs.io/) for type checking

**Code Organization:**
- Keep functions small and focused
- Use meaningful variable and function names
- Add type hints to all functions
- Write comprehensive docstrings

**Documentation:**
- Follow [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Include examples in docstrings where helpful
- Update README.md for significant changes

#### Testing

- **Write tests** for all new functionality
- **Maintain coverage** - aim for >90% test coverage
- **Test types:**
  - Unit tests for individual functions
  - Integration tests for complete workflows
  - Property-based tests with Hypothesis
  - Mock tests for external dependencies

**Test Structure:**
```python
def test_function_name():
    """Test description of what is being tested."""
    # Arrange - Set up test data
    input_data = create_test_data()
    
    # Act - Execute the function
    result = function_under_test(input_data)
    
    # Assert - Verify the results
    assert result.expected_property == expected_value
    assert len(result.items) == expected_count
```

#### Pull Request Guidelines

**PR Title Format:**
- `feat: add BERT model support`
- `fix: resolve activation extraction memory leak`
- `docs: update installation instructions`
- `test: add integration tests for visualization`

**PR Description Template:**
```markdown
## Description
Brief description of changes.

## Motivation and Context
Why is this change required? What problem does it solve?

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Integration tests updated if needed

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation made
- [ ] Changes generate no new warnings
```

## 🏷️ Good First Issues

New contributors can start with issues labeled `good-first-issue`. These are typically:

### Beginner-Friendly Tasks:
- **Documentation improvements**: Fix typos, improve examples
- **Test coverage**: Add tests for uncovered functions
- **Code cleanup**: Improve code style, add type hints
- **Example scripts**: Create tutorial notebooks or example scripts

### Intermediate Tasks:
- **New visualization methods**: Add new plot types or interactive features
- **Model support**: Add support for new transformer architectures
- **Utility functions**: Implement helper functions for data processing
- **CLI enhancements**: Improve command-line interface

### Advanced Tasks:
- **New analysis methods**: Implement interpretability techniques
- **Performance optimizations**: Improve memory usage or speed
- **Integration features**: Add support for new frameworks
- **Research implementations**: Implement methods from recent papers

## 📚 Documentation Contributions

### Types of Documentation:
- **API Documentation**: Improve docstrings and type hints
- **Tutorials**: Step-by-step guides for specific tasks
- **Examples**: Real-world usage examples
- **Architecture docs**: Explain system design and components

### Writing Guidelines:
- Use clear, concise language
- Include practical examples
- Test all code examples
- Follow the existing documentation style
- Update relevant sections when adding features

## 🧪 Research Contributions

We especially welcome contributions of:
- **New interpretability methods** from recent research papers
- **Novel visualization techniques** for neural activations
- **Evaluation metrics** for interpretability methods
- **Benchmark datasets** for testing analysis methods
- **Case studies** showing practical applications

## 🎨 Design and UI Contributions

For visualization and interface improvements:
- Follow accessibility guidelines
- Ensure compatibility across different platforms
- Test with different screen sizes and devices
- Consider colorblind-friendly color schemes
- Maintain consistency with existing designs

## 🔧 Development Tools

### Recommended VS Code Extensions:
- Python
- Pylance
- Black Formatter
- autoDocstring
- GitLens
- Docker

### Pre-commit Hooks:
We use pre-commit hooks for code quality:
```bash
pip install pre-commit
pre-commit install
```

This will automatically run:
- Black (code formatting)
- flake8 (linting)
- mypy (type checking)
- isort (import sorting)

## 🎖️ Recognition

Contributors will be:
- Listed in the project contributors
- Mentioned in release notes for significant contributions
- Invited to join the core team for consistent contributors
- Credited in academic papers that use substantial contributions

## 📞 Getting Help

If you need help:
- **Discord**: Join our [Discord server](https://discord.gg/neuronmap)
- **GitHub Discussions**: Use [GitHub Discussions](https://github.com/Emilio942/NeuronMap/discussions)
- **Email**: Contact maintainers at neuronmap@example.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Every contribution, no matter how small, helps make NeuronMap better for everyone. Thank you for your interest in improving neural network interpretability tools!

---

*Happy Contributing!* 🚀
