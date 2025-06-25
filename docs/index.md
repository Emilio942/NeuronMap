# NeuronMap Documentation

```{include} ../README_NEW.md
:start-after: "# NeuronMap - Neural Network Activation Analysis Framework"
:end-before: "## 🤝 Contributing"
```

## 📚 Documentation Index

### Quick Start
- {doc}`installation/index` - Installation guides for all platforms
- {doc}`tutorials/quickstart` - Get started in 5 minutes
- {doc}`examples/basic_usage` - Basic usage examples

### User Guides
- {doc}`tutorials/index` - Step-by-step tutorials
- {doc}`examples/index` - Comprehensive examples
- {doc}`configuration/index` - Configuration system guide

### API Reference
- {doc}`api/index` - Complete API documentation
- {doc}`api/analysis` - Analysis module reference
- {doc}`api/visualization` - Visualization module reference
- {doc}`api/utils` - Utilities module reference

### Advanced Topics
- {doc}`research/index` - Research methodology guide
- {doc}`research/experimental_design` - Experimental design patterns
- {doc}`research/reproducibility` - Reproducibility framework
- {doc}`advanced/index` - Advanced usage and customization

### Development
- {doc}`development/index` - Development guide
- {doc}`development/contributing` - How to contribute
- {doc}`development/testing` - Testing framework

### Help & Support
- {doc}`troubleshooting/index` - Common issues and solutions
- {doc}`faq` - Frequently asked questions
- {doc}`changelog` - Version history and changes

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

installation/index
tutorials/quickstart
examples/basic_usage
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

tutorials/index
examples/index
configuration/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/index
api/analysis
api/visualization
api/utils
api/data_generation
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Research & Advanced

research/index
research/experimental_design
research/reproducibility
advanced/index
advanced/plugins
advanced/custom_models
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Development

development/index
development/contributing
development/testing
development/architecture
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Support

troubleshooting/index
faq
changelog
```

## 🎯 Quick Navigation

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 Get Started
:link: tutorials/quickstart
:link-type: doc

New to NeuronMap? Start here for a quick introduction and basic setup.
:::

:::{grid-item-card} 📖 API Reference
:link: api/index
:link-type: doc

Complete API documentation with examples and type hints.
:::

:::{grid-item-card} 🔬 Research Guide
:link: research/index
:link-type: doc

Scientific methodology and experimental design for neural network analysis.
:::

:::{grid-item-card} 🛠 Development
:link: development/index
:link-type: doc

Contribute to NeuronMap development and extend functionality.
:::

::::

## 📊 Features Overview

```{mermaid}
graph TD
    A[NeuronMap Framework] --> B[Data Generation]
    A --> C[Analysis Engine]
    A --> D[Visualization]
    A --> E[Configuration]
    
    B --> B1[Question Generator]
    B --> B2[Synthetic Data]
    
    C --> C1[Activation Extraction]
    C --> C2[Layer Analysis]
    C --> C3[Statistical Analysis]
    
    D --> D1[Interactive Plots]
    D --> D2[Export Tools]
    D --> D3[Custom Themes]
    
    E --> E1[YAML Config]
    E --> E2[Environment Management]
    E --> E3[Validation]
```

## 🌟 Key Benefits

- **🔧 Production Ready**: Robust, tested, and documented codebase
- **📈 Scalable**: Handles models from small to very large scale
- **🎨 Flexible**: Extensive configuration and customization options
- **🔬 Scientific**: Built for research with reproducibility in mind
- **👥 Community**: Open source with active development and support
