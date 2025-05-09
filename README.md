# Pure Graph of Thoughts
This is the official implementation of Pure Graph of Thoughts, a new implementation of the Graph of Thoughts concepts.

Graph of Thoughts was originally introduced in the paper
[Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687.pdf).
The official implementation of the paper's proposed API can be found here: https://github.com/spcl/graph-of-thoughts.

The `pure_graph_of_thoughts` package does not conform the API proposed by the original paper nor is it a fork of it.
It aims for a more automation-friendly implementation of the general concept of Graph of Thoughts,
where both construction and traversal of a graph can be handled iteratively.

Some key differences and restrictions:
- Operations and thoughts are represented independently of their graph structure.
- As a user-facing API, operations can be defined in a declarative way over a typed and validated data structure (DSL).
- There is a strict distinction between a prompt operation executed by a language model and a code execution operation.
- To simplify parsing logic and to ensure consistent results, the JSON format is used for communication with the language model.
- The scoring is now part of an operation involving a prompt, rather than being a standalone operation that can be added arbitrarily.
  While this simplifies the automation process, it restricts the user's possibility of adding a validation operation.

## Setup Guide
To run this code, Python `3.11` or newer is required.
The latest version of the package can be installed from [PyPI](https://pypi.org/project/pure-graph-of-thoughts/):
```shell
pip install pure-graph-of-thoughts
```
Alternatively, the package can be installed from source.

### Optional Dependencies
The project comes with optional dependencies which are required for some features.

#### Graph Visualization
To visualize the graphs by using `pure_graph_of_thoughts.visualization`, 
the optional `visualization` dependencies are required.
```shell
pip install pure-graph-of-thoughts[visualization]
```
For a cleaner, hierarchical visualization, add `dot-visualization`.
```shell
pip install pure-graph-of-thoughts[visualization,dot-visualization]
```
Be aware that `dot-visualization` requires the [GraphViz](https://graphviz.org/) library to be installed.

##### pygraphviz on Windows
The Python package `pygraphviz` (included in `dot-visualization`) may cause installation problems on Windows.
Installing it separately with the following options (including the GraphViz headers) resolves the problem:
```shell
python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz
```

#### Examples
An example task definition is provided together with a notebook 
that shows the graph construction and execution.
To run the notebook, the optional `notebooks` dependencies are required.
```shell
pip install pure-graph-of-thoughts[notebooks]
```
Be aware that the examples are not part of the distributed package on PyPI.

