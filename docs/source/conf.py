# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IterativeClustering-Framework'
copyright = '2026, Aldy & Azarya'
author = 'Aldy & Azarya'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Mengambil docstring dari kode
    'sphinx.ext.napoleon',     # Mendukung format Google/NumPy docstrings
    'sphinx.ext.viewcode',     # Menampilkan link ke source code asli
    'sphinx.ext.mathjax',      # Merender persamaan LaTeX (Penting untuk TA Anda!)
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
