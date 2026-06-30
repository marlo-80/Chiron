"""
schema.py – Central column schema for the article database.

This module defines the exact column names and their order as they appear
in the SQLite table `articles`.  Every script that writes to or reads from
the database imports `ARTICLE_COLUMNS` from here to guarantee consistency.
"""

ARTICLE_COLUMNS = [
    'filename', 'pmcid', 'doi', 'journal', 'year', 'authors',
    'title', 'keywords', 'abstract',
    'introduction', 'methods', 'results', 'discussion', 'conclusion',
    'tables', 'metadata_junk'
]