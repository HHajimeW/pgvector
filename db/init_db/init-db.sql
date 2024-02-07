SELECT 'CREATE DATABASE pgvector_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pgvector_test')\gexec

\c pgvector_test

CREATE EXTENSION vector;