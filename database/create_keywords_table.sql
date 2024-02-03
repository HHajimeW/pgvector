CREATE DATABASE pgvector_test;

\c pgvector_test

CREATE EXTENSION vector;

CREATE TABLE keywords
(
    id BIGSERIAL PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL
);

CREATE TABLE keywords_vector
(
    id BIGSERIAL PRIMARY KEY,
    embedding vector(1536)
);
