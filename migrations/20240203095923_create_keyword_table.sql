-- Add migration script here
-- 10万件のキーワードを登録する
CREATE TABLE keywords
(
    keyword_id SERIAL PRIMARY KEY,
    keyword_text VARCHAR(50) NOT NULL,
    embedding vector(1536)
);

-- 1000件の学生を登録する
-- keyword_list_embedding が検索対象のキーワードの埋め込みベクトル
CREATE TABLE students
(
    student_id SERIAL PRIMARY KEY,
    student_name VARCHAR(50) NOT NULL,
    keyword_list_embedding vector(1536)
);

CREATE TABLE student_keywords_relations
(
    student_id SERIAL,
    keyword_id SERIAL,
    UNIQUE(student_id, keyword_id),
    foreign key (student_id) references students(student_id) on delete cascade,
    foreign key (keyword_id) references keywords(keyword_id) on delete cascade
);
