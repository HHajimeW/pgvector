-- Add migration script here
DROP TYPE IF EXISTS type_hit_keyword CASCADE;
CREATE TYPE type_hit_keyword AS (
    keyword_id int
    ,keyword_text VARCHAR(50)
    ,distance FLOAT
);

DROP TYPE IF EXISTS type_student_list_output CASCADE;
CREATE TYPE type_student_list_output AS (
    student_id int
    ,student_name VARCHAR(50)
    ,keywords type_hit_keyword[]
);

CREATE OR REPLACE FUNCTION get_similar_student_list(
  query_vector vector(1536)
) RETURNS SETOF type_student_list_output AS $FUNCTION$
DECLARE
    hit_keywords type_hit_keyword[];
begin
	RETURN
		QUERY
    SELECT
        stu.student_id,
        stu.student_name,
        ARRAY_AGG(
            ROW(
                key.keyword_id
                , key.keyword_text
                , key.embedding <-> query_vector
            )::type_hit_keyword
        ) AS hit_keywords
    FROM
        students stu
    LEFT JOIN
        student_keywords_relations skr
    ON
        stu.student_id = skr.student_id
    LEFT JOIN
        keywords key
    ON
        skr.keyword_id = key.keyword_id
    GROUP BY
        stu.student_id
    ORDER BY
        stu.keyword_list_embedding <-> query_vector
    limit 10
  ;
END;
$FUNCTION$ LANGUAGE plpgsql;