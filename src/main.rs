use anyhow::Context;
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, Embedding},
    Client as OpenAIClient,
};
use csv::Reader;
use pgvector::Vector;
use postgres::NoTls;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::File;
use tokio_postgres::Client;

#[derive(Debug, Deserialize)]
struct RawKeyword {
    keyword_id: i32,
    keyword_text: String,
}

#[derive(Debug, Deserialize)]
struct RawStudent {
    student_id: i32,
}

#[derive(Debug, Deserialize)]
struct RawRelation {
    student_id: i32,
    keyword_id: i32,
}

#[derive(Debug, FromSql, ToSql, Serialize, Deserialize)]
#[postgres(name = "type_hit_keyword")]
pub struct HitKeyword {
    pub keyword_id: i32,
    pub keyword_text: String,
    pub distance: f64,
}

#[derive(Debug, FromSql, ToSql, Serialize, Deserialize)]
#[postgres(name = "type_student_list_output")]
pub struct StudentListOutput {
    pub student_id: i32,
    pub student_name: String,
    pub keywords: Vec<HitKeyword>,
}

async fn insert_keyword_data(
    client: &Client,
    mut keyword_file: Reader<File>,
) -> anyhow::Result<()> {
    // CSVから読み込んだデータをデータベースに挿入
    for result in keyword_file.deserialize() {
        let record: RawKeyword = result?;
        client
            .execute(
                "insert into keywords (keyword_id, keyword_text) values ($1, $2)",
                &[&record.keyword_id, &record.keyword_text],
            )
            .await?;
    }
    Ok(())
}

async fn insert_student_data(
    client: &Client,
    mut student_file: Reader<File>,
) -> anyhow::Result<()> {
    // CSVから読み込んだデータをデータベースに挿入
    for result in student_file.deserialize() {
        let record: RawStudent = result?;
        let name = format!("田中{} 太郎{}", &record.student_id, &record.student_id);
        client
            .execute(
                "insert into students (student_id, student_name) values ($1, $2)",
                &[&record.student_id, &name],
            )
            .await?;
    }
    Ok(())
}

async fn insert_relation_data(
    client: &Client,
    mut relation_file: Reader<File>,
) -> anyhow::Result<()> {
    // CSVから読み込んだデータをデータベースに挿入
    for result in relation_file.deserialize() {
        let record: RawRelation = result?;
        client
            .execute(
                "insert into student_keywords_relations (student_id, keyword_id) values ($1, $2)",
                &[&record.student_id, &record.keyword_id],
            )
            .await?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // データベースに接続設定
    let (client, connection) = tokio_postgres::connect(
        "host=localhost user=testuser password=testpassword dbname=pgvector_test port=5433",
        NoTls,
    )
    .await
    .unwrap_or_else(|_| panic!("Cannot connect to the database"));

    // コネクションを管理するための別のタスクを起動
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });

    // keyword_dataをインサート
    let keyword_file = csv::Reader::from_path("./data/test/keywords.csv").unwrap();
    let _keyword_insert_result = insert_keyword_data(&client, keyword_file).await?;

    // 学生データをインサート
    let student_file = csv::Reader::from_path("./data/test/students.csv").unwrap();
    let _student_insert_result = insert_student_data(&client, student_file).await?;

    // 学生キーワードリレーションデータをインサート
    let relation_file = csv::Reader::from_path("./data/test/relations.csv").unwrap();
    let _relation_insert_result = insert_relation_data(&client, relation_file).await?;

    // キーワードのEmbeddingを取得し、データベースに挿入する
    let _keyword_embedding_insert_result = insert_keyword_embeddings(&client).await?;

    // 学生のキーワードリストのEmbeddingを取得し、データベースに挿入する
    let _student_embedding_insert_result = insert_student_embeddings(&client).await?;

    // 類似キーワードを検索して、学生を表示し、その中の類似キーワードTOP3を表示する

    //　類義語を検索したいキーワードの設定
    let query = vec!["機械学習 自然言語処理".to_string()];

    //　Embeddingを取得する
    let query_embeddings = get_embeddings(query.clone())
        .await
        .context("embedding fetch failed")?;
    let query_vector = Vector::from(query_embeddings[0].embedding.clone());

    let sql = r#"
        select
            get_similar_student_list($1)::type_student_list_output
    "#;

    // 類似キーワードを検索する
    let result: Vec<StudentListOutput> = client
        .query(sql, &[&query_vector])
        .await?
        .iter()
        .map(|row| row.get(0))
        .collect();

    println!();
    println!("検索クエリ: {:?}", query.clone());
    println!();

    for mut student_output in result {
        println!("Student ID: {}", student_output.student_id);
        println!("Student Name: {}", student_output.student_name);
        println!("Keywords:");

        // distance でキーワードをソートする
        student_output
            .keywords
            .sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        for keyword in student_output.keywords {
            println!(
                "\tID: {}, Text: {}, Distance: {:.6}",
                keyword.keyword_id, keyword.keyword_text, keyword.distance
            );
        }
        println!("-----------------------------------");
    }

    Ok(())
}

pub async fn insert_keyword_embeddings(client: &Client) -> anyhow::Result<()> {
    // データベースからデータを取り出して、Keywordに入れる
    let mut ids: Vec<i32> = Vec::new();
    let mut keywords: Vec<String> = Vec::new();
    let rows = client
        .query("select * from keywords where embedding is null", &[])
        .await?;

    for row in rows {
        let id: i32 = row.get(0);
        let text: String = row.get(1);
        ids.push(id);
        keywords.push(text.clone());
    }

    // 1000データずつOpenAIに投げる
    for (ids_chunk, keywords_chunk) in ids.chunks(1000).zip(keywords.chunks(1000)) {
        println!("Processing ids: {:?}", ids_chunk);

        let embeddings = get_embeddings(keywords_chunk.to_vec())
            .await
            .context("embedding fetch failed")?;

        for (id, embedding) in ids_chunk.iter().zip(embeddings) {
            let embedding_vector = Vector::from(embedding.embedding.clone());
            client
                .execute(
                    "
                update
                    keywords
                set
                    embedding = $1
                where
                    keyword_id = $2",
                    &[&embedding_vector, &id],
                )
                .await?;
        }
    }

    Ok(())
}

pub async fn insert_student_embeddings(client: &Client) -> anyhow::Result<()> {
    // データベースからデータを取り出して、Keywordに入れる
    let mut ids: Vec<i32> = Vec::new();
    let mut keywords_list: Vec<String> = Vec::new();
    let rows = client
        .query(
            "
        with tmp as (
            select
                stu.student_id,
                array_to_string(ARRAY(SELECT unnest(array_agg(key.keyword_text))), ' ') as keywords
            from
                students stu
            left join
                student_keywords_relations skr
            on
                stu.student_id = skr.student_id
            left join
                keywords as key
            on
                skr.keyword_id  = key.keyword_id
            group by
                stu.student_id
        )
        select * from tmp where keywords is not null and keywords != ''
    ",
            &[],
        )
        .await?;

    for row in rows {
        let id: i32 = row.get(0);
        let keywords: String = row.get(1);
        ids.push(id);
        keywords_list.push(keywords.clone());
    }

    // 1000データずつOpenAIに投げる
    for (ids_chunk, keywords_list_chunk) in ids.chunks(1000).zip(keywords_list.chunks(1000)) {
        println!("Processing ids: {:?}", ids_chunk);

        let embeddings = get_embeddings(keywords_list_chunk.to_vec())
            .await
            .context("embedding fetch failed")?;

        for (id, embedding) in ids_chunk.iter().zip(embeddings) {
            let embedding_vector = Vector::from(embedding.embedding.clone());
            client
                .execute(
                    "
                update
                    students
                set
                    keyword_list_embedding = $1
                where
                    student_id = $2",
                    &[&embedding_vector, &id],
                )
                .await?;
        }
    }

    Ok(())
}

pub async fn get_embeddings(texts: Vec<String>) -> anyhow::Result<Vec<Embedding>> {
    let openai_api_key = env::var("OPENAI_API_KEY").unwrap();
    let config = OpenAIConfig::new().with_api_key(openai_api_key);
    let client = OpenAIClient::with_config(config);

    let request = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-ada-002")
        .input(texts)
        .build()
        .context("failed to build openai embedding request")?;

    let res = client
        .embeddings()
        .create(request)
        .await
        .context("failed embedding request")?;

    Ok(res.data)
}
