use anyhow::Context;
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, Embedding},
    Client as OpenAIClient,
};
use pgvector::Vector;
use postgres::NoTls;
use serde::Deserialize;
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // csv読み込み
    let mut rdr = csv::Reader::from_path("./src/data/test.csv").unwrap();

    // データベース接続設定
    let (client, connection) = tokio_postgres::connect(
        "host=localhost user=postgres password=postgres dbname=pgvector_test port=5433",
        NoTls,
    )
    .await?;

    // コネクションを管理するための別のタスクを起動
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });

    // CSVから読み込んだデータをデータベースに挿入
    for result in rdr.deserialize() {
        let record: Keyword = result?;
        client
            .execute(
                "INSERT INTO keywords (id, keyword) VALUES ($1, $2)",
                &[&record.id, &record.tag],
            )
            .await?;
    }

    // データベースからデータを取り出して、Keywordに入れる
    let mut ids: Vec<i64> = Vec::new();
    let mut keywords: Vec<String> = Vec::new();
    let rows = client.query("SELECT * FROM keywords", &[]).await?;

    for row in rows {
        let id: i64 = row.get(0);
        let tag: String = row.get(1);
        ids.push(id);
        keywords.push(tag.clone());
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
                    "INSERT INTO keywords_vector (id, embedding) VALUES ($1, $2)",
                    &[&id, &embedding_vector],
                )
                .await?;
        }
    }

    //　類義語を検索したいキーワードの設定
    let query = vec!["言語モデル".to_string()];

    //　Embeddingを取得する
    let query_embeddings = get_embeddings(query)
        .await
        .context("embedding fetch failed")?;
    let query_vector = Vector::from(query_embeddings[0].embedding.clone());

    // 類似キーワードを検索する
    let embedding_rows = client
        .query(
            "SELECT * FROM keywords_vector ORDER BY embedding <-> $1 LIMIT 10",
            &[&query_vector],
        )
        .await?;

    for embedding in embedding_rows {
        let id: i64 = embedding.get(0);

        // id からキーワードを取得する
        let keyword = client
            .query_one("SELECT * FROM keywords WHERE id = $1", &[&id])
            .await?;

        let name: String = keyword.get(1);
        // let distance: f32 = embedding.get(2);
        println!("id: {}, name: {}", id, name);
    }

    Ok(())
}

pub async fn get_embeddings(kys: Vec<String>) -> anyhow::Result<Vec<Embedding>> {
    let openai_api_key = env::var("OPENAI_API_KEY").unwrap();
    let config = OpenAIConfig::new().with_api_key(openai_api_key);
    let client = OpenAIClient::with_config(config);

    let request = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-ada-002")
        .input(kys)
        .build()
        .context("failed to build openai embedding request")?;

    let res = client
        .embeddings()
        .create(request)
        .await
        .context("failed embedding request")?;

    Ok(res.data)
}

#[derive(Debug, Deserialize)]
struct Keyword {
    id: i64,
    tag: String,
}
