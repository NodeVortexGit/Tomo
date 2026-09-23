//! Ask Tomo's brain one thing from the terminal, without the app — handy for
//! checking the API key, the model and tools like look_at_screen:
//!
//!     cargo run -p tomo-core --example ask -- "what's on my screen?"
//!
//! Commands are never run from here (the executor is log-only) and nothing is
//! saved to Tomo's memory.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

use tomo_core::ai::AiClient;
use tomo_core::apps::SystemCatalog;
use tomo_core::commands::Executor;
use tomo_core::db::Db;
use tomo_core::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tomo_core::init_tracing();
    let question = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    anyhow::ensure!(!question.is_empty(), "usage: ask <question>");

    let cfg = Config::load(&std::env::current_dir()?)?;
    let executor = Executor::new(false, std::env::temp_dir().join("tomo-ask-audit.log"), vec![]);
    let ai = AiClient::new(
        cfg,
        Db::open_in_memory()?,
        executor,
        Arc::new(RwLock::new(SystemCatalog::default())),
        Arc::new(AtomicBool::new(false)),
    );

    let (ui, mut body) = tokio::sync::mpsc::unbounded_channel();
    let reply = ai.respond(&question, &ui).await?;
    while let Ok(event) = body.try_recv() {
        println!("[body] {event:?}");
    }
    println!("{reply}");
    Ok(())
}
