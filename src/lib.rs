//! `kaggle-shell-mcp` (`kgsh`) — Turn free 12h Kaggle Dual-T4 GPUs into live,
//! stateful remote terminals & detached job runners for AI Agents via MCP.

pub mod client;
pub mod config;
pub mod kaggle_api;
pub mod launcher;
pub mod mcp_server;
pub mod sync;
pub mod utils;
