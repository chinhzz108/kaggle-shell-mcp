//! End-to-end stdio MCP protocol verification & CLI smoke tests (runs completely offline).

use std::collections::HashSet;
use std::process::Command as StdCommand;

use rmcp::ServiceExt;
use rmcp::transport::TokioChildProcess;
use tokio::process::Command;

const EXPECTED_TOOLS: &[&str] = &[
    "kaggle_up",
    "kaggle_status",
    "kaggle_down",
    "term_exec",
    "job_spawn",
    "job_list",
    "job_logs",
    "job_kill",
    "fs_sync",
];

#[tokio::test]
async fn mcp_stdio_registers_all_nine_tools() {
    let bin = env!("CARGO_BIN_EXE_kgsh");
    let cmd = Command::new(bin);
    let transport = TokioChildProcess::new(cmd).expect("spawn kgsh MCP stdio server");

    let client = ().serve(transport).await.expect("initialize MCP session");
    let result = client
        .list_tools(Default::default())
        .await
        .expect("list_tools RPC");

    let actual_names: HashSet<String> = result.tools.iter().map(|t| t.name.to_string()).collect();
    let expected_names: HashSet<String> = EXPECTED_TOOLS.iter().map(|s| (*s).to_string()).collect();

    assert_eq!(actual_names, expected_names);
    for tool in &result.tools {
        let desc = tool.description.as_deref().unwrap_or("");
        assert!(
            desc.len() > 15,
            "Tool '{}' has overly short description: '{desc}'",
            tool.name
        );
    }

    let _ = client.cancel().await;
}

#[test]
fn cli_version_and_help_work() {
    let bin = env!("CARGO_BIN_EXE_kgsh");

    let ver = StdCommand::new(bin).arg("--version").output().unwrap();
    assert!(ver.status.success());
    let ver_out = String::from_utf8_lossy(&ver.stdout);
    assert!(
        ver_out.contains("kgsh"),
        "unexpected --version output: {ver_out}"
    );

    let help = StdCommand::new(bin).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help_out = String::from_utf8_lossy(&help.stdout);
    for subcmd in ["mcp", "up", "status", "down", "exec", "spawn", "jobs", "logs", "kill", "push", "pull"] {
        assert!(
            help_out.contains(subcmd),
            "missing subcommand '{subcmd}' in --help:\n{help_out}"
        );
    }
}
