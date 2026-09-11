param(
    [string]$StageName = "TKDE_HRCSR_final_release_20260911",
    [string]$ArchiveName = "TKDE_HRCSR_final.zip"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stage = Join-Path $repo $StageName
$archive = Join-Path $repo $ArchiveName

if (Test-Path -LiteralPath $stage) {
    throw "Release staging directory already exists: $stage"
}
if (Test-Path -LiteralPath $archive) {
    throw "Release archive already exists: $archive"
}

New-Item -ItemType Directory -Path $stage | Out-Null

$topFiles = @(
    ".gitattributes", ".gitignore", "CHANGELOG.md",
    "CHECKPOINT_BOUNDARY_RESULTS.csv", "COMPONENT_ABLATION_RESULTS.csv",
    "CONTEMPORARY_BASELINE_AUDIT.md", "EFFICIENCY_RESULTS.csv",
    "EXPERIMENT_MANIFEST.csv", "EXPERT_SCALING.csv",
    "FINAL_RESULT_PROVENANCE.md", "FINAL_TKDE_REVIEW.md",
    "GATE_CONSISTENCY_CHECK.txt", "main.tex", "main.pdf",
    "TKDE_HRCSR_final.tex", "TKDE_HRCSR_final.pdf", "supplement.tex",
    "supplement.pdf", "references.bib", "PROVENANCE.json", "pyproject.toml", "README.md",
    "README_CN.md", "REDDIT_GATE_FORENSIC_AUDIT.md", "requirements.txt",
    "RESULT_CONSISTENCY_REPORT.md", "RESULT_PROVENANCE.md", "RUN_ALL.md",
    "TABLE_VII_CONSISTENCY_CHECK.txt", "TABLE_VII_PROVENANCE_AUDIT.md",
    "TABLE_VII_PROVENANCE.csv", "TKDE_REJECT_RISK_AUDIT.md",
    "VERIFICATION.md"
)
foreach ($name in $topFiles) {
    Copy-Item -LiteralPath (Join-Path $repo $name) -Destination (Join-Path $stage $name)
}

foreach ($directory in @("configs", "experiments", "scripts", "src", "tests")) {
    Copy-Item -LiteralPath (Join-Path $repo $directory) -Destination $stage -Recurse
}

New-Item -ItemType Directory -Path (Join-Path $stage "figures") | Out-Null
foreach ($name in @("fig8_hrcsr_mechanism.pdf", "fig3_pretext_objectives.pdf")) {
    Copy-Item -LiteralPath (Join-Path $repo "figures/$name") -Destination (Join-Path $stage "figures/$name")
}

New-Item -ItemType Directory -Path (Join-Path $stage "paper") | Out-Null
foreach ($name in @("TGFM_paper.tex", "TGFM_paper.pdf")) {
    Copy-Item -LiteralPath (Join-Path $repo "paper/$name") -Destination (Join-Path $stage "paper/$name")
}

New-Item -ItemType Directory -Path (Join-Path $stage "docs") | Out-Null
$paperDocs = @(
    "SOCIALEVO_5SEED_AUDIT.md", "SOCIALEVO_SPARSE_PROTOCOL_AUDIT.md",
    "CONTEMPORARY_BASELINE_COMPATIBILITY.md", "TKDE_FINAL_REVIEW_SIMULATION.md",
    "CHECKPOINT_BOUNDARY_AUDIT.md", "PAPER_CODE_MAPPING.md", "PAPER_CODE_MAPPING.json"
)
foreach ($name in $paperDocs) {
    Copy-Item -LiteralPath (Join-Path $repo "docs/$name") -Destination (Join-Path $stage "docs/$name")
}

New-Item -ItemType Directory -Path (Join-Path $stage "results/aggregated") -Force | Out-Null
$aggregates = @(
    "checkpoint_boundary_matched_v2.json", "component_ablation_mrr.json",
    "COMPONENT_ABLATION_RESULTS.csv", "expert_scaling.json",
    "gate_summary_dense_sparse.json", "matched_gate_sensitivity.csv",
    "matched_gate_sensitivity.json", "reddit_gate_forensic.json",
    "socialevo_5seed_summary.csv", "socialevo_5seed_summary.json",
    "unified_table_from_raw.json"
)
foreach ($name in $aggregates) {
    Copy-Item -LiteralPath (Join-Path $repo "results/aggregated/$name") -Destination (Join-Path $stage "results/aggregated/$name")
}
Copy-Item -LiteralPath (Join-Path $repo "results/socialevo_5seed_audit.csv") -Destination (Join-Path $stage "results/socialevo_5seed_audit.csv")

$rawRoots = @(
    "target-only-matched-v1", "graphmixer-target-only-matched-v1",
    "hrcsr-ablation-matched-v1", "hrcsr-crossdomain-5seed-v1",
    "hrcsr-lastfm-socialevo-5seed-v1", "socialevo-target-only-tgat",
    "socialevo-target-only-graphmixer", "checkpoint-boundary-audit-matched-v2-main",
    "checkpoint-boundary-audit-matched-v2-extra", "component-ablation-matched-v1"
)
$rawSource = Join-Path $repo "results/raw/remote"
foreach ($rootName in $rawRoots) {
    $sourceRoot = Join-Path $rawSource $rootName
    if (-not (Test-Path -LiteralPath $sourceRoot)) { continue }
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File |
        Where-Object { $_.Extension -in @(".json", ".csv", ".txt", ".yaml", ".yml") } |
        ForEach-Object {
            $relative = $_.FullName.Substring($repo.Length + 1)
            $destination = Join-Path $stage $relative
            New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $destination
        }
}
foreach ($name in @("dataset_split_ids.json", "socialevo_remote_artifact_manifest.json")) {
    $source = Join-Path $rawSource $name
    if (Test-Path -LiteralPath $source) {
        $destination = Join-Path $stage "results/raw/remote/$name"
        New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }
}

Get-ChildItem -LiteralPath $stage -Recurse -Directory -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
Get-ChildItem -LiteralPath $stage -Recurse -File |
    Where-Object { $_.Extension -in @(".pyc", ".pyo", ".pth") } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }

$manifestPath = Join-Path $stage "MANIFEST.sha256"
$manifest = Get-ChildItem -LiteralPath $stage -Recurse -File |
    Where-Object { $_.FullName -ne $manifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $relative = $_.FullName.Substring($stage.Length + 1).Replace("\", "/")
        "$hash  $relative"
    }
Set-Content -LiteralPath $manifestPath -Value $manifest -Encoding utf8

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $archive -CompressionLevel Optimal
Write-Output "STAGE=$stage"
Write-Output "ARCHIVE=$archive"
