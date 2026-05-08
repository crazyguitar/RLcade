#!/usr/bin/env bash
# Upload RLcade model artifacts to a Hugging Face model repo.
#
# Requires the Hugging Face CLI:
#   pip install -U huggingface_hub
#   hf auth login

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/hf.sh --repo-id USER_OR_ORG/REPO --agent AGENT [options]

Required:
  --repo-id ID           Hugging Face model repo, e.g. alice/rlcade-rainbow-dqn-smb
  --agent NAME           RLcade agent: ppo, lstm_ppo, dqn, rainbow_dqn, or sac

Artifacts:
  --safetensors PATH     .safetensors model weights, uploaded as model.safetensors
  --checkpoint PATH      .pt checkpoint path
  --include-pt           Upload --checkpoint as checkpoint.pt

Metadata:
  --env ID               Env id (default: rlcade/SuperMarioBros-v0)
  --rom-name NAME        ROM filename for docs only (default: super-mario-bros.nes)
  --actions NAME         Action space: right, simple, or complex
  --world N              SMB world
  --stage N              SMB stage
  --encoder NAME         Encoder name, e.g. cnn or resnet
  --qnet NAME            Q-network name for DQN/Rainbow DQN

Upload:
  --private              Create repo as private if it does not exist
  --revision REV         Target revision (default: main)
  --commit-message MSG   Commit message
  --dry-run              Stage files and print hf commands without uploading
  --keep-staging DIR     Copy staged files to DIR before cleanup
  -h, --help             Show this help

Examples:
  tools/hf.sh --repo-id alice/rlcade-rainbow-dqn-smb --agent rainbow_dqn \
    --safetensors checkpoints/rainbow_dqn_smb.safetensors \
    --checkpoint checkpoints/rainbow_dqn_smb.pt --include-pt \
    --actions right --world 1 --stage 1 --encoder cnn --qnet rainbow_qnet
EOF
}

repo_id=""
agent=""
checkpoint=""
safetensors=""
include_pt=false
env_id="rlcade/SuperMarioBros-v0"
rom_name="super-mario-bros.nes"
actions=""
world=""
stage=""
encoder=""
qnet=""
private=false
revision="main"
commit_message=""
dry_run=false
keep_staging=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --repo-id) repo_id="$2"; shift 2 ;;
  --agent) agent="$2"; shift 2 ;;
  --checkpoint) checkpoint="$2"; shift 2 ;;
  --safetensors) safetensors="$2"; shift 2 ;;
  --include-pt) include_pt=true; shift ;;
  --env) env_id="$2"; shift 2 ;;
  --rom-name) rom_name="$2"; shift 2 ;;
  --actions) actions="$2"; shift 2 ;;
  --world) world="$2"; shift 2 ;;
  --stage) stage="$2"; shift 2 ;;
  --encoder) encoder="$2"; shift 2 ;;
  --qnet) qnet="$2"; shift 2 ;;
  --private) private=true; shift ;;
  --revision) revision="$2"; shift 2 ;;
  --commit-message) commit_message="$2"; shift 2 ;;
  --dry-run) dry_run=true; shift ;;
  --keep-staging) keep_staging="$2"; shift 2 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$repo_id" || -z "$agent" ]]; then
  echo "--repo-id and --agent are required" >&2
  usage >&2
  exit 2
fi

case "$agent" in
ppo|lstm_ppo|dqn|rainbow_dqn|sac) ;;
*) echo "Unsupported --agent: $agent" >&2; exit 2 ;;
esac

if [[ -z "$safetensors" && -n "$checkpoint" ]]; then
  inferred="${checkpoint%.*}.safetensors"
  [[ -f "$inferred" ]] && safetensors="$inferred"
fi

if [[ -z "$safetensors" && "$include_pt" != true ]]; then
  echo "Provide --safetensors, or provide --checkpoint with --include-pt" >&2
  exit 2
fi
if [[ -n "$safetensors" && ! -f "$safetensors" ]]; then
  echo "safetensors file not found: $safetensors" >&2
  exit 2
fi
if [[ "$include_pt" == true && ( -z "$checkpoint" || ! -f "$checkpoint" ) ]]; then
  echo "--include-pt requires an existing --checkpoint file" >&2
  exit 2
fi
if [[ "$dry_run" != true ]] && ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install huggingface_hub or use --dry-run." >&2
  exit 2
fi

commit_message="${commit_message:-Upload RLcade ${agent} artifacts}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/rlcade-hf.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

if [[ -n "$safetensors" ]]; then
  cp "$safetensors" "$tmp/model.safetensors"
fi
if [[ "$include_pt" == true ]]; then
  cp "$checkpoint" "$tmp/checkpoint.pt"
fi

cat >"$tmp/rlcade_config.json" <<EOF
{
  "format": "rlcade-hf-artifacts-v1",
  "agent": "$agent",
  "env": "$env_id",
  "rom_name": "$rom_name",
  "actions": "$actions",
  "world": "$world",
  "stage": "$stage",
  "encoder": "$encoder",
  "qnet": "$qnet",
  "files": {
    "safetensors": "$([[ -n "$safetensors" ]] && echo model.safetensors)",
    "checkpoint": "$([[ "$include_pt" == true ]] && echo checkpoint.pt)"
  }
}
EOF

load_file="model.safetensors"
[[ -z "$safetensors" ]] && load_file="checkpoint.pt"
stage_scope="full game"
if [[ -n "$world" && -n "$stage" ]]; then
  stage_scope="world ${world}, stage ${stage}"
elif [[ -n "$world" ]]; then
  stage_scope="world ${world}"
fi

cat >"$tmp/README.md" <<EOF
---
library_name: pytorch
tags:
- reinforcement-learning
- super-mario-bros
- rlcade
---

# $repo_id

RLcade checkpoint artifacts for a Super Mario Bros agent.

## Training Setup

- Agent: \`$agent\`
- Environment: \`$env_id\`
- Stage scope: \`$stage_scope\`
- Action space: \`${actions:-unspecified}\`
- Encoder: \`${encoder:-unspecified}\`
- Q-network: \`${qnet:-unspecified}\`

## Files

EOF

if [[ -n "$safetensors" ]]; then
  echo "- \`model.safetensors\`: model weights for inference/loading." >>"$tmp/README.md"
fi
if [[ "$include_pt" == true ]]; then
  echo "- \`checkpoint.pt\`: full RLcade training checkpoint, including optimizer state when saved." >>"$tmp/README.md"
fi
cat >>"$tmp/README.md" <<EOF
- \`rlcade_config.json\`: metadata for reconstructing the RLcade command.

The ROM is not included. Provide your own compatible ROM locally.

## Local Usage

\`\`\`bash
python -m rlcade.agent \\
  --agent $agent \\
  --env $env_id \\
  --rom /path/to/$rom_name \\
  --checkpoint $load_file${actions:+ \\}
${actions:+  --actions $actions}${world:+ \\}
${world:+  --world $world}${stage:+ \\}
${stage:+  --stage $stage}
\`\`\`
EOF

echo "Staged files in $tmp:"
find "$tmp" -maxdepth 1 -type f -print | sort | sed 's#^.*/#  #'

if [[ -n "$keep_staging" ]]; then
  rm -rf "$keep_staging"
  mkdir -p "$keep_staging"
  cp "$tmp"/* "$keep_staging"/
  echo "Copied staging directory to $keep_staging"
fi

create_cmd=(hf repos create "$repo_id" --repo-type model --exist-ok)
[[ "$private" == true ]] && create_cmd+=(--private)
upload_cmd=(hf upload "$repo_id" "$tmp" . --repo-type model --revision "$revision" --commit-message "$commit_message")

printf '+ '
printf '%q ' "${create_cmd[@]}"
echo
[[ "$dry_run" == true ]] || "${create_cmd[@]}"

printf '+ '
printf '%q ' "${upload_cmd[@]}"
echo
[[ "$dry_run" == true ]] || "${upload_cmd[@]}"
