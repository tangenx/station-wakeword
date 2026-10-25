# shellcheck shell=sh disable=SC2086,SC2029
# SC2086: $SSH_OPTS is split into ssh arguments on purpose.
# SC2029: remote commands are built on this side on purpose.
#
# Shared helpers for the install/uninstall scripts.
#
# Transport is picked once and reused: ssh by key, ssh by password (sshpass),
# or adb. Override with STATION_TRANSPORT=ssh|adb.
#
# Environment (STATION_IP or STATION is required):
#   STATION_IP     station IP, for adb (and for ssh if STATION is unset)
#   STATION        ssh host, alias from ~/.ssh/config or IP   (default $STATION_IP)
#   SSH_KEY        private key for ssh   (default: ssh's own keys / ~/.ssh/config)
#   SSH_USER       ssh user              (default root)
#   STATION_PASS   ssh password, needs sshpass   (optional)
set -eu

STATION_IP=${STATION_IP:-}
STATION=${STATION:-$STATION_IP}
[ -n "$STATION" ] || { printf '%s\n' "error: set STATION_IP=<station ip> (or STATION=<ssh host>)" >&2; exit 1; }
SSH_USER=${SSH_USER:-root}
SSH_KEY=${SSH_KEY:-}
ADB_TARGET="${STATION_IP:-$STATION}:5555"

SSH_OPTS="-o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"
TRANSPORT=""

log() { printf '%s\n' "$*" >&2; }
die() { log "error: $*"; exit 1; }

# _ssh ARGS... — ssh with the common options and SSH_KEY when given.
_ssh() {
    if [ -n "$SSH_KEY" ]; then
        ssh $SSH_OPTS -i "$SSH_KEY" "$@"
    else
        ssh $SSH_OPTS "$@"
    fi
}

_ssh_key_ok() {
    [ -z "$SSH_KEY" ] || [ -f "$SSH_KEY" ] || return 1
    _ssh -o BatchMode=yes "${SSH_USER}@${STATION}" true 2>/dev/null
}

_ssh_pass_ok() {
    [ -n "${STATION_PASS:-}" ] || return 1
    command -v sshpass >/dev/null 2>&1 || return 1
    sshpass -p "$STATION_PASS" ssh $SSH_OPTS -o PubkeyAuthentication=no \
        "${SSH_USER}@${STATION}" true 2>/dev/null
}

_adb_ok() {
    command -v adb >/dev/null 2>&1 || return 1
    adb connect "$ADB_TARGET" >/dev/null 2>&1
    adb -s "$ADB_TARGET" shell true >/dev/null 2>&1
}

pick_transport() {
    [ -z "$TRANSPORT" ] || return 0
    case "${STATION_TRANSPORT:-auto}" in
        ssh) _ssh_key_ok && TRANSPORT=ssh-key && { log "transport: ssh key"; return 0; }
             _ssh_pass_ok && TRANSPORT=ssh-pass && { log "transport: ssh password"; return 0; }
             die "ssh transport requested but neither key nor password works" ;;
        adb) _adb_ok && TRANSPORT=adb && { log "transport: adb"; return 0; }
             die "adb transport requested but the station is unreachable" ;;
    esac
    if _ssh_key_ok; then TRANSPORT=ssh-key; log "transport: ssh key"
    elif _ssh_pass_ok; then TRANSPORT=ssh-pass; log "transport: ssh password"
    elif _adb_ok; then TRANSPORT=adb; log "transport: adb fallback"
    else die "station unreachable: ssh key, ssh password and adb all failed"; fi
}

# remote_run CMD... — run a command on the station, stdin forwarded.
remote_run() {
    pick_transport
    case "$TRANSPORT" in
        ssh-key)  _ssh "${SSH_USER}@${STATION}" "$@" ;;
        ssh-pass) sshpass -p "$STATION_PASS" ssh $SSH_OPTS -o PubkeyAuthentication=no "${SSH_USER}@${STATION}" "$@" ;;
        adb)      adb -s "$ADB_TARGET" shell "$@" ;;
    esac
}

# remote_put LOCAL REMOTE — copy a file to the station. The station's BusyBox
# has no scp/sftp server, so ssh transfers stream through `cat`.
remote_put() {
    src=$1 dst=$2
    pick_transport
    case "$TRANSPORT" in
        ssh-key)  _ssh "${SSH_USER}@${STATION}" "cat > '$dst'" < "$src" ;;
        ssh-pass) sshpass -p "$STATION_PASS" ssh $SSH_OPTS -o PubkeyAuthentication=no "${SSH_USER}@${STATION}" "cat > '$dst'" < "$src" ;;
        adb)      adb -s "$ADB_TARGET" push "$src" "$dst" >&2 ;;
    esac
}
