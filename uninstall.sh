#!/bin/sh
# Stop wakeword and remove its boot script.
#
#   ./uninstall.sh            # keep trained words in /data/wakeword
#   ./uninstall.sh --purge    # also delete /data/wakeword
set -eu
here=$(CDPATH='' cd "$(dirname "$0")" && pwd)
. "$here/lib.sh"

PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

remote_run "set -e
if [ -x /etc/init.d/S97wakeword ]; then
    /etc/init.d/S97wakeword stop
    mount -o remount,rw /
    rm -f /etc/init.d/S97wakeword
    sync
    mount -o remount,ro / || true
fi
[ $PURGE = 1 ] && rm -rf /data/wakeword || true"
log "wakeword removed$( [ $PURGE = 1 ] && echo ', words deleted' || echo '; words kept in /data/wakeword')"
