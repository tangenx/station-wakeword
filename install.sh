#!/bin/sh
# Install wakeword on a station and start it at every boot.
#
#   ./install.sh
#
# Copies the binary and the negative set to /data/wakeword, installs
# /etc/init.d/S97wakeword (rootfs is remounted rw for that one file) and
# starts the daemon. Nothing in the firmware or quasar.cfg is changed.
# Transport: ssh key -> ssh password -> adb (see lib.sh).
set -eu
here=$(CDPATH='' cd "$(dirname "$0")" && pwd)
. "$here/lib.sh"

DIR=/data/wakeword
for f in dist/wakeword dist/negatives.bin station/S97wakeword; do
    [ -f "$here/$f" ] || die "missing $f (run ./build.sh)"
done

log "== copying files to $DIR =="
remote_run "mkdir -p $DIR"
remote_put "$here/dist/wakeword" "$DIR/wakeword.new"
remote_put "$here/dist/negatives.bin" "$DIR/negatives.bin"
remote_put "$here/station/S97wakeword" "$DIR/S97wakeword"

log "== installing the boot script and starting =="
remote_run "set -e
chmod 755 $DIR/wakeword.new $DIR/S97wakeword
[ -x /etc/init.d/S97wakeword ] && /etc/init.d/S97wakeword stop >/dev/null 2>&1 || true
mv $DIR/wakeword.new $DIR/wakeword
if ! cmp -s $DIR/S97wakeword /etc/init.d/S97wakeword; then
    mount -o remount,rw /
    cp $DIR/S97wakeword /etc/init.d/S97wakeword.new
    mv /etc/init.d/S97wakeword.new /etc/init.d/S97wakeword
    sync
    mount -o remount,ro / || echo 'warning: remount ro failed; it will be ro after reboot'
fi
/etc/init.d/S97wakeword start"

ip=$(remote_run "ip -4 addr show wlan0 2>/dev/null | sed -n 's/.*inet \([0-9.]*\).*/\1/p'" | tr -d '\r')
log "done. Open http://${ip:-<station-ip>}:8780 to add a word."
log "remove with ./uninstall.sh"
