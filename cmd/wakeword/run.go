package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"wakeword/internal/daemon"
	"wakeword/internal/quasar"
)

func cmdRun(args []string) error {
	exe, _ := os.Executable()
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	var opt daemon.Options
	fs.StringVar(&opt.DataDir, "data", "/data/wakeword", "words and settings")
	fs.StringVar(&opt.Negatives, "negatives", filepath.Join(filepath.Dir(exe), "negatives.bin"), "bundled negative set")
	fs.StringVar(&opt.QuasarCfg, "quasar-cfg", "/system/vendor/quasar/quasar.cfg", "service ports")
	fs.StringVar(&opt.UserConfig, "user-config", quasar.UserConfigPath, "synced config with the station's name")
	fs.StringVar(&opt.Listen, "listen", ":8780", "web UI address")
	fs.Parse(args)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	d, err := daemon.New(opt)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := &http.Server{Addr: opt.Listen, Handler: d.Handler(), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		srv.Shutdown(sctx)
	}()
	go d.Run(ctx)
	log.Printf("wakeword: UI on %s, data in %s", opt.Listen, opt.DataDir)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
