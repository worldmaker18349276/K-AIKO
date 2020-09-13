![](logo.png)

K-AIK▣  is a sound-control one-line terminal-based rhythm game.

```
 ⣿⣴⣧⣰⣄ [  384/ 2240] □   □⛶  □   ■       ■   □   □   ■   ■   □   [ 21.8%] 
```

- dependencies: numpy, scipy, pyaudio, audioread
- used characters: ⛶ 🞎 🞏 🞐 🞑 🞒 🞓 ⬚ □ ■ ⬒ ◎ ◴ ◵ ◶ ◷ ☺ ⟪ ⟨ ⟩ ⟫
- best terminal: GNOME Terminal (set __ambiguous-width characters__ to narrow)
- best font: Ubuntu Mono Regular, 16pt
- best theme: Rxvt


## TODO
- add config system
  - theme: beats (symbols/sounds), target (hit), spectrum/score/progress
  - difficulty: tolerances, incr_tol
  - accessibility: track_width, hit_decay, hit_sustain, beat_sounds, pure_mode, show_barlines
  - controls: prepare_time, skip_time, countdown
  - knock console config:
    - sample: samplerate, hop_length, win_length
    - peak: pre_max, post_max, pre_avg, post_avg, wait, delta
    - controls: display_fps, display_delay, knock_volume, knock_delay, music_volume

- add menu
  - score, error report
  - select beatmaps
  - config
  - adjust audio/display delay

- add terminal command
  - delay
  - output to report
  - add record, re-examine method

- add combo bonus and visual response
  - use bold score number
  - add hint sound (?)

- add Script
  - Sym(symbol, sound)
  - target controls: Flip, Move
  - pure mode (fixed speed, no custom theme, no script) => BeatmapStdSheet.purify()

- add time control (pause/resume/skip; scroll/Nx)
  - console pause/resume
  - in-game time control

- .ka
  - parse
  - convert from .osu
  - .ka-theme

- KnockGame
  - KnockAdjuster: display_delay, knock_delay, knock_volume
  - KnockGame.record, KnockGame.examine
  - KnockGame -> Beatmap -> BeatmapStd, BeatmapEditor
  - make KnockGame mergeable
  - KAIKO(Beatmap, BeatmapReport, BeatmapMenu)
