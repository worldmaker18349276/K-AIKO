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
  - accessibility: track_width, hit_decay, hit_sustain, show/hide spectrum/score/progress (in beatmap)
                   beat_sounds, show_barlines, fixed_speed, custom_theme, use_script (in sheet)
  - controls: prepare_time, skip_time, countdown
  - knock console config:
    - screen: color_palette, ...
    - config device (device_index, is_format_supported)

- add menu
  - score, error report
  - select beatmaps
  - config
  - adjust audio/display delay

- combo
  - normalize score
  - add combo bonus and visual response
  - use bold score number for combo
  - add hint sound (?)

- Script
  - Sym(symbol, sound)
  - target controls: Flip, Move, Jiggle

- time control
  - pause, resume, skip
  - scroll, Nx
  - audio time sync
  - console pause/resume
  - in-game time control

- BeatmapStdSheet
  - .ka, .ka-theme file format
  - parse, load, save
  - convert from .osu
  - modifiers: no_beat_sounds, show_barlines, purify (fixed_speed, no_custom_theme, no_script)

- KnockGame
  - KnockConfigurator: output/input device, display_delay/knock_delay/knock_energy
  - KnockGame: record/examine, pause/resume, merge
  - KnockGame -> Beatmap -> BeatmapStd, BeatmapEditor
