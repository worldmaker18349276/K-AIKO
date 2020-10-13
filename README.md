![](logo.png)

K-AIK▣  is a sound-control one-line terminal-based rhythm game.

```
 ⣿⣴⣧⣰⣄ [  384/ 2240] □   □⛶  □   ■       ■   □   □   ■   ■   □   [ 21.8%] 
```

- dependencies: python3.6, dataclasses, numpy, scipy, audioread, pyaudio
- used characters: ⛶ 🞎 🞏 🞐 🞑 🞒 🞓 ⬚ □ ■ ⬒ ◎ ◴ ◵ ◶ ◷ ☺ ⟪ ⟨ ⟩ ⟫
- best terminal: GNOME Terminal (set __ambiguous-width characters__ to narrow)
- best font: Ubuntu Mono Regular, 16pt
- best theme: Rxvt


## TODO
- add config system
  - theme: beats (symbols/sounds), target (hit), spectrum/score/progress
  - difficulty: tolerances, incr_tol
  - accessibility: track_width, hit_decay, hit_sustain, show/hide spectrum/score/progress (in beatmap)
                   beat_sounds, show_measures, fixed_speed, custom_theme, use_script (in sheet)
  - controls: prepare_time, skip_time, countdown
  - knock console config:
    - screen: color_palette, size, ...
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
  - target controls: Flip, Move, Jiggle

- time control
  - pause, resume, skip
  - scroll, Nx
  - audio time sync
  - console pause/resume
  - in-game time control

- BeatSheetStd
  - .kaiko, .kaiko-theme file format
  - Note: save, edit
  - modifiers: no_beat_sounds, show_measures, purify (fixed_speed, no_custom_theme, no_script)

- KnockGame
  - KnockConfigurator: output/input device, display_delay/knock_delay/knock_energy
  - KnockGame: record/examine, pause/resume, merge
  - KnockGame -> Beatmap -> BeatmapStd, BeatmapEditor
