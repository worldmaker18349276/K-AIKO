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
    - device: device_index, samplerate, sample_width, nchannel, buffer_length
    - detector: time_res, freq_res, pre_max, post_max, pre_avg, post_avg, wait, delta
    - screen: color_palette, ...
    - controls: display_fps, display_delay, knock_energy, knock_delay

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

- add time control (pause/resume/skip; scroll/Nx)
  - console pause/resume
  - in-game time control

- audio
  - ra.resample
  - remove samplerate, hop_length in beatmap, add nchannel
  - audio nodes merger
  - time sync
  - device_index, is_format_supported

- BeatmapStdSheet
  - .ka, .ka-theme file format
  - parse, load, save
  - convert from .osu
  - modifiers: no_beat_sounds, show_barlines, purify (fixed_speed, no_custom_theme, no_script)

- KnockGame
  - KnockConfigurator: output/input device, display_delay/knock_delay/knock_volume
  - KnockGame: record/examine, pause/resume, merge
  - KnockGame -> Beatmap -> BeatmapStd, BeatmapEditor
