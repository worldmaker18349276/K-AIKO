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
  - ui: drop_speed, hit_decay, hit_sustain, beat_sounds, pure_mode
  - controls: prepare_time, skip_time
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
  - pure mode (fixed speed, no custom theme, no script)

- add time control (pause/resume/skip; scroll/Nx)
  - console pause/resume
  - in-game time control

- .ka
  - subset of python
  - multi-layers
  - substitution-based
  - define layer by (time, bpm, pattern)
  - define pattern
  - convert from .osu
  - .ka-theme

- KnockGame
  - KnockGame -> Beatmap -> BeatmapStd, BeatmapEditor, BeatmapAdjuster
  - make KnockGame mergeable
  - KAIKO(Beatmap, BeatmapReport, BeatmapMenu)
  - BeatBuilder read "sheet.ka" and generate BeatmapStd
  - define custom beatmap as the same form of "sheet.ka" => "sheet.ka" is subset of python
