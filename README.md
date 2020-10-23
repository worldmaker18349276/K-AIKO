![](logo.png)

K-AIK▣  is a sound-control one-line terminal-based rhythm game.

```
 ⣿⣴⣧⣰⣄ [  384/ 2240] □   □⛶  □   ■       ■   □   □   ■   ■   □   [ 21.8%] 
```

- dependencies: python3.6, dataclasses, parsy, numpy, scipy, audioread, pyaudio
- used characters: ⛶ 🞎 🞏 🞐 🞑 🞒 🞓 ⬚ □ ■ ⬒ ◎ ◴ ◵ ◶ ◷ ☺ ⟪ ⟨ ⟩ ⟫
- best terminal: GNOME Terminal (set __ambiguous-width characters__ to narrow)
- best font: Ubuntu Mono Regular, 16pt
- best theme: Rxvt


## TODO
- add config system
  - theme: beats (symbols/sounds), target (hit), spectrum/score/progress
  - difficulty: tolerances, incr_tol
  - accessibility: track_width, hit_decay, hit_sustain, show/hide spectrum/score/progress (in beatmap)
                   beat_sounds, show_measures, fixed_speed, custom_skin, use_script (in sheet)
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
  - preload custom beat sound (by AudioMixer) for Sym

- time control
  - pause, resume, skip
  - scroll, Nx
  - audio time sync
  - console pause/resume
  - in-game time control

- KnockGame
  - KnockConfigurator: output/input device, display_delay/knock_delay/knock_energy
  - KnockGame: record/examine, pause/resume, merge
  - KnockConsole =(use)=> KnockGame =(extend)=> Beatmap <=(generate)= BeatSheet

- others
  - design K-AIKO skin format
  - sheet modifiers: no_beat_sounds, show_measures, purify (fixed_speed, no_custom_skin, no_script)
  - add preview_point, volume

- Structure: beatmap > chart > note
  - beatmap: define global properties
    - timing points
    - custom notes
  - chart: define notes stream
    - time ordering
    - different chart has different context
  - note: define how to construct event
    - context-free until constructing events
    - one note, one (or none) event
    - HitObjectNote, EventNote, ContextNote

- K-AIKO format
  - design
    - extensibility
      - as sub-lang of python => hackable
      - abstract event, note design => limited extensibility but become editable
    - editability
      - easy way to convert between pattern and note
      - no recursive structure
    - readability
      - chronological, monophony notes
      - expressive pattern syntax

  - BeatSheet
    - ast of beatsheet format
    - file <=(read/write)=> BeatSheet <=(construct/format)=> Beatmap
    - K-AIKO-std: standard format
    - K-AIKO-ext: extended format
      - enable to import custom Event/HitObject/Note
      - warnings what module are imported
    - K-AIKO-hack: hack format
      - no limit, directly execute it
      - warnings anyway
    - osu

    > #K-AIKO-std-1.0.0
      beatmap.audio = '...'
      beatmap.offset = 2.44
      beatmap.tempo = 140.0
      beatmap += r'''
      (beat=0, length=1, meter=4)
      x x o x | x x [x x] o | x x [x x] x | [x x x x] [_ x] x |
      %(2) ~ ~ ~ | < < < < | < < < < | @(2) ~ ~ ~ |
      '''

    > #K-AIKO-ext-1.0.0
      module = require('...')
      ...
      beatmap['+'] = module.Cross
      ...

  - chart pattern
    - token
      - valid note symbol: `[^\s\(\)\[\]\{\}\'\"\|\#]+`
      - line-based comment: `# ...`
      - arg types: None, bool, int, fraction, float, str
    - chart params: `(beat=0, length=1, meter=4, hide=False)`
      - forward every notes: `beat += length`

    - notes: `_ x o < % @`
      - args: `x(123, l=456, t=789)`
      - nothing: `x()`
      - bad: `x(beat=1.22, length=0.276)` => it cause some timing problem
      - beat as a float => free timing note
      - (!) event's behavior shouldn't change due to the type of beat/length
    - lengthen: `z ~`
      - lengthen previous note: `x.length += length`
    - measure: `|`
      - time signature => determined by chart param `meter`
      - assertion of time alignment: `... x | x x x x | x ...`, `... x | x x ~ ~ | ~ ...`
      - not effected by division bracket: `[ | ]` == `|`

    - division bracket: `[...]`
      - set param `length=length/divisor` in the bracket
      - param: `[(divisor=3) ...]` or `[(3) ...]`, default: `(divisor=2)`
    - instant brace: `{...}` (unconvertable)
      - set param `length=0` in the bracket
      - beat won't change in this mode: `| x {...} x x x |`
      - free timing note: `{x(beat=1.22, length=0.45)}` (convertable)

    - formating chart
      - expressive format design => hard to reformat
      - bad: `_ ~` => `_ _`, `z [~]` => `[z ~ ~]`, `[z] ~` => `[z ~ ~]`
        `{z ~}` => `{z}`, `{z _}` => `{z}`, `{[z]}` => `{z}`
      - equivalent: `[z] [z]` == `[z z]`, `[z ~]` == `z`, `[(m) [(n) ...]]` == `[(m*n) ...]`
        `{z} {z}` == `{z z}`, `[{z}]` == `{z}`
      - lengthless note: `z _` == `z ~`, `{z} _` => `z`
      - free timing note: `{z(beat)} _` == `_ {z(beat)}`
