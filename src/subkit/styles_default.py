"""Default style scheme (written by ``subkit init``).

Naming convention — the tool derives roles from names, no mapping lists:
  CN, JP            bottom dialogue (CN above JP)
  CN.Top, JP.Top    dialogue at the top of the screen (when signs occupy the bottom)
  CN.Alt, JP.Alt    a second simultaneous speaker
  OP.CN/OP.JP, ED.CN/ED.JP, IN.CN/IN.JP   songs
  <base>.Ruby       furigana of <base>      (missing ones are derived automatically)
  <base>.Pinyin     pinyin of a CN style
  <base>.Gloss      glosses (旁注) placed above words of <base>
  @读音 @备注 @设置   tool comment lines (never rendered)
"""

DEFAULT_STYLES = """[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CN,思源宋体 Heavy,68,&H005F1FA8,&H000000FF,&H14FFFFFF,&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,2,10,10,65,1
Style: JP,FOT-ModeMinBLarge ProN H,50,&H005F1FA8,&H000000FF,&H00F0F0F0,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,2,1
Style: JP.Ruby,FOT-ModeMinBLarge ProN H,25,&H005F1FA8,&H000000FF,&H00F0F0F0,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
Style: CN.Top,思源宋体 Heavy,68,&H00FFFFFF,&H000000FF,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,8,10,10,5,1
Style: JP.Top,FOT-ModeMinBLarge ProN H,50,&H00FFFFFF,&H000000FF,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,8,10,10,85,1
Style: JP.Top.Ruby,FOT-ModeMinBLarge ProN H,25,&H00FFFFFF,&H000000FF,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
Style: CN.Alt,思源宋体 Heavy,68,&H00FFFFFF,&H000000FF,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,2,10,10,190,1
Style: JP.Alt,FOT-ModeMinBLarge ProN H,50,&H00FFFFFF,&H000000FF,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,130,1
Style: JP.Alt.Ruby,FOT-ModeMinBLarge ProN H,25,&H00FFFFFF,&H000000FF,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
Style: OP.CN,思源宋体 Heavy,80,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,2,10,10,85,1
Style: OP.JP,FOT-ModeMinBLarge ProN H,70,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,2,1
Style: OP.JP.Ruby,FOT-ModeMinBLarge ProN H,35,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,0,0,0,1
Style: ED.CN,思源宋体 Heavy,80,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,2,10,10,85,1
Style: ED.JP,FOT-ModeMinBLarge ProN H,70,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,2,1
Style: ED.JP.Ruby,FOT-ModeMinBLarge ProN H,35,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,0,0,0,1
Style: IN.CN,思源宋体 Heavy,80,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3.5,0,2,10,10,85,1
Style: IN.JP,FOT-ModeMinBLarge ProN H,70,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,2,1
Style: IN.JP.Ruby,FOT-ModeMinBLarge ProN H,35,&H00FFFFFF,&H00000000,&H005F1FA8,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,0,0,0,1
Style: CN.Gloss,思源宋体 Heavy,30,&H005F1FA8,&H000000FF,&H14FFFFFF,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
Style: JP.Gloss,FOT-ModeMinBLarge ProN H,25,&H005F1FA8,&H000000FF,&H00F0F0F0,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
Style: CN.Pinyin,Noto Sans CJK SC Black,26,&H005F1FA8,&H000000FF,&H14FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,0,0,0,1
"""


def default_styles_ass() -> str:
    """The default template (styles only) for a new project."""
    return DEFAULT_STYLES
