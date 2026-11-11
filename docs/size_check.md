# Size check: is a drawn 3 m wall really 3 m on site?

A test plan for `preview/site.html`, plus how to make the report. The target is 5 to 10 cm. Treat it as
unproven until this has been run on a real phone at a real site.

## What it checks

1. **Size:** a wall line drawn with the app, measured against a tape.
2. **Place:** the drawn wall's ends sit where they were drawn after you walk away and come back
   (anchoring), and after the design is saved and reopened.
3. **Every tag:** the same check using only one wall's tags at a time, so a badly placed tag shows up.

## Kit

- An Android phone with Chrome and Google Play Services for AR.
- 4 to 8 AprilTags (36h11) printed at 100 %. Check one with a ruler: the black square must be 160 mm
  (or the size you type on the start page). A tag printed 2 % small makes everything 2 % large.
- A tape measure or laser measure, and masking tape to mark points on the floor.

## Setup (about 5 minutes)

1. Tape two tags on the first wall, about 1.3 m up and 0.2 to 0.4 m apart. Tape one or two tags on each
   other wall.
2. Mark two floor points A and B **3.000 m apart** with masking tape (measure with the tape first), about
   1 to 1.5 m in front of the first wall.

## Steps

1. Open `site.html`, type the printed tag size, press **Start AR**.
2. Aim at the first wall's tags and press **Lock**. Aim at the floor until it is found.
3. Walk slowly round the space, looking at every wall's tags for a second or two. The HUD counts
   the mapped tags. Stop when every tag is mapped.
4. **Draw** tab, **Wall line**: tap A, tap B, **Done**. Set any height.
5. Select the wall. In the **File** tab type `3.000` in the tape box and press **Tape check**. The HUD
   shows the drawn length and the error.
6. Walk a full lap round the space and come back. Check by eye that the wall's ends are still on A and B.
   If they are not, measure the gap with the tape and note it.
7. Press **Save design**. Close the page, reopen it, open the saved file, aim at any mapped tag, and check the
   ends again.
8. Repeat steps 4 and 5 three times, each time standing at a different wall, so each wall's tags are used.
   Then draw a line about 5 m long, which tests scale over a longer distance.
9. Save the design and copy the `.json` file to the laptop.

## Report

```bash
python scripts/size_check.py site-design-*.json -o docs/size_check_report.md --tol-mm 50
```

To add a reading taken after the visit, pass `--tape <object id>=<metres>` (the id is in the saved file).
The script exits with code 1 if any check is outside the tolerance.

Also write down by hand:

| Check | Result |
|---|---|
| Tag size measured on paper (mm) | |
| Gap at A / B after a full lap (mm) | |
| Gap at A / B after save and reopen (mm) | |
| Phone model, Chrome version | |
| Light (sun, shade, indoor) | |

## How to read it

- **The error has the same sign every time** (for example always +2 %): the scale is wrong. Check the
  printed tag size first.
- **The error has mixed signs and is under 3 cm:** that is pose noise from one tag. Use two tags per wall.
- **The ends move after a lap but the length is right:** that is drift. Look for walls with no tag in view
  along the way.

## Emulator baseline (desktop, 2026-11-11)

These come from `site.html`'s phone emulator, a virtual phone with shake and tracking drift in a
5 x 5 m room with 8 tags. They are an upper bound on what the method can do, not a measurement:

| Check | Result |
|---|---|
| Drawn 3 m wall, length | 3.000 m (error under 1 mm) |
| Drawn wall ends against their true place | 7 to 25 mm |
| Same wall after save and reopen | 14 to 30 mm |
| 360 degree walk, 2 laps, 23.5 m, drift 10 mm/m + 0.2 deg/m | corrected max 36 / 32 mm per lap, raw (no correction) 48 / 27 mm |

The emulator cannot model a real phone's scale error (ARCore's metric scale), camera calibration,
motion blur, or real lighting on the tags. Those are exactly what this plan measures.
